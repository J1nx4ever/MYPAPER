    private int outageSite, stopSec, resumeSec, outagePhase, nominalCapacity;
    private final Map<String,Object> samples = new java.util.LinkedHashMap<>();
    private final Map<Truck,Double> suspended = new java.util.LinkedHashMap<>();
    private double frozenReserve, frozenProduction, frozenMetal, frozenCursor, frozenRemaining;

    private void readOutage() {
        JsonObject plan = config.getAsJsonObject("outage");
        outageSite = plan.get("site_id").getAsBigDecimal().intValueExact();
        stopSec = plan.get("start_seconds").getAsBigDecimal().intValueExact();
        resumeSec = plan.get("end_seconds").getAsBigDecimal().intValueExact();
        require(outageSite == 0 && stopSec > 0 && stopSec < resumeSec && resumeSec < 28800
                && plan.size() == 3, "Invalid high-grade outage");
    }

    private Map<String,Object> physicalOutagePlan() {
        LoadSite site = main.getLoadSiteById(outageSite);
        require(site != null && site.isOre && Math.abs(site.grade - 0.32) < 1e-12
                && site.shovelCapacity == 1 && wireId(site.loadNode) == 5, "Wrong outage site");
        for (int i=0; i<4; i++) {
            LoadSite other = main.getLoadSiteById(i);
            require(i == outageSite || !other.isOre || site.grade > other.grade,
                    "Outage site must be the uniquely highest grade ore loader");
        }
        return fields("site_id",site.siteId,"start_seconds",stopSec,"end_seconds",resumeSec);
    }

    private Map<String,Object> outageSample(LoadSite site) {
        List<Integer> inbound = new ArrayList<>(), queue = new ArrayList<>(), seize = new ArrayList<>();
        List<Object> service = new ArrayList<>();
        for (Truck t : main.trucks) if (t.incomingLoadSiteId == outageSite) inbound.add(t.truckId);
        for (int i=0; i<site.queueLoad.size(); i++) queue.add(site.queueLoad.get(i).truckId);
        for (int i=0; i<site.seize.size(); i++) seize.add(site.seize.get(i).truckId);
        for (int i=0; i<site.delay.size(); i++) {
            Truck t = site.delay.get(i);
            service.add(fields("truck_id",t.truckId,"remaining_seconds",site.delay.getRemainingTime(t,SECOND),
                               "suspended",site.delay.isSuspended(t)));
        }
        return fields("time_seconds",now(),"event_cursor",main.boundaryEventRecorder.outageCursor(),
                      "available",site.available,"resource_capacity",site.shovelResource.capacity,
                      "inbound_ids",inbound,"queue_ids",queue,"seize_ids",seize,"service",service,
                      "reserved_tons",site.reservedTons,"remaining_tons",site.remainingTons,
                      "produced_tons",site.producedTons,"produced_metal_tons",site.producedMetalTons,
                      "extraction_cursor_tons",site.extractionCursorTons);
    }

    public void syncOutage() {
        double t = now();
        if (outagePhase == 0 && t + 1e-8 >= stopSec) {
            require(Math.abs(t - stopSec) < 1e-7, "Late loader-stop event");
            LoadSite s = main.getLoadSiteById(outageSite);
            samples.put("stop_before",outageSample(s));
            nominalCapacity = s.shovelCapacity;
            require(nominalCapacity == 1 && s.shovelResource.capacity == 1 && s.available,
                    "High-grade loader was not operational before the planned outage");
            frozenReserve = s.reservedTons;
            frozenRemaining = s.remainingTons;
            frozenProduction = s.producedTons;
            frozenMetal = s.producedMetalTons;
            frozenCursor = s.extractionCursorTons;
            // Make reentrant dispatch callbacks see the closed state before touching process blocks.
            outagePhase = 1;
            s.available = false;
            s.shovelCapacity = 0;
            for (int i=0; i<s.delay.size(); i++) {
                Truck truck = s.delay.get(i);
                double residual = s.delay.getRemainingTime(truck, SECOND);
                require(residual >= 0 && Double.isFinite(residual), "Invalid service remainder");
                suspended.put(truck,residual);
            }
            for (Truck truck : suspended.keySet()) {
                require(!s.delay.isSuspended(truck) && s.delay.suspend(truck) == truck
                        && s.delay.isSuspended(truck), "Native loading timer was not suspended");
            }
            s.shovelResource.set_capacity(0);
            samples.put("stop_after",outageSample(s));
        }
        if (outagePhase == 1 && t + 1e-8 >= resumeSec) {
            require(Math.abs(t - resumeSec) < 1e-7, "Late loader-recovery event");
            LoadSite s = main.getLoadSiteById(outageSite);
            samples.put("resume_before",outageSample(s));
            require(!s.available && s.shovelResource.capacity == 0 && s.delay.size() == suspended.size()
                    && s.producedTons == frozenProduction && s.producedMetalTons == frozenMetal
                    && s.reservedTons == frozenReserve && s.remainingTons == frozenRemaining
                    && s.extractionCursorTons == frozenCursor, "Loader moved material during outage");
            for (var entry : suspended.entrySet()) {
                require(s.delay.isSuspended(entry.getKey())
                        && Math.abs(s.delay.getRemainingTime(entry.getKey(), SECOND) - entry.getValue()) < 1e-6,
                        "Suspended loading time was consumed during the outage");
            }
            outagePhase = 2;
            s.available = true;
            s.shovelCapacity = nominalCapacity;
            for (Truck truck : suspended.keySet())
                require(s.delay.resume(truck) == truck && !s.delay.isSuspended(truck), "Service did not resume");
            s.shovelResource.set_capacity(nominalCapacity);
            samples.put("resume_after",outageSample(s));
        }
    }

    private String publishOutage() {
        require(outagePhase == 2 && samples.size() == 4 && main.getLoadSiteById(outageSite).available,
                "Incomplete high-grade loader intervention");
        Map<String,Object> receipt = fields("schema_version",1,"recording_id",main.episodeId,
                "run_index",main.experimentRunIndex,"environment_seed",main.experimentSeed,
                "config_sha256",configSha,"checkpoint_sha256",checkpointSha,"site_id",outageSite,
                "start_seconds",stopSec,"end_seconds",resumeSec);
        receipt.putAll(samples);
        Path out = Path.of(text("event_directory")).resolve("outage_evidence.json");
        try {
            byte[] bytes = (new Gson().toJson(receipt) + "\n").getBytes(java.nio.charset.StandardCharsets.UTF_8);
            Files.write(out, bytes, java.nio.file.StandardOpenOption.CREATE_NEW, java.nio.file.StandardOpenOption.WRITE);
            return hash(bytes);
        } catch(IOException ex) {
            throw new UncheckedIOException("Cannot publish native outage evidence", ex);
        }
    }

