LoadSite s0 = add_loadSites();
s0.setLocation(ShovelOreHigh1);
s0.loadNode = ShovelOreHigh1;
s0.initLoadSite(0, "HG1", true, 0.38, 30000, 0.45,
    ShovelOreHigh1.getX(), ShovelOreHigh1.getY(), 13.5, 2.2);


LoadSite s1 = add_loadSites();
s1.setLocation(ShovelOreHigh2);
s1.loadNode = ShovelOreHigh2;
s1.initLoadSite(1, "HG2", true, 0.30, 30000, 0.45,
    ShovelOreHigh2.getX(), ShovelOreHigh2.getY(), 13.5, 2.2);


LoadSite s2 = add_loadSites();
s2.setLocation(ShovelOreLow);
s2.loadNode = ShovelOreLow;
s2.initLoadSite(2, "LG", true, 0.15, 30000, 0.45,
    ShovelOreLow.getX(), ShovelOreLow.getY(), 13.5, 2.2);
