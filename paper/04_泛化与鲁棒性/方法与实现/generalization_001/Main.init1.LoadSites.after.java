LoadSite s0 = add_loadSites();
s0.setLocation(ShovelOreHigh1);
s0.loadNode = ShovelOreHigh1;
s0.gradeHalfRange = Baseline1CampaignPlan.loadGradeHalfRange(experimentRunIndex, 0);
s0.initLoadSite(0, "HG1", true,
    Baseline1CampaignPlan.loadGrade(experimentRunIndex, 0), 30000, 0.45,
    ShovelOreHigh1.getX(), ShovelOreHigh1.getY(), 13.5, 2.2);


LoadSite s1 = add_loadSites();
s1.setLocation(ShovelOreHigh2);
s1.loadNode = ShovelOreHigh2;
s1.gradeHalfRange = Baseline1CampaignPlan.loadGradeHalfRange(experimentRunIndex, 1);
s1.initLoadSite(1, "HG2", true,
    Baseline1CampaignPlan.loadGrade(experimentRunIndex, 1), 30000, 0.45,
    ShovelOreHigh2.getX(), ShovelOreHigh2.getY(), 13.5, 2.2);


LoadSite s2 = add_loadSites();
s2.setLocation(ShovelOreLow);
s2.loadNode = ShovelOreLow;
s2.gradeHalfRange = Baseline1CampaignPlan.loadGradeHalfRange(experimentRunIndex, 2);
s2.initLoadSite(2, "LG", true,
    Baseline1CampaignPlan.loadGrade(experimentRunIndex, 2), 30000, 0.45,
    ShovelOreLow.getX(), ShovelOreLow.getY(), 13.5, 2.2);
