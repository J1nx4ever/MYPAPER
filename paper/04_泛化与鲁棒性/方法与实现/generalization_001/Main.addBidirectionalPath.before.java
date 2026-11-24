// 正向：卸载点 / crusher / dump → loadSite
PathSegment forwardSeg = new PathSegment(
    this,
    positiveFrom,
    positiveTo,
    positiveFrom,
    positiveTo,
    speedLimit
);
// 反向：loadSite → 卸载点 / crusher / dump
PathSegment reverseSeg = new PathSegment(
    this,
    positiveTo,
    positiveFrom,
    positiveFrom,
    positiveTo,
    speedLimit
);

segmentMap.put(forwardSeg.key, forwardSeg);
segmentMap.put(reverseSeg.key, reverseSeg);

traceln("Add path: " 
    + forwardSeg.key + " dir=" + forwardSeg.directionName 
    + ", " 
    + reverseSeg.key + " dir=" + reverseSeg.directionName);
