"""Verify the recovered publication and all generated code before any service starts."""
import hashlib
import json
from pathlib import Path

def verify_protocol_artifacts(root=None):
    root=Path(root) if root else Path(__file__).resolve().parents[1]
    m=json.loads((root/'dist/protocol_manifest.json').read_text())
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    if m['authority']!='proto-src/src/main/proto/dispatch.proto':raise ValueError('unexpected protocol authority')
    source=root/m['authority']
    if source.read_bytes()!=(root/'proto/dispatch.proto').read_bytes():raise ValueError('proto mirror mismatch')
    if sha(source)!=m['proto_sha256']:raise ValueError('source changed without regeneration')
    for path in ('dist/DRL-Dispatching-1.2.0.jar','anylogic/DRL-Dispatching-1.2.0.jar'):
        if sha(root/path)!=m['jar_sha256']:raise ValueError('stale JAR: '+path)
    for name,d in m['python_generated_sha256'].items():
        if sha(root/'python/generated'/name)!=d:raise ValueError('stale generated code: '+name)
    for name,d in m['anylogic_dependencies_sha256'].items():
        if sha(root/'anylogic'/name)!=d:raise ValueError('missing/changed AnyLogic dependency: '+name)
    return m
