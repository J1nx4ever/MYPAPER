"""Bind the weak GAT variants to the sealed mixed-training runtime."""

from contextlib import contextmanager
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from business_graph_summary_campaign_v1.plan_io import sha
from business_graph_summary_v1.contract import require
from business_graph_weak_ablation_v1 import models


STRUCTURAL_BRIDGE = Path(__file__).resolve().parents[1] / 'structural_ablation_001/ablation_runtime.py'
STRUCTURAL_BRIDGE_SHA = 'fb6e5f51cb42f897270b06df80b18f8b55b16e8e1ab34f1e847761ed4cf6d063'


def _load_structural_bridge():
    require(sha(STRUCTURAL_BRIDGE) == STRUCTURAL_BRIDGE_SHA, 'sealed structural runtime source differs')
    spec = spec_from_file_location('gat_weak_structural_runtime', STRUCTURAL_BRIDGE)
    bridge = module_from_spec(spec)
    spec.loader.exec_module(bridge)
    return bridge


_bridge = _load_structural_bridge()


def source_files():
    model_sources = sorted(path for path in Path(models.__file__).resolve().parent.glob('*.py')
                           if not path.name.startswith('test_'))
    return (Path(__file__).resolve(), *_bridge.source_files(), *model_sources)


_learner = _bridge._with_globals(_bridge._learner, models=models, source_files=source_files)
# Keep the inherited runtime/admission hooks and their cleanup. Only its private
# model and source-inventory closures vary, so separate arms cannot share state.
runtime_scope = contextmanager(_bridge._with_globals(_bridge.runtime_scope.__wrapped__, _learner=_learner))
