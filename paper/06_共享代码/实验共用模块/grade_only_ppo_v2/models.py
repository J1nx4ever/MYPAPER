"""Use the frozen v1 1519D Actor and one linear material-memory Critic."""
from grade_only_ppo_v1.models import Actor, MaterialCostCritic, digest

__all__ = ['Actor', 'MaterialCostCritic', 'digest']
