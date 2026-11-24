"""Reuse frozen physical and scalar-MC evaluation for both smoke candidates.

The unchanged evaluator predicts costs through model.predict_values(episode),
then checks all model/optimizer/counter/RNG state under its read-only guard.
It reports production and waiting independently of the grade-only objective.
The inherited format version describes data fields, not the advantage method;
the v2 runner must bind the arm, execution profile and checkpoint externally.
"""
from grade_only_ppo_v1.evaluation import (
    FORMAT,
    decision_sampled_queue_statistics,
    evaluate_episodes,
)

__all__ = ['FORMAT', 'decision_sampled_queue_statistics', 'evaluate_episodes']
