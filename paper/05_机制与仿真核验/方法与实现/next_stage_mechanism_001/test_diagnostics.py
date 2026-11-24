"""Meaningful boundary/soundness checks for new diagnostic mathematics."""
import unittest
import numpy as np
from replay import envelope
from late_updates import policy_metrics

class Diagnostics(unittest.TestCase):
    def test_forced_fifo_blocks_supernatural_recovery_in_both_directions(self):
        low=envelope(4000,1040,[dict(remaining_tons=500,grade=.1)],100)
        high=envelope(4000,1120,[dict(remaining_tons=500,grade=.4)],100)
        self.assertTrue(low['excluded']);self.assertLess(low['upper'],.26)
        self.assertTrue(high['excluded']);self.assertGreater(high['lower'],.28)

    def test_feasible_extensions_are_inside_relaxed_envelope(self):
        fifo=[dict(remaining_tons=70,grade=.17),dict(remaining_tons=50,grade=.39)]
        b=envelope(1000,270,fifo,300)
        q=1120;m=270+70*.17+50*.39
        for extra in np.linspace(0,180,31):
            for g in np.linspace(0,1,31):
                v=(m+extra*g)/(q+extra)
                self.assertGreaterEqual(v,b['lower']);self.assertLessEqual(v,b['upper'])
        self.assertFalse(b['excluded'])

    def test_narrow_grade_near_boundary_is_not_over_certified(self):
        b=envelope(4100,4100*.25999,[],0)
        self.assertFalse(b['excluded'])

    def test_kl_ignores_illegal_zero_support_and_detects_strategy_change(self):
        p=np.array([[.5,.5,0.,0.],[1.,0.,0.,0.]])
        same=policy_metrics(p,p)
        self.assertEqual(same['exact_categorical_kl_mean'],0.)
        q=np.array([[.9,.1,0.,0.],[1.,0.,0.,0.]])
        r=policy_metrics(p,q)
        self.assertGreater(r['exact_categorical_kl_mean'],0)
        self.assertAlmostEqual(r['total_variation_mean'],.2)

if __name__=='__main__':unittest.main()
