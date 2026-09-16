import itertools
import math
import random
import unittest

from algorithms.private_risk_selection import risk, possible_honest_sets
from algorithms.private_paired_risk import (
    paired_sensitivity, worst_honest_risk, choose_under_bound,
    certify_paired_reports, private_paired_release_mps,
)


class PairedCertificateTests(unittest.TestCase):
    def test_replace_one_sensitivity_and_attaining_pair(self):
        n,k=1200,3
        # One record: all contrasts -1 -> +1, attainable for [0,1] losses.
        self.assertAlmostEqual(math.sqrt(k*(2/n)**2),paired_sensitivity(n,k))
        self.assertAlmostEqual(paired_sensitivity(n,k)/(4/n),math.sqrt(3)/2)

    def test_risk_of_difference_is_not_difference_of_risks(self):
        before=[0.,1.]; after=[1.,0.]
        self.assertAlmostEqual(risk(after,.5,.5)-risk(before,.5,.5),0)
        self.assertAlmostEqual(risk([1.,-1.],.5,.5),.5)

    def test_subadditive_bound_for_different_tail_memberships(self):
        rng=random.Random(160)
        for _ in range(200):
            before=[rng.random() for _ in range(10)]
            after=[rng.random() for _ in range(10)]
            delta=[a-b for a,b in zip(after,before)]
            self.assertLessEqual(risk(after)-risk(before),risk(delta)+1e-12)

    def test_top_n_minus_b_equals_exhaustive_all_sizes(self):
        rng=random.Random(17)
        for n in [4,7,10]:
            for b in range(min(3,n)):
                for _ in range(20):
                    xs=[rng.uniform(-1,1) for i in range(n)]
                    brute=max(risk([xs[i] for i in h]) for h in possible_honest_sets(n,b))
                    self.assertAlmostEqual(brute,worst_honest_risk(xs,b))

    def test_certificate_contains_actual_honest_risk_change(self):
        rng=random.Random(61)
        for hsize in [4,5,6]:
            for _ in range(50):
                losses=[[rng.random() for _ in range(4)] for i in range(6)]
                delta=[[x-losses[i][0] for x in losses[i][1:]] for i in range(6)]
                submitted=[row[:] for row in delta]
                for i in range(hsize,6): submitted[i]=[rng.uniform(-1,1) for _ in range(3)]
                d=certify_paired_reports(submitted,noise_stds=[0.]*6,max_releases=9,
                    failure_probability=.05,byzantine_bound=2)
                for k in range(3):
                    change=risk([losses[i][k+1] for i in range(hsize)])-risk([losses[i][0] for i in range(hsize)])
                    self.assertLessEqual(change,d['upper_bounds'][k]+1e-12)

    def test_indistinguishable_worlds_preclude_universal_progress(self):
        delta=[-.004]*8+[.01]*2
        before=[.5]*10; after=[.5+d for d in delta]
        ha=list(range(8)); hb=list(range(6))+[8,9]
        change_a=risk([after[i] for i in ha])-risk([before[i] for i in ha])
        change_b=risk([after[i] for i in hb])-risk([before[i] for i in hb])
        self.assertAlmostEqual(change_a,-.004)
        self.assertAlmostEqual(change_b,.00475)
        self.assertAlmostEqual(worst_honest_risk(delta,2),change_b)

    def test_damage_allowance_is_not_descent(self):
        self.assertEqual(choose_under_bound([.00475]),0)
        self.assertEqual(choose_under_bound([.00475],.005),1)
        self.assertEqual(choose_under_bound([0.]),0)
        self.assertEqual(choose_under_bound([-.01]),1)

    def test_paired_tail_certificate_has_structural_veto(self):
        # Even the most negative honest contrasts cannot overcome two +1s
        # for THIS subadditive certificate at n=10, b=2, mix=.5, tail=.2.
        # This is not an impossibility theorem for every possible selector.
        self.assertAlmostEqual(worst_honest_risk([-1.]*8+[1.]*2,2),.25)

    def test_contrast_truncation_needs_positive_bias_cost(self):
        values=[1.,-.5]; a=.1
        clipped=[min(a,max(-a,v)) for v in values]
        remainder=[max(v-a,0.) for v in values]
        self.assertNotEqual(sum(clipped)/2,min(a,max(-a,sum(values)/2)))
        self.assertLessEqual(sum(values)/2,sum(clipped)/2+sum(remainder)/2)
        self.assertGreater(sum(values)/2,sum(clipped)/2)

    def test_public_error_width_does_not_use_observed_variance(self):
        kw=dict(noise_stds=[.02]*10,max_releases=9,failure_probability=.05,byzantine_bound=2)
        one=certify_paired_reports([[0.,0.,0.]]*10,**kw)
        two=certify_paired_reports([[-.9,.9,.3]]*10,**kw)
        self.assertEqual(one['halfwidths'],two['halfwidths'])

    def test_malformed_report_cannot_create_narrow_interval(self):
        d=certify_paired_reports([[float('nan'),4.,-4.]]*10,noise_stds=[0.]*10,
            max_releases=9,failure_probability=.05,byzantine_bound=2)
        self.assertEqual(d['lower'][0],[-1.]*3)
        self.assertEqual(d['upper'][0],[1.]*3)

    def test_oracle_argument_is_not_accepted(self):
        with self.assertRaises(TypeError):
            certify_paired_reports([[0.]],noise_stds=[0.],max_releases=1,
                failure_probability=.05,byzantine_bound=0,honest_ids=[0])


class MPSPairedClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts.run_private_risk_feasibility import require_mps
        require_mps()

    def test_client_releases_same_example_contrasts_and_correct_noise(self):
        import torch
        from scripts.run_private_risk_feasibility import reseed
        losses=torch.tensor([[.7,.6,.4,.8],[.2,.3,.1,.3]],device='mps')
        reseed(177)
        out=private_paired_release_mps(losses,2.)
        reseed(177)
        expected=(losses[:,1:]-losses[:,:1]).mean(0)+2*paired_sensitivity(2,3)*torch.randn(3,device='mps')
        torch.testing.assert_close(out,expected)
        self.assertEqual(out.device.type,'mps')

    def test_bounded_loss_contract(self):
        import torch
        with self.assertRaises(ValueError):
            private_paired_release_mps(torch.tensor([[.5,1.5]],device='mps'),2.)


if __name__=='__main__': unittest.main()
