"""Certificate algebra plus actual MPS boundary checks (no CPU fallback)."""
import itertools
import math
import random
import unittest
import torch
from algorithms.private_risk_selection import cvar_upper, risk, possible_honest_sets, select_private_risk, forge_reports
from scripts.run_private_risk_feasibility import (
    config, privacy_plan, candidate_vectors, clip_cohort, flat_state, load_vector,
    local_evaluate, private_release, require_mps, derived_seed, datasets_for,
)
from models.registry import get_model
from privacy.local_dpsgd import private_gradient_release_fixed_without_replacement
from torch.utils.data import DataLoader, TensorDataset


class RiskAlgebraTests(unittest.TestCase):
    def test_fractional_cvar(self):
        self.assertAlmostEqual(cvar_upper([1., .5, 0.], .5), 5/6)
        self.assertAlmostEqual(cvar_upper([1., .5, 0.], 1), .5)
        self.assertAlmostEqual(cvar_upper([1., .5, 0.], .1), 1.)

    def test_possible_honest_sizes(self):
        sets = possible_honest_sets(10, 2)
        self.assertEqual(len(sets), 56)
        self.assertEqual({len(x) for x in sets}, {8, 9, 10})
        self.assertEqual(len(possible_honest_sets(10, 0)), 1)

    def test_exact_clean_descent_noop(self):
        reports = [[.6, .5, .4, .7]]*10
        decision = select_private_risk(reports, noise_stds=[0.]*10, max_releases=9,
            failure_probability=.05, byzantine_bound=0)
        self.assertEqual(decision["selected"], 2)
        self.assertEqual(decision["upper_bounds"][0], 0.)
        self.assertAlmostEqual(decision["upper_bounds"][2], -.2)
        worse = select_private_risk([[.4,.5,.6,.7]]*10, noise_stds=[0.]*10,
            max_releases=9, failure_probability=.05, byzantine_bound=0)
        self.assertEqual(worse["selected"], 0)

    def test_all_honest_subsets_covered_despite_false_reports(self):
        generator = random.Random(141)
        for hsize in (3,4,5):
            for _ in range(40):
                true = [[generator.random() for _ in range(4)] for i in range(5)]
                reports = [row[:] for row in true]
                for i in range(hsize,5): reports[i] = [generator.random() for _ in range(4)]
                result = select_private_risk(reports, noise_stds=[0.]*5, max_releases=1,
                    failure_probability=.05, byzantine_bound=2)
                baseline = risk([true[i][0] for i in range(hsize)])
                for k in range(4):
                    delta = risk([true[i][k] for i in range(hsize)])-baseline
                    self.assertLessEqual(delta, result["upper_bounds"][k]+1e-12)

    def test_veto_can_block_progress_not_false_success(self):
        reports = forge_reports([[.6,.4,.4,.4]]*10, identities=[0,1], mode="veto")
        result = select_private_risk(reports, noise_stds=[0.]*10, max_releases=1,
            failure_probability=.05, byzantine_bound=2)
        self.assertEqual(result["selected"], 0)

    def test_invalid_intervals_are_uninformative(self):
        reports = [[float("nan"), 5., -5., .5]]*10
        result = select_private_risk(reports, noise_stds=[.01]*10, max_releases=1,
            failure_probability=.05, byzantine_bound=2)
        self.assertEqual(result["lower"][0][:3], [0.,0.,0.])
        self.assertEqual(result["upper"][0][:3], [1.,1.,1.])
        self.assertTrue(all(map(math.isfinite,result["upper_bounds"])))

    def test_selector_rejects_oracle_argument(self):
        with self.assertRaises(TypeError):
            select_private_risk([[.5,.4]]*2, noise_stds=[0.,0.], max_releases=1,
                failure_probability=.05, byzantine_bound=0, true_risks=[.5,.4])

    def test_noise_query_streams_separate(self):
        self.assertNotEqual(derived_seed("secret","validation",8,"none",0),
                            derived_seed("secret","validation",8,"bf_x10",0))
        self.assertNotEqual(derived_seed("secret","gradient",8,0),
                            derived_seed("secret","validation",8,0))

    def test_privacy_counts_all_vector_releases(self):
        m=config(); p=privacy_plan(m)
        self.assertEqual(p["evaluation_releases"],9)
        self.assertAlmostEqual(p["evaluation_sensitivity"],2/1200)
        for row in p["ledgers"].values():
            self.assertLessEqual(row["gradient_epsilon"],3)
            self.assertLessEqual(row["evaluation_epsilon"],1)
            self.assertLessEqual(row["sequential_epsilon"],4)
            self.assertEqual(set(row["joint_ledger"]["channels"]), {"gradient","validation"})


class MPSBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): require_mps()

    def test_candidates_and_clipping_on_mps(self):
        x=torch.randn(10,17,device="mps")
        values=candidate_vectors(x,radius=2.,alpha=.1)
        self.assertTrue(all(v.device.type=="mps" for v in values))
        torch.testing.assert_close(values[1],clip_cohort(x,2.).mean(0))
        self.assertLessEqual(float(torch.linalg.vector_norm(values[3])),2.00001)

    def test_gradient_does_not_mutate_model(self):
        model=get_model("lenet5_tanh","fashionmnist").to("mps")
        x=torch.randn(4,1,28,28,device="mps").cpu(); y=torch.tensor([0,1,2,3])
        loader=DataLoader(TensorDataset(x,y),batch_size=4)
        before=flat_state(model.state_dict()).clone()
        release,stats=private_gradient_release_fixed_without_replacement(model,loader,
            device="mps",batch_size=4,clip_norm=1.,noise_multiplier=0.,return_noise_free_oracle=False)
        self.assertTrue(torch.equal(before,flat_state(model.state_dict())))
        self.assertIsNone(stats.noise_free_delta_oracle)
        self.assertLessEqual(float(torch.linalg.vector_norm(flat_state(release))),1.00001)

    def test_brier_and_model_restore(self):
        model=get_model("lenet5_tanh","fashionmnist").to("mps")
        baseline=flat_state(model.state_dict()).clone()
        loader=DataLoader(TensorDataset(torch.randn(4,1,28,28,device="mps").cpu(),torch.tensor([0,1,2,3])),batch_size=4)
        result=local_evaluate(model,[loader])[0]
        self.assertTrue(0 <= result["brier"] <= 1)
        load_vector(model,baseline+.01); load_vector(model,baseline)
        self.assertTrue(torch.equal(baseline,flat_state(model.state_dict())))

    def test_local_reports_repeat_only_same_query(self):
        levels=[[.5,.5,.5,.5]]*10; stds=[.03]*10
        one=private_release(levels,stds,"test_secret",8,"none")
        two=private_release(levels,stds,"test_secret",8,"none")
        three=private_release(levels,stds,"test_secret",8,"bf_x10")
        self.assertEqual(one,two); self.assertNotEqual(one,three)

    def test_real_split_sizes(self):
        train,val,test,audit=datasets_for(config(),934001)
        self.assertEqual({len(loader.dataset) for loader in train},{4800})
        self.assertEqual({len(loader.dataset) for loader in val},{1200})
        self.assertEqual(sum(len(loader.dataset) for loader in test),10000)
        self.assertTrue(all(row["disjoint"] for row in audit))


if __name__ == "__main__": unittest.main()
