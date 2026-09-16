import copy
from fractions import Fraction as Q
import unittest
from scripts.audit_v28_exact_count_gate import exact_metrics, exact_decision, SEEDS, METHODS, KEYS


class ExactGateTests(unittest.TestCase):
    def table(self):
        baseline = dict(zip(KEYS, map(Q, (80, 60, 20, 80))))
        primary = dict(zip(KEYS, map(Q, (79, 61, 19, 70))))
        current = {(s, m): dict(primary if m == 'risk_rfa' else baseline) for s in SEEDS for m in METHODS}
        old = {(s, m): dict(baseline) for s in SEEDS for m in ('erm_mean', 'erm_rfa')}
        return current, old

    def test_exact_integer_count_statistics(self):
        rows = [dict(N=1200, class_count=[1200]+[0]*9, class_hits=[120*i]+[0]*9) for i in range(10)]
        stats = exact_metrics(rows)
        self.assertEqual(stats['accuracy_pct'], 45)
        self.assertEqual(stats['worst20_pct'], 5)
        self.assertEqual(stats['gap_best20_worst20_pp'], 80)
        self.assertEqual(stats['variance_pp2'], 825)

    def test_boundary_is_exact_and_one_count_below_fails(self):
        current, old = self.table()
        self.assertTrue(all(r['passed'] for r in exact_decision(current, old)))
        current[SEEDS[-1], 'risk_rfa']['worst20_pct'] -= Q(1, 24)
        self.assertTrue(all(not r['passed'] for r in exact_decision(current, old)))

    def test_accuracy_boundary_one_count_below_fails(self):
        current, old = self.table()
        current[SEEDS[-1], 'risk_rfa']['accuracy_pct'] -= Q(1, 120)
        self.assertTrue(all(not r['passed'] for r in exact_decision(current, old)))

    def test_variance_and_gap_cannot_be_skipped(self):
        for key in ('variance_pp2', 'gap_best20_worst20_pp'):
            current, old = self.table()
            current[SEEDS[-1], 'risk_rfa'][key] = Q(1000)
            self.assertTrue(all(not r['passed'] for r in exact_decision(current, old)))

    def test_incomplete_matrix_and_primary_switch_rejected(self):
        current, old = self.table()
        missing = copy.deepcopy(current); del missing[SEEDS[0], 'risk_mean']
        with self.assertRaises(ValueError):
            exact_decision(missing, old)
        current[SEEDS[-1], 'risk_rfa']['worst20_pct'] = Q(60)
        current[SEEDS[-1], 'risk_mean']['worst20_pct'] = Q(100)
        self.assertFalse(all(r['passed'] for r in exact_decision(current, old)))
        with self.assertRaises(ValueError):
            exact_decision(current, {})

    def test_invalid_counts_rejected(self):
        rows = [dict(N=1200, class_count=[1200]+[0]*9, class_hits=[600]+[0]*9) for _ in range(10)]
        rows[0]['class_hits'][0] = 1201
        with self.assertRaises(ValueError):
            exact_metrics(rows)


if __name__ == '__main__':
    unittest.main()
