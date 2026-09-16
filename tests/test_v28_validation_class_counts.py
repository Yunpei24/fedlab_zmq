import unittest
from scripts.analyze_v28_validation_class_counts import summarize


class CountAuditTests(unittest.TestCase):
    def test_different_weightings(self):
        rows = [dict(N=10, class_count=[9, 1], class_hits=[9, 0]),
                dict(N=10, class_count=[1, 9], class_hits=[0, 0])]
        s = summarize(rows)
        self.assertEqual(s['client_accuracy_pct'], 45)
        self.assertEqual(s['mean_client_balanced_accuracy_pct'], 25)
        self.assertEqual(s['pooled_class_balanced_accuracy_pct'], 45)
        self.assertEqual(s['worst20_pct'], 0)

    def test_absent_class_not_zero_recall(self):
        s = summarize([dict(N=2, class_count=[2, 0], class_hits=[1, 0])])
        self.assertEqual(s['mean_client_balanced_accuracy_pct'], 50)
        self.assertEqual(s['pooled_class_balanced_accuracy_pct'], 50)
        self.assertEqual(s['class_recall_pct'], [50, None])

    def test_invalid_counts(self):
        for row in [dict(N=2, class_count=[1, 1], class_hits=[2, 0]),
                    dict(N=3, class_count=[1, 1], class_hits=[1, 1]),
                    dict(N=2, class_count=[1, 1], class_hits=[float('nan'), 0])]:
            with self.assertRaises(ValueError):
                summarize([row])


if __name__ == '__main__':
    unittest.main()
