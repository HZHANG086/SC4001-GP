import unittest
from sc4001_a.protocol import paired_splits, text_hash, clean_source

class ProtocolTests(unittest.TestCase):
    def test_nested_balance_and_no_cross_budget_leakage(self):
        labels = [0] * 5000 + [1] * 5000
        splits = paired_splits(labels, list(range(10000)), 13)
        all_train, all_val = set(), set()
        prev_train, prev_val = set(), set()
        for budget, split in splits.items():
            tr, va = split['train_indices'], split['validation_indices']
            self.assertEqual(len(tr), budget * 4 // 5)
            self.assertEqual(len(va), budget // 5)
            self.assertEqual(sum(labels[i] for i in tr), len(tr) // 2)
            self.assertEqual(sum(labels[i] for i in va), len(va) // 2)
            self.assertTrue(prev_train <= set(tr))
            self.assertTrue(prev_val <= set(va))
            prev_train, prev_val = set(tr), set(va)
            all_train.update(tr)
            all_val.update(va)
        self.assertFalse(all_train & all_val)
        self.assertEqual(splits, paired_splits(labels, list(range(10000)), 13))
        self.assertNotEqual(splits, paired_splits(labels, list(range(10000)), 21))

    def test_duplicate_overlap(self):
        self.assertEqual(text_hash('Hello<br /> WORLD &amp; cats'), text_hash('hello world & cats'))
        eligible, report = clean_source(['a', ' A ', 'b', 'c'], [' B '])
        self.assertEqual(eligible, [0, 3])
        self.assertEqual(report['train_test_overlap_rows_removed'], 1)
        self.assertEqual(report['duplicate_train_rows_removed'], 1)

if __name__ == '__main__':
    unittest.main()
