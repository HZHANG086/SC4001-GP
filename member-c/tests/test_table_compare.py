import tempfile
import unittest
from pathlib import Path
from sc4001_c.table_compare import compare_tables


class TableComparisonTests(unittest.TestCase):
    def check(self, actual):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            header = 'method,n_seeds,accuracy,p_signflip_exact,p_t_approx\n'
            (root/'reference.csv').write_text(header+'M6,5,0.9,0.125,0.48609472077393484\n')
            (root/'actual.csv').write_text(header+actual+'\n')
            return compare_tables(root/'reference.csv', root/'actual.csv')

    def test_observed_linux_roundoff_is_recorded(self):
        result = self.check('M6,5,0.9,0.125,0.4860947207739349')
        self.assertTrue(result['equivalent'])
        self.assertEqual(len(result['roundoff_differences']), 1)
        self.assertEqual(result['roundoff_differences'][0]['column'], 'p_t_approx')

    def test_real_statistical_changes_and_nonfinite_values_fail(self):
        for value in ('0.486095', 'nan', 'inf'):
            with self.subTest(value=value):
                self.assertFalse(self.check('M6,5,0.9,0.125,'+value)['equivalent'])

    def test_exact_predictions_counts_and_randomization_fields_remain_strict(self):
        for row in ('M6,5,0.9000000000000001,0.125,0.48609472077393484',
                    'M6,5.0,0.9,0.125,0.48609472077393484',
                    'M6,5,0.9,0.1250000000000001,0.48609472077393484',
                    'M3,5,0.9,0.125,0.48609472077393484',
                    'M6,5,0.9,0.125,0.48609472077393484,unexpected'):
            with self.subTest(row=row):
                self.assertFalse(self.check(row)['equivalent'])


if __name__ == '__main__':
    unittest.main()
