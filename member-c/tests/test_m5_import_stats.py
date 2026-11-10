import unittest
import tempfile
from pathlib import Path
import pandas as pd
from sc4001_c.m5_import import recompute_comparisons,verify_accepted_return,MANIFEST
from sc4001_c.common import dump,digest


class ImportStatisticsTests(unittest.TestCase):
    def fixture(self):
        spec={'budgets':[400,1600,6400],'seeds':[13,21,42,87,100]}
        selected=pd.DataFrame([{'budget':b,'seed':s,'ood_accuracy':.5+(i+1)/100}
            for b in spec['budgets'] for i,s in enumerate(spec['seeds'])])
        old=pd.DataFrame([{'method':m,'budget':b,'seed':s,'ood_accuracy':.5}
            for m in ('M1','M2','M3','M4','M6','L2SP')
            for b in spec['budgets'] for s in spec['seeds']])
        return selected,old,spec

    def test_known_paired_effect_and_exact_test_family(self):
        selected,old,spec=self.fixture()
        result=recompute_comparisons(selected,old,spec)
        self.assertEqual(len(result),18)
        self.assertTrue((result.n_seeds==5).all())
        self.assertTrue((result.wins==5).all())
        self.assertTrue((result.p_signflip_exact==.0625).all())
        self.assertTrue((result.p_signflip_exact_holm==1.).all())
        for value in result['mean']:self.assertAlmostEqual(value,3.)

    def test_duplicate_or_missing_seed_cannot_become_a_replicate(self):
        selected,old,spec=self.fixture()
        with self.assertRaisesRegex(ValueError,'M5 statistical replication grid'):
            recompute_comparisons(pd.concat([selected,selected.iloc[:1]]),old,spec)
        with self.assertRaisesRegex(ValueError,'reference statistical replication grid'):
            recompute_comparisons(selected,old.iloc[1:],spec)

    def test_synthesis_rejects_changed_cka_or_another_return_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            raw=Path(folder);name='member-c/results/M5-CKA/cka.csv'
            path=raw/name;path.parent.mkdir(parents=True);path.write_text('synthetic fixture\n')
            dump(raw/MANIFEST,{'files':{name:digest(path)}})
            audit={'return_manifest_sha256':digest(raw/MANIFEST)}
            verify_accepted_return(raw,audit)
            path.write_text('changed diagnostic value\n')
            with self.assertRaisesRegex(ValueError,'return file changed'):
                verify_accepted_return(raw,audit)
            dump(raw/MANIFEST,{'files':{name:digest(path)}})
            with self.assertRaisesRegex(ValueError,'differs from independently accepted'):
                verify_accepted_return(raw,audit)


if __name__=='__main__':unittest.main()
