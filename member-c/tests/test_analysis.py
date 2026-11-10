import itertools
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy import stats
from sc4001_c.stats import paired_stats,holm,probability_metrics
from sc4001_c.cka import linear_cka,select_samples,compare_features
from sc4001_c.common import read_predictions
from sc4001_c.bootstrap import crossed_bootstrap


class StatisticalTests(unittest.TestCase):
    def test_exact_pairing_matches_scipy(self):
        left=np.array([.92,.91,.95,.88,.94]); right=np.array([.93,.90,.94,.92,.95])
        actual=paired_stats(left-right)
        reference=stats.permutation_test((left,right),lambda a,b:np.mean(a-b),
                    permutation_type='samples',n_resamples=np.inf,alternative='two-sided')
        self.assertAlmostEqual(actual['p_signflip_exact'],reference.pvalue)
        t=stats.ttest_rel(left,right)
        self.assertAlmostEqual(actual['ci95_low'],t.confidence_interval().low)
        self.assertAlmostEqual(actual['ci95_high'],t.confidence_interval().high)

    def test_small_n_discreteness_and_ties(self):
        self.assertEqual(paired_stats([1,2,3,4,5])['p_signflip_exact'],.0625)
        self.assertEqual(paired_stats([0,0,0,0,0])['p_signflip_exact'],1)
        self.assertEqual(paired_stats([0,0,0,0,0])['p_t_approx'],1)

    def test_holm_known_result(self):
        np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])

    def test_probabilities(self):
        m=probability_metrics([0,1,0,1],[0,1,0,1],[0,1,0,1])
        for metric in ('accuracy','macro_f1','balanced_accuracy'):
            self.assertEqual(m[metric],1)
        self.assertEqual(m['brier'],0)
        self.assertEqual(m['ece15'],0)
        self.assertTrue(np.isfinite(m['nll']))

    def test_crossed_bootstrap_extremes_and_reproducibility(self):
        y=np.array([0,0,1,1])
        m=np.ones((5,4),dtype=int)
        result=crossed_bootstrap([m,m],[y,y],repetitions=100)
        self.assertEqual(result['crossed_seeds_and_examples_low'],100)
        self.assertEqual(result['conditional_fixed_models_high'],100)
        m[0,:]=-1
        a=crossed_bootstrap([m],[y],repetitions=100)
        self.assertEqual(a,crossed_bootstrap([m],[y],repetitions=100))


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(3)
        self.x=self.rng.normal(size=(40,9)); self.y=self.rng.normal(size=(40,7))

    def test_feature_space_formula(self):
        x=self.x-self.x.mean(0); y=self.y-self.y.mean(0)
        expected=np.linalg.norm(x.T@y)**2/(np.linalg.norm(x.T@x)*np.linalg.norm(y.T@y))
        self.assertAlmostEqual(linear_cka(x,y),expected,places=12)

    def test_invariances(self):
        rotation=np.linalg.qr(self.rng.normal(size=(9,9)))[0]
        for debias in (False,True):
            self.assertAlmostEqual(linear_cka(self.x,7*self.x@rotation+4,debias),1,places=10)
            self.assertAlmostEqual(linear_cka(self.x,self.y,debias),linear_cka(self.y,self.x,debias),places=12)

    def test_zero_variance_is_missing_not_perfect(self):
        self.assertIsNone(linear_cka(np.ones((40,9)),self.y))
        features={p:[np.ones((40,9))] for p in ('cls','mean')}
        result=compare_features(features,features,{'method':'test'})
        self.assertEqual(result[0]['status'],'degenerate_zero_variance')

    def test_pair_alignment_matters(self):
        self.assertLess(linear_cka(self.x,self.x[::-1]),.8)

    def test_balanced_sampling_is_order_independent(self):
        from datasets import Dataset
        rows=[{'example_id':f'x:{i}','label':i%2,'text':str(i)} for i in range(30)]
        a=select_samples(Dataset.from_list(rows),12)
        b=select_samples(Dataset.from_list(rows[::-1]),12)
        self.assertEqual(a,b)
        self.assertEqual(sum(r['label'] for r in a),6)


class PredictionTests(unittest.TestCase):
    def test_aligns_by_id_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'p.jsonl'
            rows=[{'example_id':'b','label':1,'prediction':1,'p_positive':.8},
                  {'example_id':'a','label':0,'prediction':0,'p_positive':.2}]
            p.write_text('\n'.join(json.dumps(r) for r in rows))
            ids,y,pred,prob=read_predictions(p,['a','b'],np.array([0,1]))
            self.assertEqual(prob.tolist(),[.2,.8])
            with self.assertRaises(ValueError): read_predictions(p,['a','missing'])
            rows[0]['prediction']=0
            p.write_text('\n'.join(json.dumps(r) for r in rows))
            with self.assertRaises(ValueError): read_predictions(p)


if __name__=='__main__': unittest.main()
