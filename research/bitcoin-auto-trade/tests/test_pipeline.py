from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import io
import unittest

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from threadpoolctl import threadpool_limits

from trading.decisions import probability_hysteresis
from trading.experiment_config import load_experiment,FeatureSpec,ObservationSpec,ModelSpec
from trading.feature_engine import build_feature_set
from trading.hmm import HMMLabeler
from trading.pipeline import run_experiment,_reference
from trading.regime_diagnostics import comparison_metrics
from trading.posterior import make_cached_labeler
from trading.research_cache import atomic_json,commit_manifest,read_manifest


def sample_bars(n=200):
    rng=np.random.default_rng(6)
    drift=np.resize(np.repeat([-.01,0.,.012],15),n)
    closing=100*np.exp(np.cumsum(drift+rng.normal(0,.003,n)))
    opening=np.r_[closing[0],closing[:-1]]
    return pd.DataFrame(dict(open=opening,close=closing,high=np.maximum(opening,closing)*1.01,
        low=np.minimum(opening,closing)*.99,volume=rng.uniform(100.,200.,n),
        taker_buy_volume=rng.uniform(40.,80.,n)),
        index=pd.date_range("2020-01-01",periods=n,freq="D",tz="UTC",name="datetime"))


class DecisionTests(unittest.TestCase):
    def test_evidence_threshold_and_gap_reset(self):
        p=pd.DataFrame([[.1,.1,.8],[.5,.05,.45],[.8,.05,.15],[np.nan]*3,[.1,.8,.1]],columns=['bear','side','bull'])
        labels,audit=probability_hysteresis(p,4)
        self.assertEqual(labels.tolist(),[2,2,0,-1,1])
        self.assertEqual(audit.switched.tolist(),[False,False,True,False])
        weak,_=probability_hysteresis(p,1)
        self.assertEqual(weak.tolist(),[2,0,0,-1,1])

    def test_weak_threshold_matches_argmax_and_uses_only_prefix(self):
        p=pd.DataFrame([[.7,.2,.1],[.05,.9,.05],[.1,.2,.7]],columns=['bear','side','bull'])
        labels,_=probability_hysteresis(p,1)
        np.testing.assert_array_equal(labels,p.to_numpy().argmax(axis=1))
        p.iloc[2]=[.99,.005,.005]
        changed,_=probability_hysteresis(p,1)
        np.testing.assert_array_equal(labels[:2],changed[:2])
        with self.assertRaises(ValueError): probability_hysteresis(p,.5)

    def test_endpoint_posterior_equals_manual_forward_filter(self):
        model=GaussianHMM(n_components=3,covariance_type='diag',init_params='')
        model.startprob_=np.array([.2,.3,.5])
        model.transmat_=np.array([[.9,.05,.05],[.1,.8,.1],[.05,.05,.9]])
        model.means_=np.array([[-2.],[0.],[2.]])
        model.covars_=np.ones((3,1))
        x=np.array([[-1.],[.2],[.5],[1.7]])
        emission=np.exp(model._compute_log_likelihood(x))
        alpha=model.startprob_*emission[0]
        alpha/=alpha.sum()
        for row in emission[1:]:
            alpha=(alpha@model.transmat_)*row
            alpha/=alpha.sum()
        np.testing.assert_allclose(model.predict_proba(x)[-1],alpha,atol=1e-12)


class FeatureAndCacheTests(unittest.TestCase):

    def test_numeric_fit_snapshot_preserves_inference_without_refitting(self):
        with TemporaryDirectory() as temporary,threadpool_limits(limits=1):
            b=sample_bars(120)
            spec=ObservationSpec('x',tuple(FeatureSpec(k,5) for k in ['log_return','efficiency','taker_imbalance']))
            x=build_feature_set(b,spec).iloc[10:100].to_numpy()
            y=np.log(b.close/b.close.shift(10)).iloc[10:100].to_numpy()
            model=ModelSpec(train_window=90,restarts=2,iterations=100,seeds=(42,))
            first=make_cached_labeler(model,42,temporary,{})
            scaler,fit,mapping,_=first._fit(x,y)
            second=make_cached_labeler(model,42,temporary,{})
            with patch.object(HMMLabeler,'_fit',side_effect=AssertionError('cache missed')):
                other_scale,other_fit,other_mapping,_=second._fit(x,y)
            self.assertEqual(second.fit_cache_hits,1)
            self.assertEqual(mapping,other_mapping)
            np.testing.assert_array_equal(fit.predict(scaler.transform(x)),other_fit.predict(other_scale.transform(x)))
            np.testing.assert_allclose(fit.predict_proba(scaler.transform(x)),other_fit.predict_proba(other_scale.transform(x)),atol=1e-12)


    def test_checksum_manifest_rejects_partial_or_modified_artifacts(self):
        with TemporaryDirectory() as temporary:
            folder=Path(temporary)
            atomic_json(folder/'values.json',{'x':1})
            self.assertIsNone(read_manifest(folder,'key',['values.json']))
            commit_manifest(folder,'key',['values.json'])
            self.assertIsNotNone(read_manifest(folder,'key',['values.json']))
            (folder/'values.json').write_text('{"x":2}')
            self.assertIsNone(read_manifest(folder,'key',['values.json']))



class PipelineIntegrationTests(unittest.TestCase):
    def test_incomplete_funding_is_rejected_before_training(self):
        text='''name="coverage"
[paths]
data_dir="data"
output_root="reports"
cache_dir="cache"
[[observations]]
name="obs5"
features=[{kind="log_return",window=5},{kind="efficiency",window=5},{kind="taker_imbalance",window=5}]
[[models]]
train_window=90
naming_window=10
seeds=[42]
[evaluation]
start="2020-01-01"
end="2020-07-18"
'''
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'data').mkdir()
            b=sample_bars()
            b.to_csv(root/'data/BTCUSDT_1d.csv')
            dates=pd.date_range(b.index[0],periods=len(b)*3,freq='8h').delete(450)
            pd.DataFrame(dict(datetime=dates,last_funding_rate=0.,funding_interval_hours=8)).to_csv(root/'data/BTCUSDT_funding_events.csv',index=False)
            path=root/'experiment.toml'
            path.write_text(text,encoding='utf-8')
            with patch('trading.pipeline._trained_probabilities',side_effect=AssertionError('training started')):
                with self.assertRaisesRegex(ValueError,'funding coverage'): run_experiment(path)

    def test_one_command_and_cost_changes_reuse_training(self):
        text='''name="integration"
[paths]
data_dir="data"
output_root="reports"
cache_dir="cache"
[[observations]]
name="obs5"
features=[{kind="log_return",window=5},{kind="efficiency",window=5},{kind="taker_imbalance",window=5}]
[[observations]]
name="obs10"
features=[{kind="log_return",window=10},{kind="efficiency",window=10},{kind="taker_imbalance",window=10}]
[[models]]
train_window=90
retrain_every=30
naming_window=10
decode_window=30
iterations=100
restarts=2
seeds=[42,137]
[decisions]
ratios=[1.0,2.0]
[trading]
fees=0.0005
[evaluation]
start="2020-01-01"
end="2020-07-18"
prefix_bars=35
[plots]
enabled=false
'''
        with TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            root=Path(temporary)
            (root/'data').mkdir()
            b=sample_bars()
            b.to_csv(root/'data/BTCUSDT_1d.csv')
            events=pd.date_range(b.index[0],periods=len(b)*3,freq='8h')
            pd.DataFrame(dict(datetime=events,last_funding_rate=0.,funding_interval_hours=8)).to_csv(root/'data/BTCUSDT_funding_events.csv',index=False)
            path=root/'experiment.toml'
            path.write_text(text,encoding='utf-8')
            first=run_experiment(path)
            self.assertEqual(len(first['summary']),6)
            self.assertGreater(first['cache']['fit_misses'],0)
            self.assertFalse(first['backtest_rejections'])
            second=run_experiment(path)
            self.assertEqual(second['cache']['fit_misses'],0)
            self.assertEqual(second['cache']['probability_hits'],4)
            self.assertEqual(second['cache']['backtest_hits'],12)
            alternate=pd.read_csv(Path(second['output'])/'training/obs5_hmm/seed_137/probabilities.csv',index_col=0,parse_dates=True).viterbi
            reference,returns=_reference(b,20)
            expected=comparison_metrics(alternate,reference,returns,alternate.ge(0))['agreement']
            seed_results=pd.read_csv(Path(second['output'])/'all_seed_results.csv')
            actual=seed_results.loc[seed_results.case.eq('obs5_hmm_viterbi')&seed_results.seed.eq(137),'reference20_agreement'].iloc[0]
            self.assertAlmostEqual(actual,expected)
            path.write_text(text.replace('fees=0.0005','fees=0.001'),encoding='utf-8')
            changed=run_experiment(path)
            self.assertEqual(changed['cache']['fit_misses'],0)
            self.assertEqual(changed['cache']['probability_hits'],4)
            self.assertGreater(changed['cache']['backtest_misses'],0)
            self.assertNotEqual(first['output'],changed['output'])
            self.assertTrue((Path(first['output'])/'report.json').exists())
            path.write_text(text.replace('ratios=[1.0,2.0]','ratios=[1.0,4.0]'),encoding='utf-8')
            threshold_changed=run_experiment(path)
            self.assertEqual(threshold_changed['cache']['fit_misses'],0)
            self.assertEqual(threshold_changed['cache']['probability_hits'],4)
            self.assertTrue(any(row['decoder']=='filter_r4' for row in threshold_changed['summary']))
            profiles='''
[[strategies]]
name="directional"
[[strategies]]
name="ema_global"
bull="ema"
side="ema"
bear="ema"
ema_filter=20
'''
            path.write_text(text.replace('ratios=[1.0,2.0]','ratios=[]')+profiles,encoding='utf-8')
            combined=run_experiment(path)
            self.assertEqual(len(combined['summary']),4)
            self.assertEqual(combined['cache']['probability_hits'],4)
            self.assertEqual(combined['cache']['fit_misses'],0)
            self.assertFalse(combined['backtest_rejections'])
            rows=pd.read_csv(Path(combined['output'])/'all_seed_results.csv')
            globals_=rows.loc[rows.strategy.eq('ema_global')]
            self.assertEqual(globals_.total_return.nunique(),1)
            self.assertEqual(globals_.annual_turnover_gross.nunique(),1)
            self.assertTrue(rows.annual_turnover_gross.ge(0).all())
            self.assertIn('annual_turnover_gross',combined['metric_definitions'])
            self.assertTrue((Path(combined['output'])/'cases/obs5_hmm_viterbi__directional/seed_42/targets.csv').exists())
            path.write_text(text.replace('ratios=[1.0,2.0]','ratios=[]').replace('prefix_bars=35','prefix_bars=35\nsharpe_benchmark="btc_price"')+profiles,encoding='utf-8')
            relative=run_experiment(path)
            relative_rows=pd.read_csv(Path(relative['output'])/'all_seed_results.csv')
            np.testing.assert_allclose(relative_rows.sharpe,relative_rows.sharpe_btc)
            np.testing.assert_allclose(relative_rows.sharpe_zero,rows.sharpe)
            self.assertEqual(relative['cache']['fit_misses'],0)
            self.assertEqual(relative['cache']['backtest_misses'],0)


class PublicCoreScopeTests(unittest.TestCase):
    def test_optional_research_stages_fail_before_input_loading(self):
        base='name="scope"\n[[observations]]\nname="x"\nfeatures=[{kind="log_return",window=5}]\n'
        variants=['[[models]]\nkind="hsmm_poisson"\n',
            '[diagnostics]\nbear_regimes=true\n',
            '[label_validation]\nenabled=true\n',
            '[downside_prediction]\nenabled=true\n',
            '[smoothing_validation]\nenabled=true\n']
        with TemporaryDirectory() as temporary:
            path=Path(temporary)/'test.toml'
            for suffix in variants:
                path.write_text(base+suffix,encoding='utf-8')
                with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError,'Not included in the public core'):
                    run_experiment(path,check=True)

    def test_typo_and_invalid_probability_ratio_fail_at_config_load(self):
        base='name="scope"\n[[observations]]\nname="x"\nfeatures=[{kind="log_return",window=5}]\n'
        with TemporaryDirectory() as temporary:
            path=Path(temporary)/'test.toml'
            for suffix in ['[decisions]\nratios=[0.5]\n','[[models]]\nrestatrts=10\n']:
                path.write_text(base+suffix,encoding='utf-8')
                with self.assertRaises(ValueError): load_experiment(path)


if __name__=='__main__': unittest.main()
