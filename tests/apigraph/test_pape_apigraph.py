import io,sys,tempfile,unittest,copy
from pathlib import Path
from contextlib import redirect_stdout
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import numpy as np
import pandas as pd
import yaml
from drone_health_retraining.apigraph.common import *
from drone_health_retraining.apigraph.data import build_membership,groups_of
from drone_health_retraining.pape.apigraph.pape import estimate_target_performance,domain_folds
from drone_health_retraining.pape.apigraph.pipeline import pape_stage,verify_pape
from drone_health_retraining.pape.apigraph.handoff import read_app_inputs
from drone_health_retraining.pape.apigraph.parallel import trial_results
from drone_health_retraining.pape.apigraph.source_schedule import planned_source_context
from drone_health_retraining.pape.apigraph.integration import load_operational_inputs,validate_gain_timeline

class APIGraphPapeTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.temp=tempfile.TemporaryDirectory();root=Path(cls.temp.name)
  cfg=load_config('config/apigraph.yaml');cfg['paths']={'data':str(root/'unused'),'outputs':str(root/'outputs')}
  cfg['execution']['workers']=1
  cfg['data']['features']=16;cfg['pape']['features']=[f'binary_{i:04d}' for i in range(16)]
  for k in ['domain_model','calibrator']:cfg['pape'][k].update(n_estimators=4,num_leaves=4,min_child_samples=3,n_jobs=1)
  rng=np.random.default_rng(912);base=np.unique(rng.integers(0,2,(360,16),dtype=np.uint8),axis=0)
  export=root/'fixture_export';export.mkdir();infos=[]
  (export/'resolved.yaml').write_text(yaml.safe_dump(cfg),encoding='utf-8')
  for role,years in [('source',range(2013,2015)),('validation',[2015])]:
   xs=[];months=[]
   for year in years:
    for mm in range(1,13):xs.append(base[rng.choice(len(base),24,replace=False)]);months.extend([f'{year}-{mm:02d}']*24)
   x=np.concatenate(xs);scores=.1+.8*x[:,0];ids=[role+':'+str(i) for i in range(len(x))]
   apps=pd.DataFrame({'sample_id':ids,'origin_file':'synthetic.npz','original_row':np.arange(len(x)),
    'feature_group':groups_of(x),'observed_month':months,'project_role':role,'xgb_score':scores,
    'final_prediction':(scores>.5).astype('uint8'),'decision_threshold':.5,'model_hash':'a'*64,
    'threshold_policy_hash':'b'*64,'schema_version':'rise2_apigraph_v2','feature_schema_hash':'c'*64})
   folder=export/role;folder.mkdir();apps.to_parquet(folder/'app_features.parquet',index=False)
   binary=pd.DataFrame(x,columns=cfg['pape']['features']);binary.insert(0,'sample_id',ids);binary.to_parquet(folder/'binary_features.parquet',index=False)
   labels=pd.DataFrame({'sample_id':ids,'y_true':x[:,0]});labels.to_parquet(folder/'labels_for_evaluation.parquet',index=False)
   membership=build_membership(apps,cfg,'monthly');membership.to_parquet(folder/'chunk_membership.parquet',index=False)
   pd.DataFrame({'synthetic':[True]}).to_csv(folder/'XGBoost_true_data.csv',index=False)
   info={'role':role,'source_manifest_hash':digest({'ids':ids,'scores':scores.tolist(),'truth':x[:,0].tolist()}) if role=='source' else None,
    'app_file_hash':hash_file(folder/'app_features.parquet'),'label_file_hash':hash_file(folder/'labels_for_evaluation.parquet'),
    'membership_file_hash':hash_file(folder/'chunk_membership.parquet'),'truth_file_hash':hash_file(folder/'XGBoost_true_data.csv'),
    'binary_file_hash':hash_file(folder/'binary_features.parquet')}
   write_json(folder/'role_manifest.json',info);infos.append(info)
  write_json(export/'manifest.json',{'roles':infos,'config_hash':digest(cfg),'code_revision':'model_fixture_not_pape_revision',
   'model_hash':'a'*64,'threshold_policy_hash':'b'*64,'model_run':'not_required_by_pape','synthetic_fixture':True})
  write_json(export/'run_status.json',{'status':'completed'})
  cls.cfg=cfg;cls.export=export
  with redirect_stdout(io.StringIO()):cls.result=pape_stage(cfg,export_run=export)
  cls.source=read_app_inputs(export,'source').merge(pd.read_parquet(export/'source/labels_for_evaluation.parquet'),on='sample_id')
  cls.target=read_app_inputs(export,'validation').iloc[:24].copy()
 @classmethod
 def tearDownClass(cls):cls.temp.cleanup()
 def test_72_rows_and_14_column_contract(self):
  self.assertEqual(pd.read_csv(self.result/'pape_estimates.csv').shape,(72,14));self.assertEqual(verify_pape(self.result)['status'],'PASS')
 def test_no_xgboost_training_dependency(self):
  self.assertFalse(Path('not_required_by_pape').exists());self.assertTrue((self.result/'rccda_inputs.json').exists())
 def test_full_binary_features(self):
  d=read_json(self.result/'pape_summary.json')['records'][0]['diagnostics'];self.assertEqual(d['n_input_features'],16);self.assertNotIn('xgb_score',d['calibration_features'])
 def test_target_label_proxies_ignored(self):
  cfg=copy.deepcopy(self.cfg);cfg['target_chunk_id']='fixture';cfg['pape']['weighting']='dre'
  a=estimate_target_performance(self.source,self.target,cfg);b=estimate_target_performance(self.source,self.target.assign(y_true=0,actual_fn=999),cfg)
  self.assertEqual(a.estimates,b.estimates)
 def test_group_oof_no_leakage(self):
  for _,si,so,ti,to in domain_folds(self.source,self.target,3,11):self.assertFalse((set(self.source.iloc[si].feature_group)|set(self.target.iloc[ti].feature_group))&(set(self.source.iloc[so].feature_group)|set(self.target.iloc[to].feature_group)))
 def test_operational_inputs_no_actuals(self):
  df=load_operational_inputs(self.result,seed=11,weighting='dre');self.assertEqual(len(df),12);self.assertFalse(any(c.startswith('actual_') for c in df))
 def test_confusion_matrix(self):
  df=pd.read_csv(self.result/'confusion_matrix.csv')
  for p in ['estimated','actual']:np.testing.assert_allclose(sum(df[p+'_'+k] for k in ['tp','fp','fn','tn']),df.n_target)
 def test_future_source_and_test_blocked(self):
  cfg=copy.deepcopy(self.cfg);cfg['target_chunk_id']='fixture';cfg['pape']['weighting']='dre';source=self.source.copy();source.loc[0,'observed_month']='2015-01'
  with self.assertRaisesRegex(ValueError,'Period/role'):estimate_target_performance(source,self.target,cfg)
  with self.assertRaisesRegex(ValueError,'disabled'):pape_stage(cfg,target_role='test')
 def test_path_changes_do_not_change_semantic_export_contract(self):
  exported=yaml.safe_load((self.export/'resolved.yaml').read_text());self.assertEqual(semantic_config(exported),semantic_config(self.cfg))
 def test_gain_horizon_is_monthly(self):
  v=validate_gain_timeline('2016-11',['2015-01'],['2016-12','2017-01'],2);self.assertEqual(v['gain_label_available_from_month'],'2017-02')

 def test_serial_parallel_numeric_equivalence(self):
  members=pd.read_parquet(self.export/'validation/chunk_membership.parquet')
  cid=members.sort_values('ordinal').target_chunk_id.iloc[0]
  target=read_app_inputs(self.export,'validation')
  results=[]
  for workers in [1,2]:
   cfg=copy.deepcopy(self.cfg);cfg['execution']['workers']=workers;cfg['pape']['seeds']=[11]
   results.append(list(trial_results(cfg,cfg,self.export,self.export,'validation',[cid],['uniform','dre'],self.source,target,members)))
  for a,b in zip(*results):
   self.assertEqual(a[:3],b[:3])
   np.testing.assert_array_equal(a[4].target_probabilities.calibrated_probability,b[4].target_probabilities.calibrated_probability)
   np.testing.assert_array_equal(a[4].source_weights.raw_weight,b[4].source_weights.raw_weight)

 def test_meeting_annual_source_handoff_guards(self):
  self.assertEqual(planned_source_context(2015)['source_period'],['2013-01','2014-12'])
  with self.assertRaisesRegex(ValueError,'Freeze'):planned_source_context(2016)
  self.assertEqual(planned_source_context(2016,selection_frozen=True)['source_period'],['2013-01','2015-12'])
  with self.assertRaisesRegex(ValueError,'Previous Test'):planned_source_context(2017,selection_frozen=True)
  self.assertEqual(planned_source_context(2018,selection_frozen=True,previous_year_complete=True)['source_period'],['2013-01','2017-12'])

if __name__=='__main__':unittest.main(verbosity=2)
