import io,sys,tempfile,unittest
from pathlib import Path
from contextlib import redirect_stdout
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import numpy as np
import pandas as pd
from drone_health_retraining.apigraph.common import *
from drone_health_retraining.apigraph.data import load_role
from drone_health_retraining.model_adapter.apigraph.model import train_stage,export_stage
from drone_health_retraining.model_adapter.apigraph.output import build_model_handoff,verify_model_handoff,zip_model_handoff

class APIGraphModelTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.temp=tempfile.TemporaryDirectory();root=Path(cls.temp.name);data=root/'data';data.mkdir()
  cfg=load_config('config/apigraph.yaml');cfg['paths']={'data':str(data),'outputs':str(root/'outputs')}
  cfg['data']['features']=16;cfg['pape']['features']=[f'binary_{i:04d}' for i in range(16)]
  cfg['model'].update(n_estimators=8,early_stopping_rounds=2,max_depth=2,n_jobs=1)
  rng=np.random.default_rng(911);base=np.unique(rng.integers(0,2,(360,16),dtype=np.uint8),axis=0)
  def save(name,x,t):np.savez_compressed(data/name,feature=x,binary_label=x[:,0],timestamp=np.array(t))
  save('train.npz',base[:240],['2012-XX']*240)
  rows=[];months=[]
  for year in range(2013,2019):
   for mm in range(1,13):rows.append(base[rng.choice(len(base),24,replace=False)]);months.extend([f'{year}-{mm:02d}']*24)
  rows=np.concatenate(rows);save('val.npz',rows[:288],months[:288]);save('test.npz',rows[288:],months[288:])
  cls.cfg=cfg
  with redirect_stdout(io.StringIO()):
   cls.model=train_stage(cfg,fixture=True);cls.export=export_stage(cfg,cls.model,fixture=True)
   cls.handoff=build_model_handoff(cls.model,cls.export,root/'handoff')
   cls.zip=zip_model_handoff(cls.handoff,root/'model.zip')
 @classmethod
 def tearDownClass(cls):cls.temp.cleanup()
 def test_fresh_training(self):
  m=read_json(self.model/'baseline_manifest.json');self.assertTrue(m['fresh_training']);self.assertFalse(m['warm_start']);self.assertFalse(m['previous_models_loaded'])
 def test_required_workplan_files(self):
  f=pd.read_parquet(self.handoff/'window_features.parquet');self.assertEqual(f.shape,(1152,13));self.assertFalse(f.sample_id.duplicated().any());self.assertNotIn('y_true',f)
  self.assertTrue((self.handoff/'baseline_manifest.json').exists())
 def test_months_roles_and_counts(self):
  m=pd.read_csv(self.handoff/'model_monthly_metrics.csv');self.assertEqual(len(m),48)
  self.assertEqual(m.groupby('role').n.sum().to_dict(),{'source':864,'validation':288})
 def test_prediction_replay(self):
  v=verify_model_handoff(self.handoff);self.assertTrue(v['model_predictions_replayed']);self.assertEqual(v['prediction_replay_max_absolute_difference'],0)
 def test_test_access_disabled(self):
  with self.assertRaisesRegex(ValueError,'disabled'):load_role(self.cfg,'test',True)
 def test_labels_and_binary_sidecars(self):
  for role in ['source','validation']:
   app=pd.read_parquet(self.handoff/role/'window_features.parquet');binary=pd.read_parquet(self.handoff/role/'binary_features.parquet');labels=pd.read_parquet(self.handoff/role/'labels_for_evaluation.parquet')
   self.assertEqual(app.sample_id.tolist(),binary.sample_id.tolist());self.assertEqual(len(app),len(labels));self.assertNotIn('y_true',app)
 def test_manifest_tracks_environment_and_commands(self):
  m=read_json(self.handoff/'baseline_manifest.json');self.assertEqual(m['decision_window_unit'],'calendar_month');self.assertFalse(m['gpu_used']);self.assertEqual(len(m['reproduction_commands']),3)
 def test_tampering_rejected(self):
  p=self.handoff/'window_features.parquet';original=p.read_bytes()
  try:
   p.write_bytes(b'changed')
   with self.assertRaisesRegex(ValueError,'changed'):verify_model_handoff(self.handoff)
  finally:p.write_bytes(original)

if __name__=='__main__':unittest.main(verbosity=2)
