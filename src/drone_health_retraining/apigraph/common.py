from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import yaml
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
DATA_HASHES={
 'train.npz':'0943985e072fb00bb1610cf8ea4e837c53bfc4a2203ab30bf29fb75d6be78aa9',
 'val.npz':'80ee5c01114ad055d95dee5c4d2b045b88958dcdbbf56803eacbd15326e46e2b',
 'test.npz':'cb1d892434b7b194ca568c5328aa7ba14f3ab6bc8f43b473252ea4a337d5c638'}
PAPE_COLUMNS=['target_chunk_id','input_variant','estimated_fn','estimated_fp','estimated_recall','estimated_f1',
 'estimated_loss','actual_fn','actual_fp','actual_recall','actual_f1','dre_auc','ess_ratio','max_weight']
INFERENCE_COLUMNS=['sample_id','feature_group','observed_month','project_role','xgb_score',
 'final_prediction','decision_threshold','model_hash','threshold_policy_hash']

def now():return datetime.now(timezone.utc).isoformat()
def hash_file(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=True,allow_nan=False).encode()).hexdigest()
def write_json(path,obj):
 Path(path).write_text(json.dumps(obj,ensure_ascii=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def read_json(path):return json.loads(Path(path).read_text('utf-8-sig'))
def local_path(value):
 p=Path(value);return p if p.is_absolute() else ROOT/p
def _module_revision(part):
 roots=[ROOT/'src/drone_health_retraining/apigraph',ROOT/'src/drone_health_retraining'/part/'apigraph']
 files=[p for root in roots for p in root.rglob('*.py')]
 return digest({p.relative_to(ROOT).as_posix():hash_file(p) for p in sorted(files)})

def code_revision():return _module_revision('pape')
def model_code_revision():return _module_revision('model_adapter')

def semantic_config(cfg):
 return {k:v for k,v in cfg.items() if k!='paths'}

def load_config(path):
 cfg=yaml.safe_load(local_path(path).read_text('utf-8-sig'));validate_config(cfg);return cfg
def validate_config(cfg):
 split=cfg['split']
 required={'train_year':'2012','source_start':'2013-01','source_end':'2014-12','validation_start':'2015-01','validation_end':'2015-12','test_start':'2016-01','test_end':'2018-12'}
 if split!=required:raise ValueError('V8 requires 2012 Train, 2013-2014 Source, 2015 Validation, reserved 2016-2018 Test')
 if cfg['protocol_version']!='apigraph_meeting_validation2015_v8':raise ValueError('Protocol version mismatch')
 if cfg['chunks']['export_modes']!=['monthly']:raise ValueError('Only monthly windows are supported')
 if cfg.get('training')!={'mode':'fresh','pretrained_model_allowed':False} or 'model_reference' in cfg:raise ValueError('Fresh model training required')
 execution=cfg.get('execution',{})
 if execution.get('allowed_target_roles')!=['validation'] or execution.get('final_test_enabled') is not False or not 1<=execution.get('workers',0)<=4:raise ValueError('Validation-only execution with 1-4 workers required')
 annual=cfg.get('annual_source',{})
 if annual!={'update_month':1,'policy':'cumulative_previous_year','first_test_source_end':'2015-12','validation_released_after_selection_freeze':True,'require_previous_year_completed':True,'label_availability_assumption':'all_previous_year_labels_available_at_next_january','execution_owner':'future_integration'}:raise ValueError('Meeting annual Source handoff policy required')
 expected_integration={'schema_version':'pape_gain_rccda_monthly_v1','aggregation':'per_seed_no_pooling','matrix_label_order':[0,1],'matrix_axes':['true','predicted'],'decision_timing':'after_month_close_for_next_month','source_update_retrains_detector':False}
 if cfg['integration']!=expected_integration:raise ValueError('Integration contract mismatch')
 if cfg['data']['duplicate_policy']!='keep_original_rows_report_overlap':raise ValueError('Original-row policy required')
 if cfg['decision']['threshold']!=.5 or cfg['decision']['comparator']!='gt':raise ValueError('Agreed policy is p>0.5')
 if cfg['model']['objective']!='binary:logistic' or cfg['model']['device']!='cpu':raise ValueError('CPU binary model required')
 if cfg['model'].get('missing')==0:raise ValueError('Dense zero must not be missing')
 if cfg['pape']['features']!=[f'binary_{i:04d}' for i in range(cfg['data']['features'])]:raise ValueError('All binary features in original order required')
 if not set(cfg['pape']['weightings'])<={'uniform','dre'} or not cfg['pape']['weightings']:raise ValueError('Invalid weighting')
 if not cfg['pape']['seeds'] or len(set(cfg['pape']['seeds']))!=len(cfg['pape']['seeds']):raise ValueError('Unique seeds required')
 if cfg['pape']['n_splits']<2 or not 0<cfg['pape']['clip_min']<=cfg['pape']['clip_max']:raise ValueError('Invalid DRE settings')
 if not 0<cfg['pape']['probability_epsilon']<.5:raise ValueError('Invalid probability epsilon')
 costs=[cfg['loss']['lambda_fn'],cfg['loss']['lambda_fp']]
 if not np.isfinite(costs).all() or min(costs)<0 or max(costs)==0:raise ValueError('Invalid costs')
 if not 0<=cfg['metrics']['zero_division']<=1:raise ValueError('Invalid zero_division')
 if cfg['chunks']!={'export_modes':['monthly'],'unit':'calendar_month','missing_month_policy':'error'}:raise ValueError('Monthly chunk settings mismatch')
 if cfg['internal_split']['early_stopping_fraction']<=0 or cfg['internal_split']['independent_holdout_fraction']<=0 or sum(cfg['internal_split'].values())>=1:raise ValueError('Invalid internal split')
def make_run(cfg,kind):
 root=local_path(cfg['paths']['outputs'])/kind;root.mkdir(parents=True,exist_ok=True)
 out=root/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir()
 (out/'resolved.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
 write_json(out/'run_status.json',{'status':'running','started_at':now()});return out
def complete(out,manifest):
 write_json(out/'manifest.json',manifest);write_json(out/'run_status.json',{'status':'completed','completed_at':now()})
 write_json(out.parent/'latest.json',{'run':str(out)})
def latest(cfg,kind,explicit=None):
 out=local_path(explicit) if explicit else Path(read_json(local_path(cfg['paths']['outputs'])/kind/'latest.json')['run'])
 if read_json(out/'run_status.json')['status']!='completed':raise ValueError('Upstream stage is incomplete')
 return out

def role_period(cfg,role):
 if role not in ['source','validation','test']:raise ValueError('Unknown role')
 return cfg['split'][role+'_start'],cfg['split'][role+'_end']

def verify_export_files(path,roles):
 out=Path(path);manifest=read_json(out/'manifest.json')
 if read_json(out/'run_status.json')['status']!='completed':raise ValueError('Incomplete export')
 for role in roles:
  info=read_json(out/role/'role_manifest.json')
  if info!=next(r for r in manifest['roles'] if r['role']==role):raise ValueError('Role manifest differs from export manifest')
  for name,key in [('app_features.parquet','app_file_hash'),('labels_for_evaluation.parquet','label_file_hash'),('chunk_membership.parquet','membership_file_hash'),('XGBoost_true_data.csv','truth_file_hash')]:
   if hash_file(out/role/name)!=info[key]:raise ValueError('Export artifact changed: '+role+'/'+name)
  if 'binary_file_hash' in info and hash_file(out/role/'binary_features.parquet')!=info['binary_file_hash']:raise ValueError('Binary feature sidecar changed')
  if 'split_file_hash' in info and hash_file(out/role/'split_manifest.parquet')!=info['split_file_hash']:raise ValueError('Split manifest changed')
 return manifest
def prediction(p,cfg):return (np.asarray(p,dtype=float)>cfg['decision']['threshold']).astype(np.uint8)
def confusion_metrics(tp,fp,fn,tn,fn_cost,fp_cost,zero=0.):
 if not np.isfinite([tp,fp,fn,tn,fn_cost,fp_cost]).all() or min(tp,fp,fn,tn,fn_cost,fp_cost)<0:raise ValueError('Invalid confusion')
 return {'tp':float(tp),'fp':float(fp),'fn':float(fn),'tn':float(tn),
 'recall':float(tp/(tp+fn)) if tp+fn else float(zero),
 'accuracy':float((tp+tn)/(tp+fp+fn+tn)) if tp+fp+fn+tn else float(zero),
 'accuracy_defined':bool(tp+fp+fn+tn),
 'f1':float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else float(zero),
 'loss':float(fn_cost*fn+fp_cost*fp),'recall_defined':bool(tp+fn),'f1_defined':bool(2*tp+fp+fn)}
def expected_confusion(p,d,cfg):
 p=np.asarray(p,float);d=np.asarray(d)
 if p.ndim!=1 or p.shape!=d.shape or not len(p) or not np.isfinite(p).all() or ((p<0)|(p>1)).any() or not np.isin(d,[0,1]).all():raise ValueError('Invalid aligned probabilities/predictions')
 pos=d==1
 return confusion_metrics(p[pos].sum(),(1-p[pos]).sum(),p[~pos].sum(),(1-p[~pos]).sum(),cfg['loss']['lambda_fn'],cfg['loss']['lambda_fp'],cfg['metrics']['zero_division'])
def actual_confusion(y,d,cfg):
 if not np.isin(y,[0,1]).all():raise ValueError('Binary truth required')
 return expected_confusion(y,d,cfg)
