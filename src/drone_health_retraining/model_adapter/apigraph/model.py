from importlib.metadata import version
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,brier_score_loss
from ...apigraph.common import *
from ...apigraph.data import load_train,load_role,verify_data,build_membership

def model_metrics(y,p,cfg):
 d=prediction(p,cfg);m=actual_confusion(y,d,cfg);tp,fp,fn,tn=[m[k] for k in ['tp','fp','fn','tn']]
 return {**m,'n':len(y),'accuracy':(tp+tn)/len(y),'precision':tp/(tp+fp) if tp+fp else 0.,
 'fpr':fp/(fp+tn) if fp+tn else None,'fnr':fn/(fn+tp) if fn+tp else None,
 'always_benign_accuracy':float((np.asarray(y)==0).mean()),
 'roc_auc':float(roc_auc_score(y,p)) if np.unique(y).size==2 else None,
 'average_precision':float(average_precision_score(y,p)) if np.unique(y).size==2 else None,
 'logloss':float(log_loss(y,p,labels=[0,1])),'brier':float(brier_score_loss(y,p))}

def train_stage(cfg,fixture=False):
 if not fixture:verify_data(cfg)
 out=make_run(cfg,'model_runs');started=time.perf_counter()
 try:
  raw,y,groups=load_train(cfg);x=raw.astype(np.float32);seed=int(cfg['seed']);s=cfg['internal_split']
  development,hold=next(GroupShuffleSplit(n_splits=1,test_size=s['independent_holdout_fraction'],random_state=seed).split(x,y,groups))
  fi,vi=next(GroupShuffleSplit(n_splits=1,test_size=s['early_stopping_fraction']/(1-s['independent_holdout_fraction']),random_state=seed+1).split(x[development],y[development],groups[development]))
  fit,valid=development[fi],development[vi]
  for a,b in [(fit,valid),(fit,hold),(valid,hold)]:
   if set(groups[a])&set(groups[b]):raise ValueError('Internal group leakage')
  for indices in [fit,valid,hold]:
   if np.unique(y[indices]).size!=2:raise ValueError('Both classes required in internal partitions')
  print(f'Fresh XGBoost: fit={len(fit)}, stopping={len(valid)}, holdout={len(hold)}',flush=True)
  params={**cfg['model'],'random_state':seed}
  internal=xgb.XGBClassifier(**params)
  # No checkpoint/xgb_model is passed: this always starts from a new estimator.
  internal.fit(x[fit],y[fit],eval_set=[(x[valid],y[valid])],verbose=False)
  internal_metrics=[{'role':role,**model_metrics(y[ix],internal.predict_proba(x[ix])[:,1].astype(float),cfg)} for role,ix in [('fit',fit),('early_stopping',valid),('independent_holdout',hold)]]
  pd.DataFrame(internal_metrics).to_csv(out/'internal_metrics.csv',index=False,encoding='utf-8-sig')
  pd.DataFrame([{'original_row':int(i),'feature_group':groups[i],'role':role} for role,ix in [('fit',fit),('early_stopping',valid),('independent_holdout',hold)] for i in ix]).to_csv(out/'internal_split_manifest.csv',index=False)
  pd.DataFrame({'iteration':np.arange(len(internal.evals_result()['validation_0']['logloss'])),
   'early_stopping_logloss':internal.evals_result()['validation_0']['logloss']}).to_csv(out/'learning_curve.csv',index=False)
  rounds=internal.best_iteration+1;params.pop('early_stopping_rounds');params['n_estimators']=rounds
  baseline=xgb.XGBClassifier(**params);baseline.fit(x,y,verbose=False)
  if not np.array_equal(baseline.classes_,[0,1]):raise ValueError('Unexpected class order')
  baseline.save_model(out/'model.json');restored=xgb.XGBClassifier();restored.load_model(out/'model.json')
  np.testing.assert_array_equal(baseline.predict_proba(x[:500]),restored.predict_proba(x[:500]))
  policy={'threshold':cfg['decision']['threshold'],'comparator':'gt','model_hash':hash_file(out/'model.json'),
   'positive_label':1,'selected_by':cfg['decision']['selected_by']};policy['threshold_policy_hash']=digest(policy)
  write_json(out/'threshold_policy.json',policy)
  schema={'dimension':cfg['data']['features'],'column_order':list(range(cfg['data']['features'])),
   'representation':'public_binary_features_dense_float32','zero_means':'feature_absent_not_missing',
   'feature_names_mapping_verified':False}
  write_json(out/'feature_schema.json',{**schema,'feature_schema_hash':digest(schema)})
  manifest={'schema_version':cfg['schema_version'],'fresh_training':True,'warm_start':False,
   'previous_models_loaded':False,'synthetic_fixture':fixture,'train_year':'2012','n_train':len(y),
   'training_data_hash':hash_file(local_path(cfg['paths']['data'])/'train.npz'),'tree_count':rounds,
   'model_hash':hash_file(out/'model.json'),'threshold_policy_hash':policy['threshold_policy_hash'],
   'feature_schema_hash':digest(schema),'created_at':now(),'config_hash':digest(cfg),'code_revision':model_code_revision(),
   'seed':seed,'packages':{name:version(name) for name in ['xgboost','numpy','scikit-learn']},
   'round_trip_max_absolute_diff':0.,'duration_seconds':time.perf_counter()-started,
   'internal_holdout_is_not_temporal_validation':True}
  write_json(out/'baseline_manifest.json',manifest);complete(out,manifest)
  print(f'TRAIN COMPLETE: {out}',flush=True);return out
 except Exception as e:write_json(out/'run_status.json',{'status':'failed','error':str(e)});raise

def validate_model_bundle(cfg,model_run,fixture=False):
 model_run=Path(model_run);mm=read_json(model_run/'baseline_manifest.json');policy=read_json(model_run/'threshold_policy.json')
 if hash_file(model_run/'model.json')!=mm['model_hash'] or mm['training_data_hash']!=hash_file(local_path(cfg['paths']['data'])/'train.npz'):raise ValueError('Model/data provenance mismatch')
 if not mm.get('fresh_training') or mm.get('warm_start') or mm.get('previous_models_loaded'):raise ValueError('Fresh model training required')
 if mm['config_hash']!=digest(cfg) or mm['code_revision']!=model_code_revision():raise ValueError('Model belongs to another configuration/code')
 if mm['train_year']!='2012':raise ValueError('Only a fresh 2012 detector is allowed')
 if cfg['decision']['threshold']!=policy['threshold'] or policy['comparator']!='gt':raise ValueError('Model decision policy mismatch')
 # Validate the policy content, rather than trusting a copied hash string.
 check={k:v for k,v in policy.items() if k!='threshold_policy_hash'}
 if digest(check)!=policy['threshold_policy_hash']:raise ValueError('Decision policy hash mismatch')
 if policy['model_hash']!=mm['model_hash'] or policy['threshold_policy_hash']!=mm['threshold_policy_hash']:raise ValueError('Model and decision policy are not paired')
 schema=read_json(model_run/'feature_schema.json');schema_content={k:v for k,v in schema.items() if k!='feature_schema_hash'}
 if digest(schema_content)!=schema['feature_schema_hash'] or schema['feature_schema_hash']!=mm['feature_schema_hash']:raise ValueError('Feature schema changed')
 if bool(mm['synthetic_fixture'])!=bool(fixture):raise ValueError('Synthetic and real model exports must not be mixed')
 if schema['dimension']!=cfg['data']['features']:raise ValueError('Feature dimension mismatch')
 return mm,policy

def export_one_role(cfg,role,out,model,mm,policy,previous,allow_final_test=False):
 x,frame=load_role(cfg,role,allow_final_test);frame['xgb_score']=model.predict_proba(x)[:,1].astype(float)
 frame['final_prediction']=prediction(frame.xgb_score,cfg);frame['decision_threshold']=policy['threshold']
 frame['model_hash']=mm['model_hash'];frame['threshold_policy_hash']=policy['threshold_policy_hash']
 frame['schema_version']=cfg['schema_version'];frame['feature_schema_hash']=mm['feature_schema_hash']
 if not np.isfinite(frame.xgb_score).all() or not frame.xgb_score.between(0,1).all():raise ValueError('Invalid scores')
 overlap=frame.feature_group.isin(previous);folder=out/role;folder.mkdir()
 labels=frame[['sample_id','y_true']].copy();labels.to_parquet(folder/'labels_for_evaluation.parquet',index=False)
 binary=pd.DataFrame(x.astype(np.uint8),columns=cfg['pape']['features'])
 frame.drop(columns='y_true').to_parquet(folder/'app_features.parquet',index=False)
 binary.insert(0,'sample_id',frame.sample_id.to_numpy())
 binary.to_parquet(folder/'binary_features.parquet',index=False)
 split_manifest=frame[['sample_id','origin_file','original_row','feature_group','observed_month','project_role']].copy()
 split_manifest['retention_reason']='kept_original_row';split_manifest.to_parquet(folder/'split_manifest.parquet',index=False)
 members=pd.concat([build_membership(frame,cfg,mode) for mode in cfg['chunks']['export_modes']],ignore_index=True)
 members.to_parquet(folder/'chunk_membership.parquet',index=False);truths=[]
 for cid,part in members.merge(frame,on='sample_id',validate='many_to_one').groupby('target_chunk_id',sort=False):
  truth=actual_confusion(part.y_true.to_numpy(),part.final_prediction.to_numpy(),cfg)
  truths.append({'target_chunk_id':cid,'mode':str(part['mode'].iloc[0]),'n_target':len(part),
   'member_hash':str(part.member_hash.iloc[0]),'model_hash':mm['model_hash'],
   'threshold_policy_hash':policy['threshold_policy_hash'],**{'actual_'+k:v for k,v in truth.items()},'oracle_evaluation_only':True})
 pd.DataFrame(truths).to_csv(folder/'XGBoost_true_data.csv',index=False,encoding='utf-8-sig')
 roleinfo={'role':role,'period':list(role_period(cfg,role)),'n_apps':len(frame),'raw_rows_kept':True,'T_malicious':int(frame.y_true.sum()),
    'F_benign':int((frame.y_true==0).sum()),'overlap_with_earlier_roles_rows':int(overlap.sum()),
    'within_role_unique_feature_groups':int(frame.feature_group.nunique()),
    'membership_modes':cfg['chunks']['export_modes'],'source_manifest_hash':digest({
      'ids':frame.sample_id.tolist(),'score':frame.xgb_score.tolist(),'truth':labels.y_true.tolist()}) if role=='source' else None,
    'binary_file_hash':hash_file(folder/'binary_features.parquet'),'binary_features':cfg['pape']['features'],
    'app_file_hash':hash_file(folder/'app_features.parquet'),'label_file_hash':hash_file(folder/'labels_for_evaluation.parquet'),
    'membership_file_hash':hash_file(folder/'chunk_membership.parquet'),'split_file_hash':hash_file(folder/'split_manifest.parquet'),
    'truth_file_hash':hash_file(folder/'XGBoost_true_data.csv'),'model_metrics':model_metrics(frame.y_true.to_numpy(),frame.xgb_score.to_numpy(),cfg)}
 write_json(folder/'role_manifest.json',roleinfo);previous.update(frame.feature_group)
 print(f'EXPORT {role}: {len(frame)} apps, {int(overlap.sum())} prior-feature overlaps retained',flush=True)
 return roleinfo

def export_stage(cfg,model_run=None,fixture=False,target_role='validation',selection_run=None):
 if not fixture:verify_data(cfg)
 lock=None;source_export=None
 if target_role!='validation':raise ValueError('Validation-only experiment: Test export disabled')
 model_run=latest(cfg,'model_runs',model_run);roles_to_export=['source','validation'];kind='feature_exports'
 mm,policy=validate_model_bundle(cfg,model_run,fixture)
 model=xgb.XGBClassifier();model.load_model(model_run/'model.json')
 if model.n_features_in_!=cfg['data']['features'] or not np.array_equal(model.classes_,[0,1]):raise ValueError('Feature dimension/class order mismatch')
 _,_,groups=load_train(cfg);previous=set(groups)
 out=make_run(cfg,kind);roles=[]
 try:
  for role in roles_to_export:roles.append(export_one_role(cfg,role,out,model,mm,policy,previous,allow_final_test=role=='test'))
  manifest={'protocol_version':cfg['protocol_version'],'schema_version':cfg['schema_version'],'model_run':str(model_run),'model_hash':mm['model_hash'],
   'threshold_policy_hash':policy['threshold_policy_hash'],'decision':cfg['decision'],
   'feature_schema_hash':mm['feature_schema_hash'],'roles':roles,'synthetic_fixture':fixture,
   'config_hash':digest(cfg),'code_revision':model_code_revision(),'created_at':now(),'split':cfg['split'],
   'final_test_scored':target_role=='test','final_test_period':list(role_period(cfg,'test')),'only_declared_role_rows_scored':True,
   'selection_run':str(lock_run) if lock else None,'selection_lock_hash':hash_file(lock_run/'selection_lock.json') if lock else None,
   'source_export_run':str(source_export) if source_export else None,'duplicate_policy':cfg['data']['duplicate_policy']}
  pd.DataFrame([{'role':r['role'],**r['model_metrics']} for r in roles]).to_csv(out/'model_results.csv',index=False,encoding='utf-8-sig')
  complete(out,manifest);print(f'EXPORT COMPLETE: {out}',flush=True);return out
 except Exception as e:write_json(out/'run_status.json',{'status':'failed','error':str(e)});raise
