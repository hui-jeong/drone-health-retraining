"""App adapter of the verified PAPE calculation; detector predictions remain fixed."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier,LGBMRegressor
from sklearn.metrics import roc_auc_score
from ...apigraph.common import *

@dataclass
class PapeResult:
 estimates:dict
 diagnostics:dict
 source_weights:pd.DataFrame
 target_probabilities:pd.DataFrame

def ess(weights):
 w=np.asarray(weights,float)
 if not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:raise ValueError('Invalid weights')
 return float(w.sum()**2/np.square(w).sum())

def domain_folds(source,target,n_splits,seed):
 # Original rows are retained. Identical feature patterns across the two domains
 # are held out together, so a fold cannot learn the same pattern from either domain.
 groups=np.union1d(source.feature_group.unique(),target.feature_group.unique())
 if min(source.feature_group.nunique(),target.feature_group.nunique())<n_splits:raise ValueError('Insufficient feature groups for DRE folds')
 rng=np.random.default_rng(seed);rng.shuffle(groups)
 # Assign Target groups first and evenly; distribute remaining Source groups.
 target_groups=np.asarray(target.feature_group.unique(),dtype=str).copy();rng.shuffle(target_groups)
 blocks=[set(b.tolist()) for b in np.array_split(target_groups,n_splits)]
 others=np.setdiff1d(groups,target_groups);rng.shuffle(others)
 for i,g in enumerate(others):blocks[i%n_splits].add(g)
 for fold,held in enumerate(blocks):
  so=np.flatnonzero(source.feature_group.isin(held));to=np.flatnonzero(target.feature_group.isin(held))
  si=np.flatnonzero(~source.feature_group.isin(held));ti=np.flatnonzero(~target.feature_group.isin(held))
  if min(len(so),len(to),len(si),len(ti))==0:raise ValueError('Empty domain fold')
  train_groups=set(source.iloc[si].feature_group)|set(target.iloc[ti].feature_group)
  test_groups=set(source.iloc[so].feature_group)|set(target.iloc[to].feature_group)
  if train_groups&test_groups:raise ValueError('Domain feature-group leakage')
  yield fold,si,so,ti,to

def estimate_target_performance(source_df,target_df,config)->PapeResult:
 # Target labels and label proxies are removed before validation, fitting, IDs or diagnostics.
 features=config['pape']['features']
 required=list(dict.fromkeys(INFERENCE_COLUMNS+features))
 missing=set(required)-set(target_df)
 if missing:raise ValueError('Missing Target columns: '+str(sorted(missing)))
 target=target_df.loc[:,required].reset_index(drop=True).copy()
 source=source_df.reset_index(drop=True).copy()
 if 'y_true' not in source:raise ValueError('Source truth is required')
 target_role=config.get('target_role','validation')
 if target_role!='validation':raise ValueError('Validation-only experiment: Test estimation disabled')
 for frame,role in [(source,'source'),(target,target_role)]:
  if frame.empty or not set(INFERENCE_COLUMNS)<=set(frame) or frame.sample_id.duplicated().any():raise ValueError('Invalid app table')
  if frame[INFERENCE_COLUMNS].isna().any().any():raise ValueError('Missing app input values')
  start,end=role_period(config,role)
  if not frame.project_role.eq(role).all() or not frame.observed_month.between(start,end).all():raise ValueError('Period/role guard failed')
  if not np.isfinite(frame.xgb_score).all() or not frame.xgb_score.between(0,1).all():raise ValueError('Score outside [0,1]')
  if not np.array_equal(prediction(frame.xgb_score,config),frame.final_prediction):raise ValueError('Detector prediction is not p>0.5')
  if not frame.decision_threshold.eq(.5).all():raise ValueError('Threshold mismatch')
 if set(source.sample_id)&set(target.sample_id):raise ValueError('Source/Target app row overlap')
 for field in ['model_hash','threshold_policy_hash']:
  if source[field].nunique()!=1 or target[field].nunique()!=1 or source[field].iloc[0]!=target[field].iloc[0]:raise ValueError('Detector provenance mismatch: '+field)
 if not np.isin(source.y_true,[0,1]).all() or source.y_true.nunique()!=2:raise ValueError('Source needs both classes')
 settings=config['pape'];seed=int(config['seed']);weighting=settings.get('weighting','dre');features=settings['features']
 if features!=[f'binary_{i:04d}' for i in range(config['data']['features'])]:raise ValueError('All binary features required')
 for frame in [source,target]:
  if not set(features)<=set(frame) or not np.isin(frame[features].to_numpy(),[0,1]).all():raise ValueError('Invalid binary feature matrix')
 weights_table=source[['sample_id','feature_group']].copy();fold_details=[];raw=np.ones(len(source));fold_ids=np.full(len(source),-1,dtype=int);aucs=[]
 if weighting=='dre':
  params=dict(settings['domain_model'])
  if params.get('class_weight') is not None or params.get('scale_pos_weight',1)!=1 or params.get('is_unbalance',False):raise ValueError('DRE class balancing invalidates empirical prior correction')
  if params.get('objective')!='binary':raise ValueError('Binary domain objective required')
  for fold,si,so,ti,to in domain_folds(source,target,settings['n_splits'],seed):
   x=pd.concat([source.iloc[si][features],target.iloc[ti][features]],ignore_index=True)
   domains=np.r_[np.zeros(len(si),dtype=int),np.ones(len(ti),dtype=int)]
   model=LGBMClassifier(**params,random_state=seed+fold);model.fit(x,domains)
   ps=model.predict_proba(source.iloc[so][features])[:,1];pt=model.predict_proba(target.iloc[to][features])[:,1]
   epsilon=settings['probability_epsilon'];ps=np.clip(ps,epsilon,1-epsilon)
   raw[so]=(len(si)/len(ti))*ps/(1-ps);fold_ids[so]=fold
   auc=float(roc_auc_score(np.r_[np.zeros(len(so)),np.ones(len(to))],np.r_[ps,pt]));aucs.append(auc)
   fold_details.append({'fold':fold,'n_source_train':len(si),'n_source_holdout':len(so),
    'n_target_train':len(ti),'n_target_holdout':len(to),'prior_source_over_target':len(si)/len(ti),
    'feature_group_train_holdout_overlap':0,'dre_auc':auc})
  clipped=np.clip(raw,settings['clip_min'],settings['clip_max'])
 elif weighting=='uniform':clipped=raw.copy()
 else:raise ValueError('Unknown weighting')
 weights_table['raw_weight']=raw;weights_table['clipped_weight']=clipped;weights_table['fold']=fold_ids
 calibrator=LGBMRegressor(**settings['calibrator'],random_state=seed)
 calibrator.fit(source[features],source.y_true,sample_weight=clipped)
 unbounded=calibrator.predict(target[features])
 if not np.isfinite(unbounded).all():raise ValueError('Nonfinite calibrated probability')
 probabilities=np.clip(unbounded,0,1)
 expected=expected_confusion(probabilities,target.final_prediction.to_numpy(),config)
 lower,upper=float(source.xgb_score.min()),float(source.xgb_score.max())
 outside=float(((target.xgb_score<lower)|(target.xgb_score>upper)).mean())
 inbox=source.xgb_score.between(float(target.xgb_score.min()),float(target.xgb_score.max()))
 diagnostics={'dre_auc':float(np.mean(aucs)) if aucs else None,'ess':ess(clipped),
  'ess_ratio':ess(clipped)/len(source),'max_weight':float(clipped.max()),'max_raw_weight':float(raw.max()),
  'raw_ess':ess(raw),'clip_fraction':float((raw!=clipped).mean()),'target_outside_source_range_fraction':outside,
  'source_positive_rows_in_target_score_box':int((inbox&source.y_true.eq(1)).sum()),
  'source_positive_groups_in_target_score_box':int(source.loc[inbox&source.y_true.eq(1),'feature_group'].nunique()),
  'target_feature_pattern_seen_in_source_fraction':float(target.feature_group.isin(source.feature_group).mean()),
  'calibration_probability_clip_fraction':float(((unbounded<0)|(unbounded>1)).mean()),
  'folds':fold_details,'weight_method':'source_group_oof_with_joint_domain_group_holdout' if weighting=='dre' else 'uniform',
  'target_labels_used':False,'n_source':len(source),'n_target':len(target),'seed':seed,
  'weighting':weighting,'calibration_features':features,'domain_features':features,'n_input_features':len(features),'score_range_diagnostics_only':True,'config_hash':digest(config),
  'model_hash':str(target.model_hash.iloc[0]),'threshold_policy_hash':str(target.threshold_policy_hash.iloc[0]),
  'created_at':now(),'metric_definition':'plugin_metrics_of_expected_confusion_counts'}
 warnings=[];limits=config['diagnostics']
 if diagnostics['ess_ratio']<limits['min_ess_ratio']:warnings.append('low_effective_sample_size')
 if diagnostics['clip_fraction']>limits['max_clip_fraction']:warnings.append('excessive_weight_clipping')
 if outside>limits['max_outside_range_fraction']:warnings.append('target_outside_source_score_range')
 if len(target)<limits['min_target_apps']:warnings.append('small_target_sample')
 if expected['tp']+expected['fn']<limits['min_expected_positives']:warnings.append('too_few_expected_positives')
 if not diagnostics['source_positive_groups_in_target_score_box']:warnings.append('no_source_positive_support_in_target_score_box')
 diagnostics.update(warnings=warnings,readiness='requires_review' if warnings else 'no_diagnostic_flags',accuracy_guaranteed=False,policy_action=None)
 cid=config['target_chunk_id']
 estimates={'target_chunk_id':cid,'input_variant':'all_binary__'+weighting,
  **{'estimated_'+k:v for k,v in expected.items()},
  **{k:diagnostics[k] for k in ['dre_auc','ess_ratio','max_weight']}}
 probs=target[['sample_id','xgb_score','final_prediction']].copy();probs['calibrated_probability']=probabilities
 if not np.isclose(expected['tp']+expected['fp'],target.final_prediction.sum()) or not np.isclose(expected['fn']+expected['tn'],len(target)-target.final_prediction.sum()):raise ValueError('Expected confusion invariant failed')
 return PapeResult(estimates,diagnostics,weights_table,probs)
