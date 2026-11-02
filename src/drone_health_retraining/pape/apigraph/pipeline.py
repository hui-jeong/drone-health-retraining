"""Fresh-model monthly Validation-only experiment."""
import argparse,copy,time
import numpy as np
import pandas as pd
from ...apigraph.common import *
from .pape import estimate_target_performance
from .handoff import read_app_inputs,write_handoff,verify_handoff
from .parallel import trial_results

def verified_export(cfg,explicit=None):
 out=latest(cfg,'feature_exports',explicit);manifest=verify_export_files(out,['source','validation'])
 exported=__import__('yaml').safe_load((out/'resolved.yaml').read_text('utf-8-sig'))
 if semantic_config(exported)!=semantic_config(cfg):raise ValueError('Export semantic configuration differs')
 return out,manifest

def pape_stage(cfg,export_run=None,mode='monthly',max_chunks=0,target_role='validation',selection_run=None,test_export_run=None,target_year=None):
 if mode!='monthly':raise ValueError('Only monthly windows are supported')
 if max_chunks<0:raise ValueError('max_chunks must be nonnegative')
 if target_role!='validation':raise ValueError('Validation-only experiment: Test estimation disabled')
 source_export,em=verified_export(cfg,export_run);target_export=source_export;tm=em
 lock=lock_run=None;weightings=cfg['pape']['weightings'];kind='pape_validation'
 stagecfg=copy.deepcopy(cfg);source_context={'source_version':'validation_2013_2014','source_available_from_month':'2015-01','source_period':['2013-01','2014-12']}
 if tm['model_hash']!=em['model_hash'] or tm['threshold_policy_hash']!=em['threshold_policy_hash']:raise ValueError('Source/Target detector mismatch')
 source=read_app_inputs(source_export,'source').merge(pd.read_parquet(source_export/'source/labels_for_evaluation.parquet'),on='sample_id',validate='one_to_one')
 source_hash=read_json(source_export/'source/role_manifest.json')['source_manifest_hash']
 target_pool=read_app_inputs(target_export,target_role)
 membership=pd.read_parquet(target_export/target_role/'chunk_membership.parquet');membership=membership[membership['mode'].eq(mode)]
 if membership.empty:raise ValueError('Requested mode was not exported')
 chunkids=membership.sort_values('ordinal').target_chunk_id.drop_duplicates().tolist();all_count=len(chunkids)
 if all_count!=12:raise ValueError('A full calendar year must contain 12 monthly chunks')
 if max_chunks:chunkids=chunkids[:max_chunks]
 out=make_run(cfg,kind);records=[];rows=[];started=time.perf_counter();truth_labels=None
 try:
  for cid,seed,weighting,trialconfig,result in trial_results(cfg,stagecfg,source_export,target_export,target_role,chunkids,weightings,source,target_pool,membership):
   members=membership[membership.target_chunk_id.eq(cid)].sort_values('position')
   target=members[['sample_id']].merge(target_pool,on='sample_id',validate='one_to_one')
   index=len(rows);trial=out/'trials'/f'{index:05d}';trial.mkdir(parents=True)
   result.target_probabilities.to_csv(trial/'target_probabilities.csv',index=False,encoding='utf-8-sig');w=result.source_weights
   np.savez_compressed(trial/'source_weights.npz',source_row_index=np.arange(len(w),dtype=np.int32),raw_weight=w.raw_weight.to_numpy(),clipped_weight=w.clipped_weight.to_numpy(),fold=w['fold'].to_numpy())
   write_json(trial/'diagnostics.json',result.diagnostics)
   # This is the first access to Target truth in this estimator execution stage.
   if truth_labels is None:truth_labels=pd.read_parquet(target_export/target_role/'labels_for_evaluation.parquet')
   evaluation=target[['sample_id','final_prediction']].merge(truth_labels,on='sample_id',validate='one_to_one')
   actual=actual_confusion(evaluation.y_true.to_numpy(),evaluation.final_prediction.to_numpy(),cfg);estimate=result.estimates
   row={key:estimate.get(key) for key in PAPE_COLUMNS}
   for name in ['fn','fp','recall','f1']:row['actual_'+name]=actual[name]
   expected={k:estimate['estimated_'+k] for k in ['tp','fp','fn','tn','recall','f1','loss','recall_defined','f1_defined','accuracy','accuracy_defined']}
   errors={name+'_absolute_error':abs(expected[name]-actual[name]) for name in ['fn','fp','recall','f1','loss']}
   record={'csv_row_index':index,'target_chunk_id':cid,'target_role':target_role,'mode':mode,'seed':int(seed),'weighting':weighting,
    'n_target':len(target),'expected':expected,'actual':actual,'errors':errors,'member_hash':str(members.member_hash.iloc[0]),
    'input_variant':estimate['input_variant'],'model_hash':em['model_hash'],'threshold_policy_hash':em['threshold_policy_hash'],
    'source_manifest_hash':source_hash,'config_hash':digest(trialconfig),'executed_at':result.diagnostics['created_at'],
    'code_revision':code_revision(),'diagnostics':result.diagnostics,'source_context':source_context,
    'probabilities_path':str((trial/'target_probabilities.csv').relative_to(out)),'probabilities_hash':hash_file(trial/'target_probabilities.csv'),
    'weights_path':str((trial/'source_weights.npz').relative_to(out)),'weights_hash':hash_file(trial/'source_weights.npz')}
   write_json(trial/'record.json',record);records.append(record);rows.append(row)
   print(f'PAPE {target_role} {len(rows)}/{len(chunkids)*len(cfg["pape"]["seeds"])*len(weightings)}: {cid}, seed={seed}, {weighting}',flush=True)
  pd.DataFrame(rows,columns=PAPE_COLUMNS).to_csv(out/'pape_estimates.csv',index=False,encoding='utf-8-sig')
  errors=pd.DataFrame([{'input_variant':r['input_variant'],'seed':r['seed'],**r['errors']} for r in records])
  comparison=errors.groupby('input_variant',sort=False)[[c for c in errors if c.endswith('_absolute_error')]].mean().reset_index()
  mae_file=target_role+'_mae.csv';comparison.to_csv(out/mae_file,index=False,encoding='utf-8-sig')
  info={'schema_version':'pape_document_14_columns_app_adapter_v2','protocol_version':cfg['protocol_version'],'csv_columns':PAPE_COLUMNS,
   'study_name':cfg['study_name'],'target_role':target_role,'source_period':list(role_period(stagecfg,'source')),'target_period':list(role_period(stagecfg,target_role)),
   'source_export_run':str(source_export),'target_export_run':str(target_export),'source_context':source_context,'target_year':target_year,
   'source_export_manifest_hash':hash_file(source_export/'manifest.json'),'target_export_manifest_hash':hash_file(target_export/'manifest.json'),
   'model_hash':em['model_hash'],'threshold_policy_hash':em['threshold_policy_hash'],'source_manifest_hash':source_hash,
   'mode':mode,'n_available_chunks':all_count,'n_selected_chunks':len(chunkids),'partial_validation':len(chunkids)<all_count,
   'n_rows':len(rows),'seeds':cfg['pape']['seeds'],'weightings':weightings,'created_at':now(),'config_hash':digest(cfg),'code_revision':code_revision(),
   'duration_seconds':time.perf_counter()-started,'loss_definition':cfg['loss'],'zero_division':cfg['metrics']['zero_division'],
   'input_selection':'frozen_before_final_test' if lock else 'comparison_only_not_final_selection',
   'selection_run':str(lock_run) if lock else None,'selection_lock_hash':hash_file(lock_run/'selection_lock.json') if lock else None,
   'target_labels_used_in_estimator':False,'actual_columns_role':'evaluation_only_not_operational_features','synthetic_fixture':em['synthetic_fixture'],
   'original_app_rows_kept':True,'final_test_evaluated':target_role=='test',
   'artifact_files':{'pape_estimates.csv':hash_file(out/'pape_estimates.csv'),mae_file:hash_file(out/mae_file)},'records':records,'mae_summary':comparison.to_dict(orient='records')}
  write_json(out/'pape_summary.json',info);write_handoff(out,cfg);verification=verify_pape(out);write_json(out/'verification.json',verification)
  complete(out,{'status':'completed','target_role':target_role,'rows':len(rows),'partial_validation':info['partial_validation'],
   'summary_hash':hash_file(out/'pape_summary.json'),'verification':'PASS','created_at':now()})
  print(f'PAPE {target_role.upper()} COMPLETE: {out}',flush=True);return out
 except Exception as e:write_json(out/'run_status.json',{'status':'failed','error':str(e),'completed_rows':len(rows)});raise

def verify_pape(out):
 out=Path(out);cfg=__import__('yaml').safe_load((out/'resolved.yaml').read_text('utf-8-sig'));summary=read_json(out/'pape_summary.json');role=summary['target_role']
 validate_config(cfg)
 if role!='validation':raise ValueError('Validation-only result required')
 if summary['config_hash']!=digest(cfg) or summary['code_revision']!=code_revision():raise ValueError('Result configuration/code changed')
 if (out/'manifest.json').exists() and read_json(out/'manifest.json')['summary_hash']!=hash_file(out/'pape_summary.json'):raise ValueError('Completed summary changed')
 source_export=Path(summary['source_export_run']);target_export=Path(summary['target_export_run'])
 em=verify_export_files(source_export,['source'] if role=='test' else ['source','validation']);verify_export_files(target_export,[role])
 if hash_file(source_export/'manifest.json')!=summary['source_export_manifest_hash'] or hash_file(target_export/'manifest.json')!=summary['target_export_manifest_hash']:raise ValueError('Export manifest changed')
 source_labels=pd.read_parquet(source_export/'source/labels_for_evaluation.parquet');labels=pd.read_parquet(target_export/role/'labels_for_evaluation.parquet')
 apps=pd.read_parquet(target_export/role/'app_features.parquet');membership=pd.read_parquet(target_export/role/'chunk_membership.parquet')
 table=pd.read_csv(out/'pape_estimates.csv')
 if table.columns.tolist()!=PAPE_COLUMNS or len(table)!=len(summary['records']):raise ValueError('PAPE 14-column contract mismatch')
 for name,expected in summary['artifact_files'].items():
  if hash_file(out/name)!=expected:raise ValueError('Result file changed: '+name)
 for i,record in enumerate(summary['records']):
  if record['csv_row_index']!=i:raise ValueError('CSV row mapping changed')
  row=table.iloc[i];probfile=out/record['probabilities_path'];weightfile=out/record['weights_path']
  if hash_file(probfile)!=record['probabilities_hash'] or hash_file(weightfile)!=record['weights_hash']:raise ValueError('Trial artifact changed')
  p=pd.read_csv(probfile)
  if 'y_true' in p or any(c.startswith('actual_') for c in p):raise ValueError('Truth leaked into probability logs')
  members=membership[membership.target_chunk_id.eq(record['target_chunk_id'])&membership['mode'].eq(record['mode'])].sort_values('position')
  if not len(members) or p.sample_id.tolist()!=members.sample_id.tolist() or digest(p.sample_id.tolist())!=record['member_hash']:raise ValueError('Chunk membership mismatch')
  frozen=members[['sample_id']].merge(apps,on='sample_id',validate='one_to_one')
  if not np.array_equal(p.final_prediction,frozen.final_prediction) or not np.allclose(p.xgb_score,frozen.xgb_score,rtol=0,atol=1e-15):raise ValueError('Frozen detector outputs changed')
  for field in ['model_hash','threshold_policy_hash']:
   if record[field]!=summary[field] or not frozen[field].eq(summary[field]).all():raise ValueError('Detector provenance changed')
  expected=expected_confusion(p.calibrated_probability.to_numpy(),p.final_prediction.to_numpy(),cfg)
  evaluated=p.merge(labels,on='sample_id',validate='one_to_one');actual=actual_confusion(evaluated.y_true.to_numpy(),evaluated.final_prediction.to_numpy(),cfg)
  for key in expected:
   if not np.isclose(record['expected'][key],expected[key],rtol=1e-10,atol=1e-10) or not np.isclose(record['actual'][key],actual[key],rtol=1e-10,atol=1e-10):raise ValueError('Summary confusion/metrics changed')
  if row.target_chunk_id!=record['target_chunk_id'] or row.input_variant!=record['input_variant']:raise ValueError('Row identity mismatch')
  for key in ['fn','fp','recall','f1','loss']:
   if not np.isclose(float(row['estimated_'+key]),expected[key],rtol=1e-10,atol=1e-10):raise ValueError('Estimated metric mismatch')
  for key in ['fn','fp','recall','f1']:
   if not np.isclose(float(row['actual_'+key]),actual[key],rtol=1e-10,atol=1e-10):raise ValueError('Actual metric mismatch')
  with np.load(weightfile,allow_pickle=False) as z:
   if not np.array_equal(z['source_row_index'],np.arange(len(source_labels))) or len(z['raw_weight'])!=len(source_labels) or not np.isfinite(z['raw_weight']).all() or not (z['raw_weight']>0).all() or not np.isfinite(z['clipped_weight']).all() or not (z['clipped_weight']>0).all():raise ValueError('Source weight log invalid')
 handoff_check=verify_handoff(out,cfg)
 return {'handoff':handoff_check,'status':'PASS','rows_checked':len(table),'csv_columns':14,'target_role':role,'synthetic_fixture':summary['synthetic_fixture'],
  'frozen_scores_predictions_and_membership_verified':True,'target_labels_used_in_estimator':False,'final_test_evaluated':role=='test','partial_validation':summary['partial_validation']}

def main():
 from .cli import main as run_cli
 return run_cli()
