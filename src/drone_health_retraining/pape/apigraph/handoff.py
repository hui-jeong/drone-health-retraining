"""Versioned monthly PAPE outputs for Gain/RCCDA integration.

Actual labels, confusion counts and errors are never included in safe inputs.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from ...apigraph.common import *

def read_app_inputs(export,role):
    folder=Path(export)/role
    apps=pd.read_parquet(folder/'app_features.parquet')
    binary=pd.read_parquet(folder/'binary_features.parquet')
    if binary.sample_id.tolist()!=apps.sample_id.tolist():raise ValueError('Binary sidecar alignment mismatch')
    return apps.merge(binary,on='sample_id',validate='one_to_one',sort=False)

def next_month(month):
    year,mm=map(int,month.split('-'))
    return f'{year+1}-01' if mm==12 else f'{year}-{mm+1:02d}'

FEATURES=['estimated_tp','estimated_fp','estimated_fn','estimated_tn','estimated_recall',
          'estimated_f1','estimated_accuracy','estimated_loss','estimated_loss_per_app',
          'dre_auc','ess_ratio','max_weight','clip_fraction']

def build_payload(summary,cfg):
    safe=[];matrices=[]
    for r in summary['records']:
        e=r['expected'];a=r['actual'];d=r['diagnostics'];context=r['source_context']
        month=r['target_chunk_id'].split(':')[3]
        if len(month)!=7 or r['mode']!='monthly':raise ValueError('Monthly handoff required')
        if context['source_period'][1]>=month or context['source_available_from_month']>month:raise ValueError('Source availability leakage')
        key=digest([r['target_chunk_id'],r['input_variant'],r['seed'],r['model_hash'],r['source_manifest_hash']])
        provenance={'csv_row_index':r['csv_row_index'],'seed':r['seed'],'observed_month':month,
          'decision_available_from_month':next_month(month),'target_role':r['target_role'],'mode':'monthly',
          'n_target':r['n_target'],'n_source':d['n_source'],'member_hash':r['member_hash'],
          'model_hash':r['model_hash'],'threshold_policy_hash':r['threshold_policy_hash'],
          'source_manifest_hash':r['source_manifest_hash'],'source_version':context['source_version'],
          'source_period':context['source_period'],'source_available_from_month':context['source_available_from_month'],
          'config_hash':r['config_hash'],'code_revision':r['code_revision'],'executed_at':r['executed_at']}
        features={**{'estimated_'+k:e[k] for k in ['tp','fp','fn','tn','recall','f1','accuracy','loss']},
          'estimated_loss_per_app':e['loss']/r['n_target'],**{k:d[k] for k in ['dre_auc','ess_ratio','max_weight','clip_fraction']}}
        assert list(features)==FEATURES
        safe.append({'record_id':key,'target_chunk_id':r['target_chunk_id'],'input_variant':r['input_variant'],
          'features':features,'provenance':provenance,
          'readiness':{'status':d['readiness'],'warnings':d['warnings'],'accuracy_guaranteed':False,'policy_action':None},
          'metric_defined':{k:e[k+'_defined'] for k in ['recall','f1','accuracy']},'label_derived_oracle':False})
        matrices.append({'record_id':key,'target_chunk_id':r['target_chunk_id'],'input_variant':r['input_variant'],
          'csv_row_index':r['csv_row_index'],'seed':r['seed'],'observed_month':month,'n_target':r['n_target'],
          'model_hash':r['model_hash'],'source_manifest_hash':r['source_manifest_hash'],
          'matrix_axes':'rows=true,columns=predicted','label_order':'0,1',
          **{'estimated_'+k:e[k] for k in ['tp','fp','fn','tn','recall','f1','accuracy','loss']},
          **{'actual_'+k:a[k] for k in ['tp','fp','fn','tn','recall','f1','accuracy','loss']},
          'evaluation_only':True})
    payload={'schema_version':cfg['integration']['schema_version'],'records':safe,
      'feature_columns':FEATURES,'matrix_label_order':[0,1],'decision_timing':cfg['integration']['decision_timing'],
      'loss_definition':cfg['loss'],'retrain_decision_generated':False,'gain_targets_generated':False,
      'aggregation':'per_seed_no_pooling','synthetic_fixture':summary['synthetic_fixture']}
    flat=[]
    for r in safe:
        p=dict(r['provenance']);p['source_period_start'],p['source_period_end']=p.pop('source_period')
        flat.append({'record_id':r['record_id'],'target_chunk_id':r['target_chunk_id'],
          'input_variant':r['input_variant'],**p,**r['features'],
          **{k+'_defined':v for k,v in r['metric_defined'].items()},'label_derived_oracle':False})
    return payload,pd.DataFrame(flat),pd.DataFrame(matrices)

def write_handoff(out,cfg):
    out=Path(out);summary=read_json(out/'pape_summary.json')
    payload,flat,matrices=build_payload(summary,cfg)
    write_json(out/'rccda_inputs.json',payload)
    flat.to_parquet(out/'gain_inputs.parquet',index=False)
    matrices.to_csv(out/'confusion_matrix.csv',index=False,encoding='utf-8-sig')
    files={name:hash_file(out/name) for name in ['rccda_inputs.json','gain_inputs.parquet','confusion_matrix.csv']}
    write_json(out/'handoff_manifest.json',{'schema_version':cfg['integration']['schema_version'],
      'summary_hash':hash_file(out/'pape_summary.json'),'files':files,'rows':len(flat),
      'safe_files':['rccda_inputs.json','gain_inputs.parquet'],'evaluation_only_files':['confusion_matrix.csv','pape_estimates.csv','pape_summary.json'],
      'source_context':summary['source_context'],'created_at':now()})

def verify_handoff(out,cfg):
    out=Path(out);summary=read_json(out/'pape_summary.json');manifest=read_json(out/'handoff_manifest.json')
    if manifest['summary_hash']!=hash_file(out/'pape_summary.json'):raise ValueError('Handoff summary changed')
    for name,expected in manifest['files'].items():
        if hash_file(out/name)!=expected:raise ValueError('Handoff artifact changed: '+name)
    payload,flat,matrices=build_payload(summary,cfg)
    if read_json(out/'rccda_inputs.json')!=payload:raise ValueError('RCCDA input contract mismatch')
    pd.testing.assert_frame_equal(pd.read_parquet(out/'gain_inputs.parquet'),flat)
    observed=pd.read_csv(out/'confusion_matrix.csv',dtype={'label_order':str})
    pd.testing.assert_frame_equal(observed,matrices,check_dtype=False,rtol=1e-10,atol=1e-10)
    if flat.record_id.duplicated().any():raise ValueError('Duplicate handoff record')
    if any(c.startswith('actual_') or c in ['y_true','realized_gain','future_loss'] for c in flat):raise ValueError('Truth leakage into Gain inputs')
    for prefix in ['estimated','actual']:
        if not np.allclose(sum(matrices[prefix+'_'+k] for k in ['tp','fp','fn','tn']),matrices.n_target):raise ValueError('Confusion matrix total mismatch')
    return {'status':'PASS','rows':len(flat),'safe_inputs_without_actual_labels':True,'matrix_axes':'true_by_predicted_0_1'}
