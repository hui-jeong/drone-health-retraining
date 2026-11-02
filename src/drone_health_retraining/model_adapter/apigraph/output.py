"""Model-adapter handoff required by work plan 4.1, adapted to APIGraph apps.

One row is one app; window_features is a compatibility filename, while monthly
chunk_membership defines the decision window. Target labels remain separate.
"""
from pathlib import Path
import importlib.metadata,json,shutil,zipfile
import numpy as np
import pandas as pd
import xgboost as xgb
from ...apigraph.common import *

APP_COLUMNS=['sample_id','origin_file','original_row','feature_group','observed_month','project_role',
 'xgb_score','final_prediction','decision_threshold','model_hash','threshold_policy_hash','schema_version','feature_schema_hash']

def build_model_handoff(model_run,export_run,output_dir,*,replay_predictions=True,implementation_commit=None):
    model_run=Path(model_run);export_run=Path(export_run);out=Path(output_dir)
    if out.exists():raise ValueError('Do not overwrite an existing model handoff')
    em=verify_export_files(export_run,['source','validation'])
    mm=read_json(model_run/'baseline_manifest.json');policy=read_json(model_run/'threshold_policy.json')
    schema=read_json(model_run/'feature_schema.json')
    if hash_file(model_run/'model.json')!=mm['model_hash'] or em['model_hash']!=mm['model_hash']:raise ValueError('Model mismatch')
    if not mm['fresh_training'] or mm['warm_start'] or mm['previous_models_loaded'] or mm['train_year']!='2012':raise ValueError('Expected a fresh 2012 model')
    if em['final_test_scored'] or {r['role'] for r in em['roles']}!={'source','validation'}:raise ValueError('Validation-only export required')
    if policy['model_hash']!=mm['model_hash'] or policy['threshold_policy_hash']!=mm['threshold_policy_hash'] or digest({k:v for k,v in policy.items() if k!='threshold_policy_hash'})!=policy['threshold_policy_hash']:raise ValueError('Policy provenance mismatch')
    if schema['feature_schema_hash']!=mm['feature_schema_hash'] or schema['dimension']<1 or digest({k:v for k,v in schema.items() if k!='feature_schema_hash'})!=schema['feature_schema_hash']:raise ValueError('Feature schema mismatch')
    if policy['threshold']!=.5 or policy['comparator']!='gt':raise ValueError('Expected score>0.5')
    dimension=schema['dimension'];binary_cols=[f'binary_{i:04d}' for i in range(dimension)]
    if not mm['synthetic_fixture'] and (dimension!=1159 or mm['n_train']!=30533):raise ValueError('Unexpected real training geometry')
    out.mkdir(parents=True)
    model=xgb.XGBClassifier(n_jobs=2);model.load_model(model_run/'model.json');model.set_params(n_jobs=2)
    apps_all=[];monthly=[];annual=[];source_ids=None;max_diff=0.;roles=[]
    cfg=__import__('yaml').safe_load((export_run/'resolved.yaml').read_text('utf-8-sig'))
    cfg=__import__('copy').deepcopy(cfg)
    for role,start,end in [('source','2013-01','2015-12'),('validation','2016-01','2016-12')]:
        apps=pd.read_parquet(export_run/role/'app_features.parquet')
        binary=pd.read_parquet(export_run/role/'binary_features.parquet')
        labels=pd.read_parquet(export_run/role/'labels_for_evaluation.parquet')
        membership=pd.read_parquet(export_run/role/'chunk_membership.parquet')
        if apps.columns.tolist()!=APP_COLUMNS or apps.sample_id.duplicated().any() or apps.isna().any().any():raise ValueError('Invalid 13-column apps')
        if not apps.project_role.eq(role).all() or not apps.observed_month.between(start,end).all():raise ValueError('Role period mismatch')
        if apps.sample_id.tolist()!=binary.sample_id.tolist() or binary.columns.tolist()!=['sample_id']+binary_cols:raise ValueError('Binary alignment mismatch')
        if not np.isin(binary[binary_cols].to_numpy(),[0,1]).all():raise ValueError('Invalid binary values')
        if not apps.model_hash.eq(mm['model_hash']).all() or not apps.threshold_policy_hash.eq(policy['threshold_policy_hash']).all():raise ValueError('Row model provenance mismatch')
        if not np.array_equal(apps.final_prediction,(apps.xgb_score>.5).astype('uint8')):raise ValueError('Prediction policy mismatch')
        if source_ids is None:source_ids=set(apps.sample_id)
        elif source_ids&set(apps.sample_id):raise ValueError('Source/Validation ID overlap')
        if membership['mode'].unique().tolist()!=['monthly'] or membership.sample_id.duplicated().any() or set(membership.sample_id)!=set(apps.sample_id):raise ValueError('Monthly membership invalid')
        n_months=36 if role=='source' else 12
        if membership.target_chunk_id.nunique()!=n_months:raise ValueError('Missing monthly chunk')
        joined=apps.merge(labels,on='sample_id',validate='one_to_one')
        if len(joined)!=len(apps) or not np.isin(joined.y_true,[0,1]).all():raise ValueError('Label alignment mismatch')
        if replay_predictions:
            for offset in range(0,len(apps),4096):
                block=binary.iloc[offset:offset+4096][binary_cols].to_numpy(dtype=np.float32)
                scores=model.predict_proba(block)[:,1].astype(float)
                diff=float(np.max(np.abs(scores-apps.xgb_score.iloc[offset:offset+len(block)].to_numpy())))
                max_diff=max(max_diff,diff)
                if diff>1e-15:raise ValueError('Saved model scores differ from original export')
        folder=out/role;shutil.copytree(export_run/role,folder)
        shutil.copy2(folder/'app_features.parquet',folder/'window_features.parquet')
        apps_all.append(apps)
        from .model import model_metrics
        role_metrics=model_metrics(joined.y_true.to_numpy(),joined.xgb_score.to_numpy(),cfg)
        annual.append({'role':role,**role_metrics})
        for month,part in joined.groupby('observed_month',sort=True):
            vals=model_metrics(part.y_true.to_numpy(),part.xgb_score.to_numpy(),cfg)
            cid=membership.loc[membership.sample_id.eq(part.sample_id.iloc[0]),'target_chunk_id'].iloc[0]
            monthly.append({'role':role,'observed_month':month,'target_chunk_id':cid,'model_hash':mm['model_hash'],**vals})
        roles.append({'role':role,'period':[start,end],'n_apps':len(apps),'n_months':n_months,'T_malicious':int(joined.y_true.sum()),'F_benign':int(len(joined)-joined.y_true.sum())})
    combined=pd.concat(apps_all,ignore_index=True)
    if combined.sample_id.duplicated().any():raise ValueError('Combined window IDs are not unique')
    combined.to_parquet(out/'window_features.parquet',index=False)
    monthly_df=pd.DataFrame(monthly)
    monthly_df.to_csv(out/'model_monthly_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(annual).to_csv(out/'model_results.csv',index=False,encoding='utf-8-sig')
    confusion_cols=['role','observed_month','target_chunk_id','model_hash','n','tp','fp','fn','tn','f1','accuracy','recall','loss']
    cm=monthly_df[confusion_cols].copy();cm['matrix_axes']='rows=true,columns=predicted';cm['label_order']='0,1';cm['evaluation_only']=True
    cm.to_csv(out/'confusion_matrix.csv',index=False,encoding='utf-8-sig')
    for name in ['model.json','threshold_policy.json','feature_schema.json','internal_metrics.csv','internal_split_manifest.csv','learning_curve.csv']:
        shutil.copy2(model_run/name,out/name)
    shutil.copy2(model_run/'baseline_manifest.json',out/'training_baseline_manifest.json')
    for name in ['manifest.json','resolved.yaml','run_status.json']:
        shutil.copy2(export_run/name,out/name)
    env={n:importlib.metadata.version(n) for n in ['xgboost','numpy','pandas','scikit-learn','PyYAML','pyarrow']}
    baseline={**mm,'handoff_schema':'apigraph_model_adapter_handoff_v1','roles':roles,'feature_row_unit':'android_app',
      'decision_window_unit':'calendar_month','window_features_columns':APP_COLUMNS,'window_features_rows':len(combined),
      'target_truth_separate':True,'gpu_used':False,'device':'cpu','n_jobs':2,'handoff_environment':env,
      'baseline_results':annual,'original_training_manifest_hash':hash_file(out/'training_baseline_manifest.json'),
      'implementation_commit':implementation_commit,'original_training_code_revision':mm['code_revision'],
      'reproduction_commands':['python scripts/run_xgboost_apigraph.py train --config config/apigraph.yaml',
        'python scripts/run_xgboost_apigraph.py export --config config/apigraph.yaml',
        'python scripts/run_xgboost_apigraph.py handoff --model-run MODEL_RUN --export-run EXPORT_RUN --output HANDOFF_DIR'],
      'adaptation_note':'Work-plan Window Feature filename retained; app-level 13-column schema, separate labels/binary sidecar and monthly membership replace drone time-series/latent fields.'}
    write_json(out/'baseline_manifest.json',baseline)
    check={'status':'PASS','source_rows':len(apps_all[0]),'validation_rows':len(apps_all[1]),'window_features_rows':len(combined),
      'window_features_columns':13,'duplicate_sample_ids':0,'required_values_missing':0,'monthly_rows':len(monthly_df),
      'matrix_order':[[ 'tn','fp'],['fn','tp']],'model_predictions_replayed':bool(replay_predictions),
      'prediction_replay_max_absolute_difference':max_diff,'target_labels_separate':True,'final_test_scored':False,
      'synthetic_fixture':mm['synthetic_fixture'],'created_at':now()}
    write_json(out/'verification.json',check)
    (out/'README.md').write_text('''# XGBoost APIGraph model-adapter handoff

window_features.parquet: one Android app per row, 13 columns, Source and Validation combined.
Monthly windows are in chunk_membership.parquet, not artificial 5-second sequences.
Source/Validation labels are in separate labels_for_evaluation.parquet files; Target labels are evaluation-only.
binary_features.parquet preserves the original binary column order. No AE/Hybrid/latent values are fabricated.
baseline_manifest.json contains training provenance, environment, commands and metrics. training_baseline_manifest.json preserves the original manifest.
model_monthly_metrics.csv and confusion_matrix.csv are evaluation-only; they are not operational Gain/RCCDA features.
model_output_manifest.json verifies all delivered file hashes. Data/model/Parquet/ZIP artifacts are excluded from Git.
Original absolute paths inside historical manifests are provenance; copied role files are the portable PAPE input.
''',encoding='utf-8')
    hashes={p.relative_to(out).as_posix():hash_file(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write_json(out/'model_output_manifest.json',{'schema_version':'apigraph_model_adapter_handoff_v1','files':hashes,
      'model_hash':mm['model_hash'],'source_export_manifest_hash':hash_file(export_run/'manifest.json'),
      'implementation_commit':implementation_commit,'created_at':now(),'synthetic_fixture':mm['synthetic_fixture']})
    verify_model_handoff(out)
    return out

def verify_model_handoff(out):
    out=Path(out);manifest=read_json(out/'model_output_manifest.json')
    files={p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()}
    if files!=set(manifest['files'])|{'model_output_manifest.json'}:raise ValueError('Model handoff file list changed')
    for name,expected in manifest['files'].items():
        if hash_file(out/name)!=expected:raise ValueError('Model handoff file changed: '+name)
    frame=pd.read_parquet(out/'window_features.parquet')
    baseline=read_json(out/'baseline_manifest.json');check=read_json(out/'verification.json')
    if frame.columns.tolist()!=APP_COLUMNS or len(frame)!=check['window_features_rows'] or frame.sample_id.duplicated().any():raise ValueError('Window Feature contract invalid')
    if hash_file(out/'model.json')!=baseline['model_hash']:raise ValueError('Model changed')
    if not np.array_equal(frame.final_prediction,(frame.xgb_score>.5).astype('uint8')):raise ValueError('Window prediction mismatch')
    matrix=pd.read_csv(out/'confusion_matrix.csv')
    if not np.allclose(matrix.tp+matrix.fp+matrix.fn+matrix.tn,matrix.n):raise ValueError('Actual confusion totals invalid')
    return {**check,'files_checked':len(manifest['files'])}

def zip_model_handoff(out,zip_path):
    out=Path(out);verify_model_handoff(out)
    with zipfile.ZipFile(zip_path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out).as_posix())
    with zipfile.ZipFile(zip_path) as z:
        m=json.loads(z.read('model_output_manifest.json'))
        for name,expected in m['files'].items():
            if __import__('hashlib').sha256(z.read(name)).hexdigest()!=expected:raise ValueError('Archived model output changed')
    return Path(zip_path)
