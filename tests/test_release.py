import json
from copy import deepcopy

import pandas as pd
import pytest

from src.common.metrics import confusion_metrics
from src.pape.contracts import PAPE_ESTIMATE_COLUMNS
from src.pape.release import FEATURES, VERSION, load_release_inputs, read_release, sha256


def make_bundle(root, *, oracle=False, review=False):
    root.mkdir()
    actual=confusion_metrics(3,1,2,4,1,1)
    estimated=actual if oracle else confusion_metrics(2.5,1.5,1.5,4.5,1,1)
    kind='lstm_ae_true_data' if oracle else 'pape_estimates'
    row={'target_chunk_id':'fixture::007::record_01','input_variant':'oracle' if oracle else 'hybrid_score__dre',
         **{'actual_'+k:v for k,v in actual.items()},**{'estimated_'+k:v for k,v in estimated.items()},
         'dre_auc':None if oracle else .6,'ess_ratio':None if oracle else .8,'max_weight':None if oracle else 2.0,
         'seed':None if oracle else 11,'source_key':'fixture/007','run_id':'007','scenario_id':'A1',
         'model_version':'fixed_model','n_target':10,'csv_row_index':0,'target_labels_used':oracle,
         'warnings':['needs_review'] if review else [],
         'readiness':{'status':'requires_review' if review else 'no_diagnostic_flags','production_approved':False}}
    summary={'release_version':VERSION,'artifact_kind':kind,'target_labels_used_for_inputs':oracle,
             'official_test_used':False,'chronology_verified':False,'loss_definition':{'lambda_fn':1,'lambda_fp':1},'records':[row]}
    (root/'pape_summary.json').write_text(json.dumps(summary),encoding='utf-8')
    pd.DataFrame([row])[PAPE_ESTIMATE_COLUMNS].to_csv(root/'pape_estimates.csv',index=False)
    pd.DataFrame([{'target_chunk_id':row['target_chunk_id']}]).to_csv(root/'chunk_index.csv',index=False)
    seal(root,kind)
    return root


def seal(root,kind):
    m={'release_version':VERSION,'artifact_kind':kind,'files':{p.name:sha256(p) for p in root.iterdir() if p.name!='manifest.json'}}
    (root/'manifest.json').write_text(json.dumps(m),encoding='utf-8')


def test_release_separates_oracle_from_normal_inputs(tmp_path):
    normal=make_bundle(tmp_path/'normal')
    truth=make_bundle(tmp_path/'truth',oracle=True)
    a=load_release_inputs(normal)[0]
    assert set(a['features'])==set(FEATURES)
    assert not any(k.startswith('actual_') for k in a['features'])
    assert a['label_derived_oracle'] is False
    with pytest.raises(ValueError,match='Oracle'):
        load_release_inputs(truth)
    b=load_release_inputs(truth,allow_oracle=True)[0]
    assert b['target_chunk_id']==a['target_chunk_id']
    assert b['features']['estimated_fn']==2
    assert b['label_derived_oracle'] is True
    assert b['features']['dre_auc'] is None


def test_changed_file_rejected(tmp_path):
    root=make_bundle(tmp_path/'release')
    with (root/'pape_estimates.csv').open('a') as f:f.write('bad')
    with pytest.raises(ValueError,match='Changed'):
        read_release(root)


def test_hashes_do_not_replace_semantic_csv_validation(tmp_path):
    root=make_bundle(tmp_path/'release')
    frame=pd.read_csv(root/'pape_estimates.csv');frame.loc[0,'estimated_fn']=8
    frame.to_csv(root/'pape_estimates.csv',index=False)
    seal(root,'pape_estimates')
    with pytest.raises(ValueError,match='CSV/JSON'):
        read_release(root)


def test_review_flags_require_explicit_research_opt_in(tmp_path):
    root=make_bundle(tmp_path/'release',review=True)
    with pytest.raises(ValueError,match='requires review'):
        load_release_inputs(root)
    result=load_release_inputs(root,allow_review=True)
    assert result[0]['warnings']==['needs_review']
    assert result[0]['chronology_verified'] is False


@pytest.mark.parametrize('field,value',[('run_id','008'),('target_labels_used',True),('n_target',99)])
def test_invalid_release_provenance_rejected(tmp_path,field,value):
    root=make_bundle(tmp_path/'release')
    s=json.loads((root/'pape_summary.json').read_text(encoding='utf-8'))
    s['records'][0][field]=value
    (root/'pape_summary.json').write_text(json.dumps(s),encoding='utf-8')
    seal(root,'pape_estimates')
    with pytest.raises(ValueError):
        read_release(root)


def test_extra_evaluation_values_do_not_enter_features(tmp_path):
    root=make_bundle(tmp_path/'release')
    s=json.loads((root/'pape_summary.json').read_text(encoding='utf-8'))
    baseline=deepcopy(load_release_inputs(root))
    s['records'][0].update(y_true='SECRET',data_class='SECRET',future_keep_loss='SECRET',realized_gain='SECRET')
    (root/'pape_summary.json').write_text(json.dumps(s),encoding='utf-8');seal(root,'pape_estimates')
    assert load_release_inputs(root)==baseline


@pytest.mark.parametrize('oracle', [False, True])
def test_release_preserves_execution_provenance_separately_from_features(tmp_path, oracle):
    root=make_bundle(tmp_path/'release',oracle=oracle)
    s=json.loads((root/'pape_summary.json').read_text(encoding='utf-8'))
    s.update(code_revision='revision',created_at='created',original_study_config_hash='study',
             bundle_id='bundle',selection_status='pending')
    hash_key='reference_pape_config_hash' if oracle else 'config_hash'
    s['records'][0].update({hash_key:'trial','study_config_hash':'study','executed_at':'executed'})
    (root/'pape_summary.json').write_text(json.dumps(s),encoding='utf-8');seal(root,s['artifact_kind'])
    record=load_release_inputs(root,allow_oracle=oracle)[0]
    p=record['provenance']
    assert p[hash_key]=='trial' and p['study_config_hash']=='study'
    assert p['executed_at']=='executed' and p['code_revision']=='revision'
    assert p['seed'] is None if oracle else p['seed']==11
    assert not any(k.startswith('actual_') for k in p)
    assert 'config_hash' not in record['features']
