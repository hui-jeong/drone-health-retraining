from copy import deepcopy
import json

import pandas as pd
import pytest
import yaml

from scripts.export_pape_release import package
from src.common.metrics import actual_confusion
from src.pape.contracts import PAPE_ESTIMATE_COLUMNS
from src.pape.io import config_hash, write_json
from src.pape.release import read_release, load_release_inputs, sha256
from src.pape.release_export import export_settings, portable_config, write_reference_artifacts


@pytest.fixture
def archive(tmp_path):
    root=tmp_path/'original';(root/'config').mkdir(parents=True)
    roles={'model_fit':[],'model_calibration':[], 'pape_source':[f'{i:03d}' for i in range(7)],'pape_validation':['007']}
    cfg={'seed':23,'loss':{'status':'test','lambda_fn':7.0,'lambda_fp':2.0,'sensitivity':[]},
         'metrics':{'zero_division':1.0},'model_version':'fixture',
         'paths':{'input':'C:/private/data/file.parquet'},'extra_setting':{'preserve':True}}
    (root/'config/resolved.yaml').write_text(yaml.safe_dump(cfg),encoding='utf-8')
    target=pd.DataFrame([{'record_id':i,'run_id':'007','window_id':f'{i}:{j}', 'window_start':float(j),
                         'window_end':float(j+1),'y_true':int(i>0 and j==0),'final_prediction':0}
                        for i in range(36) for j in range(2)])
    target_path=tmp_path/'target.parquet';target.to_parquet(target_path,index=False)
    rows=[]
    for seed in [11,23,37,53,71]:
        for name in ['ae_score','residual_probability','hybrid_score','combined_scores']:
            for weighting in ['uniform','dre']:
                directory=root/f'trials/{seed}/{name}/{weighting}';directory.mkdir(parents=True)
                log=directory/'fixture.parquet';log.write_bytes(b'unchanged-evidence')
                for i in range(36):
                    effective={**deepcopy(cfg),'seed':seed,'target_chunk_id':f'chunk_{i}',
                               'pape':{'input_variant':name,'weighting':weighting}}
                    group=target[target.record_id.eq(i)]
                    metrics=actual_confusion(group.y_true,group.final_prediction,7,2,1)
                    rows.append({'target_chunk_id':f'chunk_{i}','target_record_id':i,'seed':seed,
                        'input_variant':name+'__'+weighting,'input_name':name,'weighting':weighting,
                        'run_id':'007','source_key':str(i),'scenario_id':'A1','model_version':'fixture',
                        'n_target':2,'n_source':4,'config_hash':config_hash(effective),
                        'study_config_hash':config_hash(cfg),'executed_at':'fixture-time',
                        'effective_config':effective,'target_labels_used':False,'csv_row_index':len(rows),
                        'dre_auc':.5,'ess_ratio':1.0,'max_weight':1.0,'warnings':[],
                        'readiness':{'status':'no_diagnostic_flags'},
                        'trial_artifacts':{'source_weights':{'file':log.name,'sha256':sha256(log)}},
                        'loss_sensitivity':[],
                        **{prefix+k:v for prefix in ['actual_','estimated_'] for k,v in metrics.items()}})
    summary={'config_hash':config_hash(cfg),'roles':roles,'records':rows,'seeds':[11,23,37,53,71],
        'model_version':'fixture','executed_at':'time','official_test_used':False,'target_labels_used_in_estimation':False,
        'model_provenance':{'exports':{'pape_validation':{'sha256':sha256(target_path)}}},
        'metrics':[],'seed_claims':{},'loss_sensitivity_metrics':[]}
    write_json(root/'pape_summary.json',summary)
    pd.DataFrame(rows)[PAPE_ESTIMATE_COLUMNS].to_csv(root/'pape_estimates.csv',index=False)
    write_json(root/'run_status.json',{'status':'complete','config_hash':summary['config_hash']})
    write_json(root/'verification_report.json',{'status':'PASS','trial_artifact_hashes_valid':True,'test_used':False})
    return root,target_path,summary,cfg


def test_export_uses_archived_costs_seed_and_zero_division_and_complete_configs(archive,tmp_path):
    root,target,source,cfg=archive
    result=package(root,target,tmp_path/'delivered',code_revision='new-author-commit')
    normal=read_release(result['packages'][0]['directory'])
    assert normal['selected_seed']==23
    assert normal['loss_definition']['lambda_fn']==7
    assert normal['records'][0]['actual_recall']==1
    assert normal['records'][1]['actual_loss']==7
    directory=result['packages'][0]['directory']
    payload=load_release_inputs(directory)[0]
    assert payload['provenance']['code_revision']=='new-author-commit'
    assert payload['provenance']['config_hash']==normal['records'][0]['config_hash']
    from pathlib import Path
    delivered=yaml.safe_load((Path(directory)/'config/resolved_portable.yaml').read_text(encoding='utf-8'))
    assert delivered['metrics']==cfg['metrics'] and delivered['extra_setting']==cfg['extra_setting']
    assert 'C:/private' not in json.dumps(delivered)
    provenance=json.loads((Path(directory)/'config/provenance.json').read_text(encoding='utf-8'))
    assert provenance['portable_config_hash']==config_hash(delivered)
    assert provenance['original_study_config_hash']==config_hash(cfg)
    original_cfg=Path(result['reference_artifacts_directory'])/'config/resolved.yaml'
    assert original_cfg.read_bytes()==(root/'config/resolved.yaml').read_bytes()
    truth=read_release(result['packages'][1]['directory'],allow_oracle=True)
    assert truth['records'][0]['seed'] is None
    assert truth['records'][1]['estimated_loss']==7


def test_export_override_and_invalid_config_hash(archive):
    root,_,source,cfg=archive
    assert export_settings(root,source,11)[1]==11
    with pytest.raises(ValueError,match='not run'):
        export_settings(root,source,99)
    source['config_hash']='changed'
    with pytest.raises(ValueError,match='hash mismatch'):
        export_settings(root,source)


def test_portable_settings_keep_all_nonpath_values_and_disclose_replacements():
    cfg={'x':{'path':'C:\\private\\input.parquet'},'y':['/private/labels.csv',42],
         'metrics':{'zero_division':1},'loss':{'lambda_fn':7,'lambda_fp':2}}
    portable,changes=portable_config(cfg)
    assert portable['metrics']==cfg['metrics'] and portable['loss']==cfg['loss']
    assert len(changes)==2 and cfg['x']['path'].startswith('C:')


def test_changed_raw_artifact_cannot_be_repackaged(archive,tmp_path):
    root,_,source,_=archive
    row=source['records'][0]
    log=root/f"trials/{row['seed']}/{row['input_name']}/{row['weighting']}/fixture.parquet"
    log.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='artifact hash mismatch'):
        write_reference_artifacts(root,tmp_path,source['records'])
