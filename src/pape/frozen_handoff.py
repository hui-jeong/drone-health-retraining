"""Read-only exported detector outputs; no detector/scaler fitting entrypoint."""
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.pape.contracts import validate_window_features
from src.pape.handoff import validate_window_geometry
from src.pape.io import config_hash, utc_now, write_json


def validate_roles(config):
    spec = config['fixed_reference']
    roles = spec['roles']
    if spec['protocol'] != 'frozen_export_source_v1':
        raise ValueError('Not a frozen model-export protocol')
    if roles.get('model_fit') or roles.get('model_calibration'):
        raise ValueError('Frozen export protocol forbids detector fit/calibration')
    source = roles['pape_source']
    target = roles['pape_validation']
    allowed = set(config['split']['baseline_train_run_ids'])
    mode = spec.get('target_mode', 'validation_007')
    source_allowed = allowed | ({'007'} if mode == 'target005_with_007_source' else set())
    if not source or len(source) != len(set(source)) or not set(source) <= source_allowed:
        raise ValueError('Source must contain unique allowed Train Run IDs')
    if mode == 'validation_007':
        if target != ['007']:
            raise ValueError('007 must remain exclusively the validation Target')
    elif mode == 'heldout_train_run':
        if len(target) != 1 or not set(target) <= allowed:
            raise ValueError('Holdout Target must be one original Train Run')
    elif mode == 'target005_with_007_source':
        if target != ['005'] or set(source) != {'000', '001', '002', '003', '004', '006', '007'}:
            raise ValueError('007 Source reuse is restricted to the approved Target 005 comparison')
        if '007' not in config['split']['baseline_validation_run_ids']:
            raise ValueError('007 must retain its original validation split')
    else:
        raise ValueError('Unknown frozen Target mode')
    if set(source) & set(target):
        raise ValueError('Source and Target Run IDs overlap')
    if (set(source) | set(target)) & set(config['split']['official_test_run_ids']):
        raise ValueError('Official Test is excluded')


def original_run_splits(config):
    """PAPE roles never rewrite the delivered detector's split metadata."""
    return {**{run: 'train' for run in config['split']['baseline_train_run_ids']},
            **{run: 'validation' for run in config['split']['baseline_validation_run_ids']}}


def original_role_splits(config):
    mapping = original_run_splits(config)
    roles = config['fixed_reference']['roles']
    result = []
    for role in ['pape_source', 'pape_validation']:
        splits = sorted({mapping[run] for run in roles[role]})
        result.append((role, splits[0] if len(splits) == 1 else splits))
    return result


def validate_frames(source, target, config):
    validate_roles(config)
    frames = {'pape_source': source, 'pape_validation': target}
    for role, _ in original_role_splits(config):
        frame = frames[role]
        if set(frame.run_id) != set(config['fixed_reference']['roles'][role]):
            raise ValueError('Export contains incorrect or missing Run IDs')
        if not frame['split'].eq(frame.run_id.map(original_run_splits(config))).all():
            raise ValueError('Original export split changed')
        if frame.source_key.nunique() != config['fixed_reference']['expected_source_keys'][role]:
            raise ValueError('Unexpected original group count')
        validate_window_features(frame)
        validate_window_geometry(frame, config, check_runs=True)
        identities = frame.groupby('record_id')[['source_key', 'run_id', 'scenario_id', 'data_class']].nunique()
        if identities.gt(1).any().any():
            raise ValueError('Record identifier mixes files or groups')
        if frame.groupby('source_key').data_class.agg(set).map(lambda x: x != {'normal', 'abnormal'}).any():
            raise ValueError('Missing normal/derived pair')
    if set(source.source_key) & set(target.source_key):
        raise ValueError('Source and Target overlap')
    if set(source.window_id) & set(target.window_id):
        raise ValueError('Window IDs overlap')


def prepare(config):
    from src.pape.fixed_reference import sha256
    validate_roles(config)
    spec = config['fixed_reference']
    root = Path(spec['export_root'])
    if root.exists():
        raise FileExistsError('Frozen export directory already exists; do not overwrite')
    input_path = Path(config['pape_data']['window_features'])
    manifest_path = Path(config['pape_data']['baseline_manifest'])
    # Hashing is opaque integrity checking. Only allowed Train/Validation rows are decoded below.
    before = {'window_features': sha256(input_path), 'baseline_manifest': sha256(manifest_path)}
    incoming = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if incoming['model_version'] != config['model_version']:
        raise ValueError('Configured detector version differs from supplied manifest')
    frames = {}
    mapping = original_run_splits(config)
    for role, splits in original_role_splits(config):
        splits = [splits] if isinstance(splits, str) else splits
        filters = [[('split', '=', split), ('run_id', 'in',
                    [run for run in spec['roles'][role] if mapping[run] == split])]
                   for split in splits]
        frames[role] = pq.read_table(input_path, filters=filters).to_pandas()
    validate_frames(frames['pape_source'], frames['pape_validation'], config)
    if before != {'window_features': sha256(input_path), 'baseline_manifest': sha256(manifest_path)}:
        raise ValueError('Original handoff changed during filtered extraction')
    root.mkdir(parents=True)
    manifest = {'protocol': spec['protocol'], 'model_version': config['model_version'],
        'target_mode': spec.get('target_mode', 'validation_007'),
        'original_role_splits': dict(original_role_splits(config)),
        'executed_at': utc_now(), 'fresh_training': False, 'detector_fit_performed': False,
        'role_config_hash': config_hash(spec['roles']), 'original_file_sha256': before,
        'original_handoff': {k: incoming.get(k) for k in
            ['model_version', 'integration_timestamp_utc', 'seed', 'company_config_sha256']},
        'checkpoint_identity_verified': False,
        'provenance_limit': 'The delivered export has no exact checkpoint hash or fit source_key list. Frozen export identity is verified, not checkpoint lineage.',
        'official_test_rows_selected': False, 'exports': {}}
    manifest['validation_007_reused_as_pape_source'] = '007' in spec['roles']['pape_source']
    for role, frame in frames.items():
        folder = root / role
        folder.mkdir()
        path = folder / 'window_features.parquet'
        frame.to_parquet(path, index=False)
        manifest['exports'][role] = {'sha256': sha256(path), 'window_count': len(frame),
            'source_keys': sorted(frame.source_key.unique()),
            'record_ids': sorted(frame.record_id.unique().tolist()),
            'run_ids': sorted(frame.run_id.unique()),
            'a1_positive_windows': int(frame.loc[frame.scenario_id.eq('A1'), 'y_true'].sum())}
    write_json(root / 'baseline_manifest.json', manifest)
    return manifest


def load(config):
    from src.pape.fixed_reference import sha256
    validate_roles(config)
    root = Path(config['fixed_reference']['export_root'])
    manifest = json.loads((root / 'baseline_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('protocol') != 'frozen_export_source_v1' or manifest.get('fresh_training') is not False:
        raise ValueError('Only the unchanged delivered-model export is allowed')
    if manifest['role_config_hash'] != config_hash(config['fixed_reference']['roles']):
        raise ValueError('Frozen Source/Target role configuration changed')
    if manifest.get('target_mode', 'validation_007') != config['fixed_reference'].get('target_mode', 'validation_007'):
        raise ValueError('Frozen Target mode changed')
    if manifest['model_version'] != config['model_version']:
        raise ValueError('Detector version changed')
    frames = {}
    for role in ['pape_source', 'pape_validation']:
        path = root / role / 'window_features.parquet'
        expected = manifest['exports'][role]
        if sha256(path) != expected['sha256']:
            raise ValueError('Frozen feature export hash mismatch')
        frame = pd.read_parquet(path)
        if (len(frame) != expected['window_count'] or sorted(frame.source_key.unique()) != expected['source_keys']
                or sorted(frame.record_id.unique().tolist()) != expected['record_ids']):
            raise ValueError('Incomplete frozen feature export')
        frames[role] = frame
    validate_frames(frames['pape_source'], frames['pape_validation'], config)
    return frames['pape_source'], frames['pape_validation'], manifest
