"""Small integration boundary; future Gain/RCCDA implementations stay external."""
from pathlib import Path
import pandas as pd
from ...apigraph.common import read_json
from .handoff import verify_handoff

def load_operational_inputs(result_run,*,seed,weighting,expected_model_hash=None):
    """Select exactly one method and seed; preserve one row per month."""
    result_run=Path(result_run)
    import yaml
    cfg=yaml.safe_load((result_run/'resolved.yaml').read_text('utf-8-sig'))
    verify_handoff(result_run,cfg)
    if weighting not in ['uniform','dre']:raise ValueError('Select uniform or dre explicitly')
    frame=pd.read_parquet(result_run/'gain_inputs.parquet')
    frame=frame[frame.seed.eq(seed)&frame.input_variant.eq('all_binary__'+weighting)].copy()
    if frame.empty:raise ValueError('Requested seed/method was not executed')
    if frame.target_chunk_id.duplicated().any() or frame.observed_month.duplicated().any():raise ValueError('Duplicate operational month')
    if expected_model_hash is not None and not frame.model_hash.eq(expected_model_hash).all():raise ValueError('Rescore Source/Target after detector changes')
    if frame.model_hash.nunique()!=1:raise ValueError('Mixed detector versions')
    return frame.sort_values('observed_month').reset_index(drop=True)

def validate_gain_timeline(candidate_month,training_months,future_months,horizon_months):
    """Future-loss labels become usable only after the entire H-month horizon."""
    from .handoff import next_month
    if horizon_months<1:raise ValueError('H must be positive months')
    expected=[];month=candidate_month
    for _ in range(horizon_months):month=next_month(month);expected.append(month)
    if list(future_months)!=expected:raise ValueError('Future evaluation must be the next H consecutive months')
    if any(month>candidate_month for month in training_months):raise ValueError('Future data in candidate training')
    return {'future_chunk_months':expected,'gain_label_available_from_month':next_month(expected[-1])}
