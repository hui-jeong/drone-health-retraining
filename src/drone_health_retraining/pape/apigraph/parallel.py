"""Bounded parallel computation with deterministic month/seed/method output order."""
import copy
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from .handoff import read_app_inputs
from .pape import estimate_target_performance

SOURCE=None
TARGET=None
MEMBERS=None

def _initialize(source_export,target_export,target_role):
    global SOURCE,TARGET,MEMBERS
    from pathlib import Path
    source_export=Path(source_export);target_export=Path(target_export)
    SOURCE=read_app_inputs(source_export,'source').merge(pd.read_parquet(source_export/'source/labels_for_evaluation.parquet'),on='sample_id',validate='one_to_one')
    TARGET=read_app_inputs(target_export,target_role).set_index('sample_id',drop=False)
    MEMBERS=pd.read_parquet(target_export/target_role/'chunk_membership.parquet')
    # Target labels are never read by the worker.

def _compute(config):
    members=MEMBERS[MEMBERS.target_chunk_id.eq(config['target_chunk_id'])].sort_values('position')
    target=TARGET.loc[members.sample_id.tolist()].reset_index(drop=True)
    return estimate_target_performance(SOURCE,target,config)

def trial_results(cfg,stagecfg,source_export,target_export,target_role,chunkids,weightings,source,target_pool,membership):
    tasks=[]
    for cid in chunkids:
        for seed in cfg['pape']['seeds']:
            for weighting in weightings:
                config=copy.deepcopy(stagecfg)
                config.update(seed=int(seed),target_chunk_id=cid,target_role=target_role)
                config['pape']['weighting']=weighting
                tasks.append((cid,int(seed),weighting,config))
    workers=cfg['execution']['workers']
    if workers==1:
        for task in tasks:
            cid,seed,weighting,config=task
            members=membership[membership.target_chunk_id.eq(cid)].sort_values('position')
            target=members[['sample_id']].merge(target_pool,on='sample_id',validate='one_to_one')
            yield (*task,estimate_target_performance(source,target,config))
        return
    with ProcessPoolExecutor(max_workers=workers,initializer=_initialize,initargs=(str(source_export),str(target_export),target_role)) as pool:
        waiting=deque();it=iter(tasks)
        for _ in range(min(len(tasks),2*workers)):
            task=next(it);waiting.append((task,pool.submit(_compute,task[3])))
        while waiting:
            task,future=waiting.popleft()
            result=future.result()
            try:
                following=next(it);waiting.append((following,pool.submit(_compute,following[3])))
            except StopIteration:pass
            yield (*task,result)
