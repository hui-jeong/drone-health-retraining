import numpy as np
import pandas as pd
from .common import *

def groups_of(x):
 return np.array([__import__('hashlib').sha256(row.tobytes()).hexdigest() for row in x])
def verify_data(cfg):
 root=local_path(cfg['paths']['data']);result={}
 for name,expected in DATA_HASHES.items():
  path=root/name;actual=hash_file(path)
  if actual!=expected:raise ValueError('Dataset SHA256 mismatch: '+name)
  result[name]=actual
 return result
def load_train(cfg):
 path=local_path(cfg['paths']['data'])/'train.npz'
 with np.load(path,allow_pickle=False) as z:x,y,t=z['feature'],z['binary_label'],z['timestamp']
 if x.ndim!=2 or x.shape[1]!=cfg['data']['features'] or y.shape!=(len(x),) or not np.isin(x,[0,1]).all() or not np.isin(y,[0,1]).all() or not np.all(t=='2012-XX'):raise ValueError('Invalid 2012 data')
 return x,y,groups_of(x)
def load_role(cfg,role,allow_final_test=False):
 if role=='test':raise ValueError('Validation-only experiment: Test access disabled')
 start,end=role_period(cfg,role);parts=[];features=[]
 for name in (['val.npz','test.npz'] if role=='source' else ['test.npz']):
  path=local_path(cfg['paths']['data'])/name;fh=hash_file(path)
  with np.load(path,allow_pickle=False) as z:
   t=z['timestamp'];ix=np.flatnonzero((t>=start)&(t<=end));x=z['feature'][ix];y=z['binary_label'][ix];months=t[ix]
  if not len(ix):continue
  if x.ndim!=2 or x.shape!=(len(y),cfg['data']['features']) or not np.isin(x,[0,1]).all() or not np.isin(y,[0,1]).all():raise ValueError('Invalid app features/labels')
  parts.append(pd.DataFrame({'sample_id':[f'{fh}:{int(i)}' for i in ix],
   'origin_file':name,'original_row':ix,'feature_group':groups_of(x),'observed_month':months,
   'project_role':role,'y_true':y}));features.append(x)
 if not parts:raise ValueError('No data in requested role period')
 frame=pd.concat(parts,ignore_index=True)
 if frame.sample_id.duplicated().any():raise ValueError('Duplicate app row ID')
 return np.concatenate(features).astype(np.float32),frame
def audit(cfg,check_hash=True):
 if check_hash:verify_data(cfg)
 annual={};file_info=[];metadata=[]
 for name in DATA_HASHES:
  with np.load(local_path(cfg['paths']['data'])/name,allow_pickle=False) as z:
   x,y,t=z['feature'],z['binary_label'],z['timestamp']
  if x.shape!=(len(y),cfg['data']['features']) or not np.isin(x,[0,1]).all() or not np.isin(y,[0,1]).all():raise ValueError('Invalid data geometry')
  for year in sorted(set(s[:4] for s in t)):
   ix=np.char.startswith(t,year);a=annual.setdefault(year,{'n':0,'malicious_T':0,'benign_F':0})
   a['n']+=int(ix.sum());a['malicious_T']+=int(y[ix].sum());a['benign_F']+=int((y[ix]==0).sum())
  file_info.append({'file':name,'n':len(y),'sha256':hash_file(local_path(cfg['paths']['data'])/name)})
  metadata.append(pd.DataFrame({'month':t,'y':y,'group':groups_of(x)}))
 for a in annual.values():a['malicious_fraction']=a['malicious_T']/a['n']
 all_rows=pd.concat(metadata,ignore_index=True);prior=set();roles=[]
 for role in ['train','source','validation','test']:
  if role=='train':part=all_rows[all_rows.month.str.startswith('2012-')];start=end='2012-XX'
  else:
   start,end=role_period(cfg,role);part=all_rows[all_rows.month.between(start,end)]
  n=len(part);t=int(part.y.sum());unique=set(part['group'])
  roles.append({'role':role,'period_start':start,'period_end':end,'n':n,'malicious_T':t,'benign_F':n-t,
   'malicious_fraction':t/n if n else None,'unique_feature_groups':len(unique),'overlap_with_earlier_roles_rows':int(part['group'].isin(prior).sum()),
   'monthly_chunks':int(part.month.nunique()) if role!='train' else None,
   'window_unit':'calendar_month' if role!='train' else None})
  prior.update(unique)
 return {'protocol_version':cfg['protocol_version'],'annual':annual,'roles':roles,'files':file_info,'rows_not_assigned_to_roles':len(all_rows)-sum(r['n'] for r in roles),
  'labels_profiled_for_counts_only':True,'test_performance_evaluated':False,'model_inference_performed':False}

def build_membership(frame,cfg,mode):
 if mode!='monthly':raise ValueError('Only monthly windows are supported')
 if frame.project_role.nunique()!=1:raise ValueError('One role per membership')
 blocks=[(str(month),part.index.to_numpy()) for month,part in frame.groupby('observed_month',sort=True)]
 rows=[]
 for ordinal,(name,ix) in enumerate(blocks):
  part=frame.loc[ix];mh=digest(part.sample_id.tolist());role=str(frame.project_role.iloc[0])
  # Membership identity is independent of model/policy; those are separately verified.
  context=digest([role,mode,mh,cfg['chunks']]);cid=f'apigraph:{role}:{mode}:{name}:{context[:20]}'
  for position,i in enumerate(ix):rows.append({'target_chunk_id':cid,'mode':mode,'ordinal':ordinal,
   'sample_id':frame.loc[i,'sample_id'],'position':position,'member_hash':mh,
   'is_partial':False})
 table=pd.DataFrame(rows)
 if len(table)!=len(frame) or table.sample_id.duplicated().any():raise ValueError('Membership coverage error')
 return table
