from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from drone_health_retraining.apigraph.common import *
from drone_health_retraining.model_adapter.apigraph.model import train_stage,export_stage
from drone_health_retraining.model_adapter.apigraph.output import build_model_handoff,verify_model_handoff,zip_model_handoff

def main():
    p=argparse.ArgumentParser(description='Fresh XGBoost APIGraph adapter; Validation only')
    p.add_argument('stage',choices=['train','export','handoff','verify-handoff'])
    p.add_argument('--config',default='config/apigraph.yaml');p.add_argument('--model-run');p.add_argument('--export-run')
    p.add_argument('--output');p.add_argument('--zip-path');p.add_argument('--implementation-commit')
    a=p.parse_args()
    if a.stage in ['handoff','verify-handoff'] and not a.output:p.error('--output is required')
    if a.stage=='verify-handoff':print(verify_model_handoff(a.output));return
    if a.stage=='handoff':
        if not a.model_run or not a.export_run:p.error('--model-run and --export-run are required')
        result=build_model_handoff(a.model_run,a.export_run,a.output,implementation_commit=a.implementation_commit)
        if a.zip_path:zip_model_handoff(result,a.zip_path)
        print(result);return
    cfg=load_config(a.config)
    print(train_stage(cfg) if a.stage=='train' else export_stage(cfg,a.model_run))

if __name__=='__main__':main()
