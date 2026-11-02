import argparse
from ...apigraph.common import *
from .pipeline import pape_stage,verify_pape

def main():
    p=argparse.ArgumentParser(description='APIGraph monthly PAPE; consume model-adapter export, Validation only')
    p.add_argument('stage',choices=['run','verify']);p.add_argument('--config',default='config/apigraph.yaml')
    p.add_argument('--export-run');p.add_argument('--validation-run');p.add_argument('--max-chunks',type=int,default=0)
    a=p.parse_args();cfg=load_config(a.config)
    if a.stage=='run':print(pape_stage(cfg,export_run=a.export_run,max_chunks=a.max_chunks))
    else:
        out=latest(cfg,'pape_validation',a.validation_run);result=verify_pape(out)
        write_json(out/'verification.json',result);print(result)

if __name__=='__main__':main()
