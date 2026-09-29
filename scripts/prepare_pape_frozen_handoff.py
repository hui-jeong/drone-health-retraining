"""Select approved rows from existing model outputs without fitting any model."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_pape_real import _branch_info
from src.pape.frozen_handoff import prepare
from src.pape.io import load_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', nargs='+', required=True)
    args = parser.parse_args()
    _branch_info(ROOT)
    result = prepare(load_config(*args.config))
    print(json.dumps({role: {'windows': part['window_count'], 'source_keys': len(part['source_keys']),
        'a1_positive_windows': part['a1_positive_windows']} for role, part in result['exports'].items()}), flush=True)
