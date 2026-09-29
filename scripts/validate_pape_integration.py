"""Verify only the PAPE side of the future Gain/RCCDA integration (no policy stub)."""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pape.integration import read_verified_handoff, load_pape_inputs
from src.pape.io import utc_now, write_json


def validate_pape_integration(output_dir):
    summary = read_verified_handoff(output_dir)
    groups = sorted({(r["seed"], r["input_variant"]) for r in summary["records"] if r["split"] == "validation"})
    n_rows = n_review = strict_rejections = 0
    examples = []
    for seed, variant in groups:
        payloads = load_pape_inputs(output_dir, seed=seed, input_variant=variant, allow_review=True)
        n_rows += len(payloads)
        review = sum(p.readiness["status"] == "requires_review" for p in payloads)
        n_review += review
        try:
            load_pape_inputs(output_dir, seed=seed, input_variant=variant)
        except ValueError as exc:
            if not review or not str(exc).startswith("PAPE estimate requires review:"):
                raise
            strict_rejections += 1
        else:
            if review:
                raise AssertionError("Strict integration accepted a warning-bearing batch")
        examples.append(asdict(payloads[0]))
    report = {
        "status": "pape_boundary_pass", "executed_at": utc_now(),
        "config_hash": summary["config_hash"], "checked_validation_rows": n_rows,
        "rows_requiring_review": n_review, "strict_batches_rejected": strict_rejections,
        "checks": ["mandatory_file_hashes", "exact_csv_schema", "csv_json_row_mapping",
                   "label_free_feature_allowlist", "explicit_seed_and_input_variant",
                   "configured_loss_and_provenance", "official_test_exclusion", "warning_propagation"],
        "examples_not_a_time_ordered_stream": examples,
        "gain_rccda_implemented": False, "actual_downstream_integration_executed": False,
        "policy_decisions_generated": False, "production_approved": False,
        "remaining": ["Gain/RCCDA implementation and agreed input mapping",
                      "verified Run chronology", "final reference/input decision and failure handling agreement"],
    }
    write_json(Path(output_dir) / "integration_validation.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = validate_pape_integration(PROJECT_ROOT / args.output_dir)
    print({k: result[k] for k in ["status", "checked_validation_rows", "rows_requiring_review",
                                "actual_downstream_integration_executed"]})
