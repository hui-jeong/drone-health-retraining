from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


class CompanyModelAdapter:
    """Read-only adapter around Drone Health Hybrid LSTM-AE v2.3.

    The company source tree is kept unchanged. This adapter reproduces the same
    preprocessing/window scoring/postprocessing path and additionally exports
    the 32-D AE latent vector required by the retraining project.
    """

    def __init__(
        self,
        company_model_root: str | Path,
        company_config_path: str | Path,
        model_output_root: str | Path,
    ) -> None:
        self.company_model_root = Path(company_model_root).resolve()
        self.company_config_path = Path(company_config_path).resolve()
        self.model_output_root = Path(model_output_root).resolve()
        self._install_company_path()
        self._load_modules()
        self.config = self.company_utils.load_yaml(self.company_config_path)
        self._load_artifacts()

    def _install_company_path(self) -> None:
        if not (self.company_model_root / "drone_ae").is_dir():
            raise FileNotFoundError(f"drone_ae package not found: {self.company_model_root}")
        text = str(self.company_model_root)
        if text not in sys.path:
            sys.path.insert(0, text)

    def _load_modules(self) -> None:
        self.company_utils = importlib.import_module("drone_ae.utils")
        self.hybrid = importlib.import_module("drone_ae.hybrid")
        self.engine = importlib.import_module("drone_ae.engine")
        self.calibration = importlib.import_module("drone_ae.calibration")
        self.postprocess = importlib.import_module("drone_ae.postprocess")

    def _load_artifacts(self) -> None:
        artifacts = self.model_output_root / "artifacts"
        checkpoint = self.model_output_root / "checkpoints" / "best_lstm_ae.pt"
        required = [
            artifacts / "feature_columns.json",
            artifacts / "hybrid_policy.json",
            artifacts / "threshold.json",
            artifacts / "residual_classifier.joblib",
            checkpoint,
        ]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "Company model artifacts are incomplete. Run company run_all.py first or provide the output folder.\n"
                + "\n".join(missing)
            )

        self.features = list(self.company_utils.read_json(artifacts / "feature_columns.json")["features"])
        self.hybrid_policy = self.company_utils.read_json(artifacts / "hybrid_policy.json")
        self.threshold_payload = self.company_utils.read_json(artifacts / "threshold.json")
        self.residual_classifier = self.hybrid.load_residual_classifier(self.model_output_root)

        train_cfg = self.config.get("training", {})
        self.device = self.company_utils.resolve_device(str(train_cfg.get("device", "auto")))
        self.amp_enabled = bool(train_cfg.get("amp", True)) and self.device.type == "cuda"
        self.model, self.checkpoint = self.engine.load_model(checkpoint, self.device)
        self.model.eval()

    def _loader(self, dataset: Any) -> DataLoader:
        train_cfg = self.config.get("training", {})
        num_workers = int(train_cfg.get("num_workers", 0))
        kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": int(train_cfg.get("evaluation_batch_size", train_cfg.get("batch_size", 512))),
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": bool(train_cfg.get("pin_memory", True)) and torch.cuda.is_available(),
            "drop_last": False,
        }
        if num_workers > 0:
            kwargs["persistent_workers"] = bool(train_cfg.get("persistent_workers", True))
            kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 2))
        return DataLoader(**kwargs)

    @staticmethod
    def _forward_with_latent(model: Any, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Mirrors drone_ae.model.LSTMAutoencoder.forward exactly, while exposing latent.
        _, (hidden, _) = model.encoder(x)
        latent = model.to_latent(hidden[-1])
        repeated = latent.unsqueeze(1).expand(-1, x.size(1), -1)
        decoded, _ = model.decoder(repeated)
        reconstructed = model.output_layer(decoded)
        return reconstructed, latent

    @staticmethod
    def _merge_record_metadata(frame: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
        wanted = [
            "record_id",
            "table_path",
            "metadata_path",
            "data_class",
            "anomaly_type",
            "anomaly_severity",
            "scenario_id",
            "scenario_name",
            "scenario_category",
            "source_severity",
            "run_id",
            "source_key",
            "split",
        ]
        columns = [c for c in wanted if c in records.columns]
        lookup = records[columns].reset_index().rename(columns={"index": "record_idx"})
        return frame.merge(lookup, on="record_idx", how="left", validate="many_to_one")

    def score_records(self, records: pd.DataFrame) -> pd.DataFrame:
        records = records.reset_index(drop=True).copy()
        if records.empty:
            raise ValueError("records is empty")

        dataset = self.calibration.build_window_dataset(records, self.config)
        loader = self._loader(dataset)

        collected: dict[str, list[np.ndarray]] = {
            "ae_score": [],
            "y_true": [],
            "anomaly_fraction": [],
            "record_idx": [],
            "start_idx": [],
            "start_time": [],
            "end_time": [],
            "latent": [],
        }
        classifier_chunks: list[np.ndarray] = []

        with torch.inference_mode():
            for batch in loader:
                x = batch["x"].to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                    reconstructed, latent = self._forward_with_latent(self.model, x)
                    ae_score, _, classifier_features = self.hybrid.extract_residual_features(x, reconstructed)

                collected["ae_score"].append(ae_score.float().cpu().numpy())
                collected["latent"].append(latent.float().cpu().numpy())
                classifier_chunks.append(classifier_features.float().cpu().numpy())
                for source_key, target_key in (
                    ("y", "y_true"),
                    ("anomaly_fraction", "anomaly_fraction"),
                    ("record_idx", "record_idx"),
                    ("start_idx", "start_idx"),
                    ("start_time", "start_time"),
                    ("end_time", "end_time"),
                ):
                    collected[target_key].append(batch[source_key].cpu().numpy())

        scalar_keys = [k for k in collected if k != "latent"]
        frame = pd.DataFrame({k: np.concatenate(collected[k]) for k in scalar_keys})
        latent_matrix = np.concatenate(collected["latent"], axis=0).astype(np.float32)
        classifier_matrix = np.concatenate(classifier_chunks, axis=0).astype(np.float32)

        frame = self._merge_record_metadata(frame, records)
        frame = self.calibration.attach_phases(
            frame,
            records,
            int(self.config.get("window", {}).get("sequence_length", 50)),
        )

        residual_probability = self.residual_classifier.predict_proba(classifier_matrix)[:, 1]
        ae_percentile = self.hybrid.ae_percentile_score(
            frame["ae_score"].to_numpy(dtype=np.float64),
            self.hybrid_policy["normal_ae_quantiles"],
        )
        fused = self.hybrid.fuse_scores(
            ae_percentile,
            residual_probability,
            self.hybrid_policy["fusion"],
        )
        frame["ae_percentile_score"] = ae_percentile
        frame["residual_probability"] = residual_probability
        frame["score"] = fused

        # IMPORTANT: final prediction is not independent per window. The company
        # implementation applies contextual thresholds, hysteresis and consecutive
        # filtering over the whole record/run.
        frame = self.postprocess.apply_window_postprocess(frame, self.config, self.threshold_payload)

        for i in range(latent_matrix.shape[1]):
            frame[f"latent_{i:02d}"] = latent_matrix[:, i]
        return frame
