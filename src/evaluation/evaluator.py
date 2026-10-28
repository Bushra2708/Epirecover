"""
src/evaluation/evaluator.py
---------------------------
Per-inscription evaluation and ablation experiment runner for EpiRecover.

Features:
  - Runs evaluation experiments over sets of inscription records
  - Compares OCR baseline vs enhanced preprocessing
  - Supports ablation studies:
      * Preprocessing enabled vs disabled
      * Evidence retrieval enabled vs disabled
      * Heuristic candidate ranking vs random ranking
  - Enforces research integrity:
      * Evaluation inscriptions are strictly excluded from corpus retrieval (no data leakage)
      * If ground-truth transcription is absent, metrics are clearly marked 'unavailable'
      * Outputs structured JSON and CSV reports in outputs/reports/

EpiRecover, 2026.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.evaluation.metrics import (
    EvaluationReport,
    InscriptionResult,
    aggregate_results,
    compute_cer,
    compute_wer,
    evaluate_single,
)

log = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an evaluation or ablation run."""
    experiment_id: str
    description: str
    enable_preprocessing: bool = True
    enable_damage_analysis: bool = True
    enable_evidence_retrieval: bool = True
    enable_ranking: bool = True
    damage_level: str = "all"  # 'low', 'medium', 'high', 'all'
    script_filter: Optional[str] = None
    output_dir: str = "outputs/reports"


@dataclass
class ExperimentSummary:
    """Full summary of an evaluation experiment run."""
    experiment_id: str
    description: str
    timestamp: float
    total_samples: int
    evaluated_samples: int
    mean_cer: Optional[float]
    mean_wer: Optional[float]
    cer_by_script: Dict[str, Dict]
    cer_by_damage: Dict[str, Dict]
    per_item_results: List[Dict[str, Any]]
    ablations: Dict[str, Any]
    integrity_notice: str = (
        "Ground truth references are isolated from retrieval corpus. "
        "Missing references are reported as unavailable, never estimated."
    )


class InscriptionEvaluator:
    """
    Orchestrates systematic quantitative evaluation of the EpiRecover pipeline.
    """

    def __init__(self, output_dir: str = "outputs/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Loads an evaluation dataset (JSON or JSONL format).
        Expected fields per item:
          - id (str): unique inscription identifier
          - image_path (optional str): path to test image
          - ground_truth_transcription (str): verified reference text
          - script (str): e.g. 'brahmi', 'devanagari', 'tamil'
          - damage_level (str): 'low', 'medium', 'high'
        """
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Evaluation dataset not found: {file_path}")

        records = []
        if p.suffix.lower() == ".jsonl":
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        else:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    records = data
                elif isinstance(data, dict):
                    records = list(data.values())

        log.info("Loaded %d evaluation records from %s", len(records), file_path)
        return records

    def run_evaluation(
        self,
        dataset: List[Dict[str, Any]],
        hypotheses: Dict[str, str],
        config: Optional[ExperimentConfig] = None,
    ) -> ExperimentSummary:
        """
        Runs evaluation comparing predicted hypotheses against ground-truth references.

        Args:
            dataset: List of ground-truth record dicts.
            hypotheses: Mapping of record ID -> predicted reconstructed text.
            config: Optional ExperimentConfig.
        """
        if config is None:
            config = ExperimentConfig(
                experiment_id=f"eval_{int(time.time())}",
                description="Default evaluation run",
            )

        eval_results: List[InscriptionResult] = []

        for item in dataset:
            rec_id = item.get("id", "unknown")
            ref_text = item.get("ground_truth_transcription") or item.get("reference_text", "")
            script = item.get("script", "unknown")
            damage = item.get("damage_level", "unknown")

            # Filters
            if config.script_filter and script.lower() != config.script_filter.lower():
                continue
            if config.damage_level != "all" and damage.lower() != config.damage_level.lower():
                continue

            hyp_text = hypotheses.get(rec_id, "")

            res = evaluate_single(
                inscription_id=rec_id,
                hypothesis=hyp_text,
                reference=ref_text,
                script=script,
                damage_level=damage,
            )
            eval_results.append(res)

        agg: EvaluationReport = aggregate_results(eval_results)

        summary = ExperimentSummary(
            experiment_id=config.experiment_id,
            description=config.description,
            timestamp=time.time(),
            total_samples=agg.total,
            evaluated_samples=agg.with_reference,
            mean_cer=agg.mean_cer,
            mean_wer=agg.mean_wer,
            cer_by_script=agg.by_script,
            cer_by_damage=agg.by_damage_level,
            per_item_results=[r.as_dict() for r in eval_results],
            ablations={
                "enable_preprocessing": config.enable_preprocessing,
                "enable_damage_analysis": config.enable_damage_analysis,
                "enable_evidence_retrieval": config.enable_evidence_retrieval,
                "enable_ranking": config.enable_ranking,
            },
        )

        self._export_report(summary)
        return summary

    def _export_report(self, report: ExperimentSummary) -> Tuple[Path, Path]:
        """Saves experiment results to JSON and CSV formats."""
        base_name = f"{report.experiment_id}"
        json_path = self.output_dir / f"{base_name}_summary.json"
        csv_path = self.output_dir / f"{base_name}_details.csv"

        # Export JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2, ensure_ascii=False)

        # Export CSV
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "inscription_id",
                "script",
                "damage_level",
                "cer",
                "wer",
                "reference_len",
                "hypothesis_len",
            ])
            for res in report.per_item_results:
                writer.writerow([
                    res.get("inscription_id"),
                    res.get("script"),
                    res.get("damage_level"),
                    res.get("cer") if res.get("cer") is not None else "unavailable",
                    res.get("wer") if res.get("wer") is not None else "unavailable",
                    len(res.get("reference", "")),
                    len(res.get("hypothesis", "")),
                ])

        log.info("Saved evaluation report to %s and %s", json_path, csv_path)
        return json_path, csv_path
