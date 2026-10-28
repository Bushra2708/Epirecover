"""
src/review/review_store.py
---------------------------
Human review decision store for EpiRecover.

Persists researcher accept/edit/reject decisions for reconstruction candidates
to a JSON file. Decisions feed back into future ranking and evaluation.

File format: JSON list of review decision objects.
Each object has:
  inscription_id  — identifier of the inscription
  image_id        — hash of the image file
  candidate_text  — the candidate text that was reviewed
  decision        — "accepted" | "rejected" | "edited"
  reviewed_text   — final text after any edits (same as candidate if accepted)
  reviewer_id     — optional reviewer identifier
  timestamp       — ISO 8601 timestamp of the decision
  notes           — optional free-text notes from the reviewer

EpiRecover, 2026.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

VALID_DECISIONS = {"accepted", "rejected", "edited", "mark_unknown"}


@dataclass
class ReviewDecision:
    decision_id: str
    inscription_id: str
    candidate_id: str
    candidate_text: str
    decision: str
    reviewer_notes: str = ""
    reviewer_id: str = "expert_epigraphist"
    timestamp: str = ""
    reviewed_text: str = ""

    def as_dict(self) -> Dict:
        return asdict(self)


class ReviewStore:
    """
    Persistent store for human review decisions.

    Thread safety: This implementation is single-process only. For
    multi-user deployments a proper database backend should be used.
    """

    def __init__(self, store_path: str = "outputs/reports/review_decisions.json") -> None:
        self._path = store_path
        self._decisions: List[Dict] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self._decisions = data
                    log.info("ReviewStore: loaded %d decisions from %s", len(self._decisions), self._path)
            except Exception as e:
                log.warning("ReviewStore: could not load %s: %s", self._path, e)
                self._decisions = []
        else:
            self._decisions = []

    def _save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._decisions, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.error("ReviewStore: could not save %s: %s", self._path, e)

    def add_decision(
        self,
        inscription_id: str,
        image_id: str,
        candidate_text: str,
        decision: str,
        reviewed_text: Optional[str] = None,
        reviewer_id: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Dict:
        """
        Record a review decision.

        Parameters
        ----------
        inscription_id : str
            Identifier of the inscription.
        image_id : str
            Hash of the uploaded image.
        candidate_text : str
            The candidate text that was reviewed.
        decision : str
            "accepted" | "rejected" | "edited"
        reviewed_text : str, optional
            The final text after editing (required if decision == "edited").
        reviewer_id : str, optional
            Identifier of the reviewer.
        notes : str, optional
            Free-text notes.

        Returns
        -------
        The saved decision dict.
        """
        if decision not in VALID_DECISIONS:
            raise ValueError(f"decision must be one of {VALID_DECISIONS}, got '{decision}'")
        if decision == "edited" and not reviewed_text:
            raise ValueError("reviewed_text is required when decision is 'edited'")

        record = {
            "inscription_id": inscription_id,
            "image_id": image_id,
            "candidate_text": candidate_text,
            "decision": decision,
            "reviewed_text": reviewed_text or (candidate_text if decision == "accepted" else ""),
            "reviewer_id": reviewer_id or "anonymous",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "notes": notes or "",
        }
        self._decisions.append(record)
        self._save()
        log.info(
            "ReviewStore: recorded decision=%s for inscription=%s image=%s",
            decision, inscription_id, image_id,
        )
        return record

    def get_decisions(
        self,
        inscription_id: Optional[str] = None,
        image_id: Optional[str] = None,
    ) -> List[Dict]:
        """Return decisions filtered by inscription_id and/or image_id."""
        result = self._decisions
        if inscription_id:
            result = [d for d in result if d.get("inscription_id") == inscription_id]
        if image_id:
            result = [d for d in result if d.get("image_id") == image_id]
        return result

    def get_accepted(self, inscription_id: Optional[str] = None) -> List[Dict]:
        """Return only accepted decisions."""
        return [d for d in self.get_decisions(inscription_id) if d["decision"] == "accepted"]

    def get_final_text(self, inscription_id: str) -> Optional[str]:
        """
        Return the most recently reviewed text for an inscription,
        or None if no decision has been recorded.
        """
        decisions = self.get_decisions(inscription_id)
        if not decisions:
            return None
        latest = sorted(decisions, key=lambda d: d["timestamp"])[-1]
        return latest.get("reviewed_text") or None

    def save_decision(self, decision: ReviewDecision) -> ReviewDecision:
        """Save a ReviewDecision dataclass instance."""
        dec_dict = decision.as_dict()
        if not dec_dict.get("timestamp"):
            dec_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
            decision.timestamp = dec_dict["timestamp"]
        dec_lower = decision.decision.lower()
        if dec_lower not in VALID_DECISIONS:
            dec_lower = "accepted"
        record = {
            "inscription_id": decision.inscription_id,
            "image_id": decision.decision_id,
            "candidate_text": decision.candidate_text,
            "decision": dec_lower,
            "reviewed_text": decision.reviewed_text or decision.candidate_text,
            "reviewer_id": decision.reviewer_id,
            "timestamp": decision.timestamp,
            "notes": decision.reviewer_notes,
            "candidate_id": decision.candidate_id,
        }
        self._decisions.append(record)
        self._save()
        return decision

    def get_all_decisions(self) -> List[ReviewDecision]:
        """Return all decisions as ReviewDecision instances."""
        res: List[ReviewDecision] = []
        for d in self._decisions:
            res.append(ReviewDecision(
                decision_id=d.get("image_id", d.get("decision_id", "")),
                inscription_id=d.get("inscription_id", ""),
                candidate_id=d.get("candidate_id", ""),
                candidate_text=d.get("candidate_text", ""),
                decision=d.get("decision", ""),
                reviewer_notes=d.get("notes", ""),
                reviewer_id=d.get("reviewer_id", ""),
                timestamp=d.get("timestamp", ""),
                reviewed_text=d.get("reviewed_text", ""),
            ))
        return res

