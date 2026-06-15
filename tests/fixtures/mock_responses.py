"""
Deterministic mock API responses for --dry-run.

Keyed by (mode, case_set) where case_set is "headline" | "blind"
(derived from truth.json key_reasons via is_blind).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.scoring import is_blind

MOCK_RESPONSES: dict[tuple[str, str], dict[str, Any]] = {
    ("free", "headline"): {
        "verdict": "no_ship",
        "mechanism": "guardrail_violation",
        "confidence": 0.62,
        "checks": None,
        "rationale": (
            "Revenue uplift is statistically significant, but CTR fell -6.6%, "
            "breaching the 3% guardrail — do not ship."
        ),
    },
    ("sgr", "headline"): {
        "verdict": "no_ship",
        "mechanism": "guardrail_violation",
        "confidence": 0.60,
        "checks": {
            "significance": "effect +1.3%, stat-sig",
            "practical": "above threshold",
            "guardrails": "CTR -6.6% breaches 3% — VIOLATION",
            "segments": "none",
            "reversal_risk": "not applicable, signal visible",
        },
        "rationale": (
            "Checks show stat-sig uplift, but CTR guardrail is breached; "
            "verdict follows from guardrails check."
        ),
    },
    ("free", "blind"): {
        "verdict": "ship",
        "mechanism": "none",
        "confidence": 0.81,
        "checks": None,
        "rationale": (
            "Headline revenue effect looks positive; shipping despite limited "
            "visibility into longer horizons."
        ),
    },
    ("sgr", "blind"): {
        "verdict": "investigate",
        "mechanism": "not_significant",
        "confidence": 0.30,
        "checks": {
            "significance": "aggregated +0.48%, not stat-sig",
            "practical": "below practical threshold",
            "guardrails": "none",
            "segments": "none",
            "reversal_risk": "no weekly horizon in data — cannot confirm",
        },
        "rationale": (
            "Reversal risk cannot be assessed from available data; "
            "investigate rather than ship."
        ),
    },
}


def case_set_from_truth(truth: dict) -> str:
    key_reasons = [str(r) for r in truth.get("key_reasons", [])]
    return "blind" if is_blind(key_reasons) else "headline"


def mock_run_case(
    case_dir: Path,
    model: str,
    *,
    mode: str = "free",
    truth: dict,
) -> dict[str, Any]:
    """Return a run_case-shaped dict without calling the API."""
    case_set = case_set_from_truth(truth)
    payload = MOCK_RESPONSES[(mode, case_set)]
    body = {
        "verdict": payload["verdict"],
        "mechanism": payload["mechanism"],
        "confidence": payload["confidence"],
        "rationale": payload["rationale"],
    }
    if payload.get("checks") is not None:
        body["checks"] = payload["checks"]

    return {
        "case_id": case_dir.name,
        "model": model,
        "mode": mode,
        "verdict": payload["verdict"],
        "mechanism": payload["mechanism"],
        "confidence": payload["confidence"],
        "rationale": payload["rationale"],
        "checks": payload.get("checks"),
        "raw_response": json.dumps(body, ensure_ascii=False),
        "latency_s": 0.0,
        "retries": 0,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
