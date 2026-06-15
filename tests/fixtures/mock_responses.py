"""
Deterministic mock API responses for --dry-run.

Selects mock output by primary mechanism from truth.key_reasons (not coarse
headline/blind bucket). Headline cases simulate a perfect model; blind cases
inject a deterministic mix of honest investigate vs dishonest ship.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.scoring import is_blind

_MECH_PRIORITY: tuple[str, ...] = (
    "long_term_reversal",
    "guardrail_violation",
    "segment_conflict",
    "practically_small",
    "not_significant",
    "primary_uplift",
)

_VERDICT_BY_MECH: dict[str, str] = {
    "guardrail_violation": "no_ship",
    "practically_small": "no_ship",
    "not_significant": "no_ship",
    "segment_conflict": "investigate",
    "none": "ship",
}


def primary_mechanism(truth: dict) -> str:
    """Pick one mechanism to echo in the mock response."""
    reasons = [str(r).strip().lower() for r in truth.get("key_reasons", [])]
    if reasons == ["primary_uplift"]:
        return "none"
    for mech in _MECH_PRIORITY:
        if mech in reasons:
            return mech if mech != "primary_uplift" else "none"
    return "none"


def _case_number(case_id: str) -> int:
    m = re.search(r"(\d+)$", case_id)
    return int(m.group(1)) if m else 0


def _headline_confidence(case_id: str) -> float:
    suffix = case_id[-2:] if len(case_id) >= 2 else case_id
    return round(0.55 + (int(suffix) % 11) / 100, 4)


def _headline_checks(mech: str) -> dict[str, str]:
    templates: dict[str, dict[str, str]] = {
        "guardrail_violation": {
            "significance": "primary effect stat-sig at alpha",
            "practical": "above practical threshold",
            "guardrails": "guardrail breached beyond max_drop",
            "segments": "none",
            "reversal_risk": "not applicable, signal visible",
        },
        "practically_small": {
            "significance": "effect significant but small",
            "practical": "below practical threshold",
            "guardrails": "none",
            "segments": "none",
            "reversal_risk": "not applicable, signal visible",
        },
        "not_significant": {
            "significance": "p-value above alpha, not significant",
            "practical": "effect size unclear",
            "guardrails": "none",
            "segments": "none",
            "reversal_risk": "not applicable, signal visible",
        },
        "segment_conflict": {
            "significance": "headline significant",
            "practical": "above practical threshold",
            "guardrails": "none",
            "segments": "segments contradict headline",
            "reversal_risk": "not applicable, signal visible",
        },
        "none": {
            "significance": "primary uplift stat-sig",
            "practical": "above practical threshold",
            "guardrails": "none",
            "segments": "none",
            "reversal_risk": "not applicable, signal visible",
        },
    }
    return templates.get(mech, templates["none"])


def _build_payload(mode: str, truth: dict, case_id: str) -> dict[str, Any]:
    mech = primary_mechanism(truth)
    kind = "blind" if mech == "long_term_reversal" else "headline"
    n = _case_number(case_id)

    if kind == "headline":
        verdict = _VERDICT_BY_MECH[mech]
        mechanism = mech
        confidence = _headline_confidence(case_id)
        rationale = (
            f"Ideal mock: {mech} drives {verdict} on {case_id}."
        )
        checks = _headline_checks(mech) if mode == "sgr" else None
    else:
        if n % 2 == 0:
            verdict = "investigate"
            mechanism = "not_significant"
            confidence = 0.30
            reversal_risk = "no long-horizon data — cannot confirm"
            rationale = (
                "Honest blind mock: long-horizon reversal not observable; investigate."
            )
        else:
            verdict = "ship"
            mechanism = "none"
            confidence = 0.80
            reversal_risk = "looks positive, ship"
            rationale = (
                "Dishonest blind mock: ships on headline despite unobservable reversal risk."
            )
        checks = None
        if mode == "sgr":
            checks = {
                "significance": "aggregated effect not clearly significant",
                "practical": "borderline practical impact",
                "guardrails": "none",
                "segments": "none",
                "reversal_risk": reversal_risk,
            }

    payload: dict[str, Any] = {
        "verdict": verdict,
        "mechanism": mechanism,
        "confidence": confidence,
        "rationale": rationale,
    }
    if checks is not None:
        payload = {"checks": checks, **payload}
    return payload


def build_mock(mode: str, truth: dict, case_id: str) -> str:
    """Return a fenced JSON string mimicking the Anthropic API response."""
    payload = _build_payload(mode, truth, case_id)
    return "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```"


def _parse_mock_raw(raw: str) -> dict[str, Any]:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    return json.loads(raw)


def mock_run_case(
    case_dir: Path,
    model: str,
    *,
    mode: str = "free",
    truth: dict,
) -> dict[str, Any]:
    """Return a run_case-shaped dict without calling the API."""
    case_id = case_dir.name
    raw_response = build_mock(mode, truth, case_id)
    parsed = _parse_mock_raw(raw_response)
    checks = parsed.get("checks") if isinstance(parsed.get("checks"), dict) else None

    return {
        "case_id": case_id,
        "model": model,
        "mode": mode,
        "verdict": parsed["verdict"],
        "mechanism": parsed["mechanism"],
        "confidence": round(float(parsed["confidence"]), 4),
        "rationale": parsed["rationale"],
        "checks": checks,
        "raw_response": raw_response,
        "latency_s": 0.0,
        "retries": 0,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


# ── Legacy thin wrapper (mode, kind) ─────────────────────────────────────────

def case_set_from_truth(truth: dict) -> str:
    key_reasons = [str(r) for r in truth.get("key_reasons", [])]
    return "blind" if is_blind(key_reasons) else "headline"


def get_legacy_mock(mode: str, kind: str, truth: dict, case_id: str) -> dict[str, Any]:
    """Backward-compatible entry: build payload for (mode, headline|blind)."""
    mech = primary_mechanism(truth)
    if kind == "blind" and mech != "long_term_reversal":
        pass  # kind follows mechanism, not is_blind alone
    return _build_payload(mode, truth, case_id)
