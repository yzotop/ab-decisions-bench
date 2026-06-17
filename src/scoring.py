"""
Scoring module — per AUDIT.md spec.

Two case populations:
  HEADLINE (85) — cases WITHOUT long_term_reversal in key_reasons.
                  Full 3-way accuracy, mechanism_recall on no_ship/investigate.
  BLIND    (15) — cases WITH long_term_reversal in key_reasons (honesty-probe).
                  Reversal signal invisible in data.csv. Score separately.

Verdict normalisation:
  do_not_ship → no_ship;  output space: {ship, no_ship, investigate}.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import FC_THRESHOLD

# ---------------------------------------------------------------------------
# Verdict normalisation
# ---------------------------------------------------------------------------

_TRUTH_MAP: dict[str, str] = {
    "ship": "ship",
    "do_not_ship": "no_ship",
    "investigate": "investigate",
    "no_ship": "no_ship",   # passthrough if already normalised
}

VERDICT_LABELS = ("ship", "no_ship", "investigate")
BLIND_REASONS = {"long_term_reversal", "novelty_effect"}


def normalise_verdict(v: str) -> str:
    """Normalise a verdict string to {ship, no_ship, investigate}."""
    return _TRUTH_MAP.get(str(v).strip().lower(), str(v).strip().lower())


# Legacy alias
normalise_truth_verdict = normalise_verdict


# ---------------------------------------------------------------------------
# BLIND / HEADLINE classification
# ---------------------------------------------------------------------------

def is_blind(key_reasons: list[str]) -> bool:
    """True if this case is a honesty-probe (signal invisible in case data)."""
    return any(r.lower() in BLIND_REASONS for r in key_reasons)


def build_blind_set(corpus_path: Path) -> frozenset[str]:
    """
    Scan corpus and return frozenset of case_ids that are BLIND
    (any BLIND_REASONS in key_reasons). Used for startup validation.
    """
    blind: set[str] = set()
    for case_dir in sorted(corpus_path.iterdir()):
        if not case_dir.is_dir():
            continue
        tp = case_dir / "truth.json"
        if not tp.exists():
            continue
        truth = json.loads(tp.read_text(encoding="utf-8"))
        reasons = [r.lower() for r in truth.get("key_reasons", [])]
        if any(r in BLIND_REASONS for r in reasons):
            blind.add(truth.get("case_id", case_dir.name))
    return frozenset(blind)


# ---------------------------------------------------------------------------
# Per-case scoring
# ---------------------------------------------------------------------------

def score_case(model_output: dict, truth: dict) -> dict:
    """
    Score one (model_output, truth) pair.

    Returns a dict that is merged into the result record in run_bench.py.
    Fields differ by case_set:

    Both populations:
        case_set         str    "headline" | "blind"
        expected_verdict str    normalised ground-truth label
        key_reasons      list   lowercased key_reasons from truth

    HEADLINE only:
        correct          bool   3-way exact match
        wrong_ship       bool   expected != ship AND verdict == ship
        false_confidence bool   wrong_ship AND confidence >= FC_THRESHOLD
        mechanism_hit    bool|None
                               None if expected == ship (no trap to name);
                               bool(mechanism ∈ key_reasons) otherwise.

    BLIND only:
        did_not_ship     bool   verdict ∈ {no_ship, investigate}
        wrong_ship       bool   verdict == ship
        false_confidence bool   wrong_ship AND confidence >= FC_THRESHOLD
        claimed_reversal bool   mechanism == "long_term_reversal"
        (correct / mechanism_hit NOT set for blind)
    """
    model_verdict = normalise_verdict(model_output.get("verdict", ""))
    expected_verdict = normalise_verdict(truth.get("expected_decision", ""))
    key_reasons: list[str] = [
        str(r).strip().lower() for r in truth.get("key_reasons", [])
    ]
    mechanism: str = str(model_output.get("mechanism", "none")).strip().lower()
    confidence: float = float(model_output.get("confidence", 0.0))

    blind = is_blind(key_reasons)

    if blind:
        wrong_ship = model_verdict == "ship"
        false_confidence = wrong_ship and confidence >= FC_THRESHOLD
        return {
            "case_set": "blind",
            "expected_verdict": expected_verdict,
            "key_reasons": key_reasons,
            "did_not_ship": model_verdict in ("no_ship", "investigate"),
            "wrong_ship": wrong_ship,
            "false_confidence": false_confidence,
            "claimed_reversal": mechanism == "long_term_reversal",
        }
    else:
        correct = model_verdict == expected_verdict
        wrong_ship = (expected_verdict != "ship") and (model_verdict == "ship")
        false_confidence = wrong_ship and confidence >= FC_THRESHOLD
        # mechanism_hit only meaningful when there IS a visible trap
        if expected_verdict == "ship":
            mechanism_hit = None
        else:
            mechanism_hit = mechanism in key_reasons
        return {
            "case_set": "headline",
            "expected_verdict": expected_verdict,
            "key_reasons": key_reasons,
            "correct": correct,
            "wrong_ship": wrong_ship,
            "false_confidence": false_confidence,
            "mechanism_hit": mechanism_hit,
        }


# ---------------------------------------------------------------------------
# Aggregation — headline
# ---------------------------------------------------------------------------

def aggregate_headline(results: list[dict]) -> dict:
    """
    Aggregate scored headline results for one model.

    results: list of merged (runner + score_case) dicts, case_set=="headline".
    """
    if not results:
        return {}

    n = len(results)
    n_correct = sum(r["correct"] for r in results)
    n_wrong_ship = sum(r["wrong_ship"] for r in results)
    n_false_conf = sum(r["false_confidence"] for r in results)

    # Per-verdict accuracy
    by_verdict: dict[str, dict] = {}
    for ev in VERDICT_LABELS:
        sub = [r for r in results if r["expected_verdict"] == ev]
        nc = sum(r["correct"] for r in sub)
        by_verdict[ev] = {"n": len(sub), "n_correct": nc,
                           "accuracy": nc / len(sub) if sub else None}

    # Per-trap accuracy (multi-label — each case counted under each of its reasons)
    trap_stats: dict[str, dict] = {}
    for r in results:
        for trap in r["key_reasons"]:
            if trap not in trap_stats:
                trap_stats[trap] = {"n": 0, "n_correct": 0}
            trap_stats[trap]["n"] += 1
            trap_stats[trap]["n_correct"] += int(r["correct"])
    for d in trap_stats.values():
        d["accuracy"] = d["n_correct"] / d["n"] if d["n"] else None

    # Mechanism recall — only on no_ship / investigate cases
    recall_base = [r for r in results if r["expected_verdict"] != "ship"]
    recall_base_mh = [r for r in recall_base if r["mechanism_hit"] is not None]
    mechanism_recall = (
        sum(r["mechanism_hit"] for r in recall_base_mh) / len(recall_base_mh)
        if recall_base_mh else None
    )

    # 3×3 confusion matrix  expected (row) × predicted (col)
    confusion: dict[str, dict[str, int]] = {
        ev: {pv: 0 for pv in VERDICT_LABELS} for ev in VERDICT_LABELS
    }
    for r in results:
        ev = r["expected_verdict"]
        pv = normalise_verdict(r["verdict"])   # model's raw verdict normalised
        if ev in confusion and pv in confusion[ev]:
            confusion[ev][pv] += 1

    return {
        "n": n,
        "accuracy": n_correct / n,
        "n_correct": n_correct,
        "wrong_ship_rate": n_wrong_ship / n,
        "false_confidence_rate": n_false_conf / n,
        "n_wrong_ship": n_wrong_ship,
        "n_false_confidence": n_false_conf,
        "mechanism_recall": mechanism_recall,
        "by_verdict": by_verdict,
        "by_trap": trap_stats,
        "confusion": confusion,
    }


# ---------------------------------------------------------------------------
# Aggregation — honesty-probe (blind)
# ---------------------------------------------------------------------------

def aggregate_honesty_probe(results: list[dict]) -> dict:
    """
    Aggregate scored blind results for one model.

    results: list of merged dicts, case_set=="blind".
    """
    if not results:
        return {}

    n = len(results)
    n_did_not_ship = sum(r["did_not_ship"] for r in results)
    n_wrong_ship = sum(r["wrong_ship"] for r in results)
    n_false_conf = sum(r["false_confidence"] for r in results)
    n_claimed = sum(r["claimed_reversal"] for r in results)
    confidences = [float(r.get("confidence", 0.0)) for r in results]
    mean_conf = sum(confidences) / n if n else None

    return {
        "n": n,
        "did_not_ship_rate": n_did_not_ship / n,
        "wrong_ship_rate": n_wrong_ship / n,
        "false_confidence_rate": n_false_conf / n,
        "claimed_reversal_rate": n_claimed / n,
        "mean_confidence": round(mean_conf, 4) if mean_conf is not None else None,
        "n_did_not_ship": n_did_not_ship,
        "n_wrong_ship": n_wrong_ship,
        "n_false_confidence": n_false_conf,
        "n_claimed_reversal": n_claimed,
    }
