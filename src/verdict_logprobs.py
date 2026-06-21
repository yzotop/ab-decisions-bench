"""
Extract normalized verdict probability distribution from chat logprobs.

Uses the first token after `"verdict": "` in the model JSON output.
Aggregates exp(logprob) across ALL top-k tokens mapping to each verdict.
"""

from __future__ import annotations

import math
from typing import Any

VERDICTS = ("ship", "no_ship", "investigate")


def _norm_tok(token: str) -> str:
    t = token.replace("Ġ", " ").replace("▁", " ").strip()
    return t.strip('"').strip()


def _as_dict(item: Any) -> dict:
    if isinstance(item, dict):
        return item
    return {
        "token": getattr(item, "token", ""),
        "logprob": getattr(item, "logprob", None),
    }


def _verdict_bucket(token: str) -> str | None:
    """Map a top-k token string to ship / no_ship / investigate, or None."""
    t = _norm_tok(token).lower()
    if t == "ship":
        return "ship"
    if t == "no":
        return "no_ship"
    if t in ("invest", "investig", "investigate"):
        return "investigate"
    return None


def find_verdict_value_start(content: list[dict]) -> int | None:
    """Index of the first token of the verdict string value in logprobs.content."""
    for i, raw in enumerate(content):
        tok = _norm_tok(raw.get("token", ""))
        if tok != "dict" or i + 1 >= len(content):
            continue
        nxt_raw = content[i + 1].get("token", "")
        nxt = _norm_tok(nxt_raw)
        if nxt != '":' and '":' not in nxt_raw:
            continue
        j = i + 2
        while j < len(content):
            tj = _norm_tok(content[j].get("token", ""))
            if tj in ('"', ""):
                j += 1
                continue
            return j
    return None


def _min_top_logprob(top_logprobs: list) -> float | None:
    """Lowest logprob in top-k — upper bound for any token outside top-k."""
    best: float | None = None
    for raw in top_logprobs or []:
        item = _as_dict(raw)
        lp = item.get("logprob")
        if lp is None:
            continue
        best = lp if best is None else min(best, lp)
    return best


def _aggregate_verdict_masses(
    top_logprobs: list,
    *,
    sampled_token: str,
    sampled_logprob: float | None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """
    Sum exp(logprob) over every top-k token that maps to each verdict bucket.
    Returns per-verdict mass and the serialized top-k for raw export.
    """
    pairs: list[tuple[str, float]] = []
    for raw in top_logprobs or []:
        item = _as_dict(raw)
        lp = item.get("logprob")
        tok = item.get("token", "")
        if lp is None:
            continue
        pairs.append((tok, float(lp)))

    sampled_norm = _norm_tok(sampled_token)
    if sampled_logprob is not None and not any(
        _norm_tok(t) == sampled_norm for t, _ in pairs
    ):
        pairs.append((sampled_token, float(sampled_logprob)))

    masses: dict[str, float] = {k: 0.0 for k in VERDICTS}
    for tok, lp in pairs:
        bucket = _verdict_bucket(tok)
        if bucket:
            masses[bucket] += math.exp(lp)

    verdict_top_logprobs = [
        {"token": tok, "logprob": round(lp, 6)} for tok, lp in pairs
    ]
    return masses, verdict_top_logprobs


def extract_verdict_logprobs(logprobs_content: list[dict]) -> dict[str, Any]:
    """
    Return p_verdict_dist (normalized), p_chosen (filled by caller), and debug fields.
    """
    empty = {
        "p_verdict_dist": None,
        "p_chosen": None,
        "verdict_first_token": None,
        "verdict_logprob_idx": None,
        "verdict_top_logprobs": None,
    }
    if not logprobs_content:
        return empty

    idx = find_verdict_value_start(logprobs_content)
    if idx is None:
        return empty

    first = logprobs_content[idx]
    first_tok = _norm_tok(first.get("token", ""))
    top = first.get("top_logprobs") or []

    masses, verdict_top_logprobs = _aggregate_verdict_masses(
        top,
        sampled_token=first.get("token", ""),
        sampled_logprob=first.get("logprob"),
    )
    censored: dict[str, bool] = {}

    # Verdict buckets with zero mass in top-k: censored floor at min top-k logprob.
    floor_lp = _min_top_logprob(top)
    for key in VERDICTS:
        if masses[key] > 0:
            continue
        if floor_lp is not None:
            masses[key] = math.exp(floor_lp)
            censored[key] = True

    total = sum(masses.values())
    raw_logps = {
        k: round(math.log(masses[k]), 6) if masses[k] > 0 else None
        for k in VERDICTS
    }

    if total <= 0:
        return {
            **empty,
            "verdict_first_token": first_tok,
            "verdict_logprob_idx": idx,
            "raw_logprobs": raw_logps,
            "verdict_top_logprobs": verdict_top_logprobs,
            "p_verdict_dist_censored": censored or None,
            "topk_floor_logprob": floor_lp,
        }

    p_dist = {k: round(masses[k] / total, 6) for k in VERDICTS}

    return {
        "p_verdict_dist": p_dist,
        "p_chosen": None,
        "verdict_first_token": first_tok,
        "verdict_logprob_idx": idx,
        "raw_logprobs": raw_logps,
        "verdict_top_logprobs": verdict_top_logprobs,
        "p_verdict_dist_censored": censored or None,
        "topk_floor_logprob": floor_lp,
    }


def attach_p_chosen(parsed: dict[str, Any], logprob_info: dict[str, Any]) -> dict[str, Any]:
    """Set p_chosen from normalized dist and parsed verdict."""
    dist = logprob_info.get("p_verdict_dist")
    verdict = parsed.get("verdict")
    if dist and verdict in dist:
        logprob_info = {**logprob_info, "p_chosen": dist[verdict]}
    return logprob_info
