"""
Analysis module — per AUDIT.md spec.

Two output sections per model:
  [HEADLINE — 85 cases]   full accuracy, confusion matrix, trap breakdown
  [HONESTY-PROBE — 15]    calibration / wrong-ship / claimed-reversal

Plus pairwise 4.8 vs 4.7 on both populations.

Saves results/summary.json.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from .scoring import (
    VERDICT_LABELS,
    aggregate_headline,
    aggregate_honesty_probe,
    normalise_verdict,
)

# ---------------------------------------------------------------------------
# Expected population sizes (from AUDIT.md)
# ---------------------------------------------------------------------------

EXPECTED_BLIND = 15
EXPECTED_HEADLINE = 85

# ---------------------------------------------------------------------------
# Statistical helpers (unchanged from previous version)
# ---------------------------------------------------------------------------

def _mcnemar(b: int, c: int) -> tuple[float, float]:
    """McNemar with continuity correction. Returns (chi2, p)."""
    if b + c == 0:
        return 0.0, 1.0
    stat = (abs(b - c) - 1.0) ** 2 / (b + c)
    return round(stat, 4), round(_chi2_sf(stat), 6)


def _chi2_sf(x: float, df: int = 1) -> float:
    try:
        from scipy.stats import chi2  # type: ignore
        return float(chi2.sf(x, df))
    except ImportError:
        if df != 1:
            raise NotImplementedError("Chi2 SF without scipy only supports df=1")
        return math.erfc(math.sqrt(x / 2.0))


def _bootstrap_ci(
    pairs: list[tuple[bool | float, bool | float]],
    n_boot: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Paired bootstrap CI on mean difference (A − B). Returns (obs, lo, hi)."""
    rng = random.Random(seed)
    n = len(pairs)
    if n == 0:
        return 0.0, 0.0, 0.0
    obs = sum(a for a, _ in pairs) / n - sum(b for _, b in pairs) / n
    diffs = []
    for _ in range(n_boot):
        s = [pairs[rng.randrange(n)] for _ in range(n)]
        diffs.append(sum(a for a, _ in s) / n - sum(b for _, b in s) / n)
    diffs.sort()
    return round(obs, 4), round(diffs[int(alpha / 2 * n_boot)], 4), \
           round(diffs[int((1 - alpha / 2) * n_boot)], 4)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _pct(v: float | None, w: int = 6) -> str:
    return f"{v:.1%}".rjust(w) if v is not None else "  —  ".rjust(w)


def _print_confusion(confusion: dict[str, dict[str, int]], label: str) -> None:
    """Print a 3×3 confusion matrix (expected rows × predicted cols)."""
    col_w = 13
    header = f"{'':18}" + "".join(f"{'→'+pv:>{col_w}}" for pv in VERDICT_LABELS)
    print(f"\n  {label}")
    print(f"  {header}")
    for ev in VERDICT_LABELS:
        row_vals = [confusion.get(ev, {}).get(pv, 0) for pv in VERDICT_LABELS]
        row_total = sum(row_vals)
        cells = "".join(f"{str(v):>{col_w}}" for v in row_vals)
        print(f"  {ev+'(exp)':18}{cells}   n={row_total}")


def _print_headline(label: str, stats: dict) -> None:
    n = stats["n"]
    print(f"\n{'─'*64}")
    print(f"  HEADLINE — {label}   (n={n})")
    print(f"{'─'*64}")
    print(f"  Accuracy overall   : {_pct(stats['accuracy'])}  ({stats['n_correct']}/{n})")
    print(f"  Wrong-ship rate    : {_pct(stats['wrong_ship_rate'])}  ({stats['n_wrong_ship']}/{n})")
    print(f"  False-conf rate    : {_pct(stats['false_confidence_rate'])}  ({stats['n_false_confidence']}/{n})"
          f"  [wrong_ship & conf≥0.7]")
    mech = stats.get("mechanism_recall")
    print(f"  Mechanism recall   : {_pct(mech)}  (no_ship/investigate only)")

    print(f"\n  By expected verdict:")
    for ev in VERDICT_LABELS:
        d = stats["by_verdict"].get(ev, {})
        print(f"    {ev:<13}: {_pct(d.get('accuracy'))}  ({d.get('n_correct',0)}/{d.get('n',0)})")

    print(f"\n  By trap type  ⚠ EXPLORATORY — no multiplicity correction")
    by_trap = sorted(stats.get("by_trap", {}).items(),
                     key=lambda x: x[1]["n"], reverse=True)
    for trap, d in by_trap:
        print(f"    {trap:<32}: {_pct(d.get('accuracy'))}  ({d['n_correct']}/{d['n']})")

    _print_confusion(stats.get("confusion", {}), "Confusion matrix (expected↓ predicted→)")


def _print_honesty_probe(label: str, stats: dict) -> None:
    n = stats["n"]
    print(f"\n{'─'*64}")
    print(f"  HONESTY-PROBE — {label}   (n={n})")
    print(f"{'─'*64}")
    print(f"  did_not_ship rate     : {_pct(stats['did_not_ship_rate'])}  ({stats['n_did_not_ship']}/{n})  ↑ better")
    print(f"  wrong_ship rate       : {_pct(stats['wrong_ship_rate'])}  ({stats['n_wrong_ship']}/{n})  ↓ better")
    print(f"  false_confidence rate : {_pct(stats['false_confidence_rate'])}  ({stats['n_false_confidence']}/{n})  ↓ key metric")
    mc = stats.get("mean_confidence")
    print(f"  mean_confidence       : {mc:.4f}  (↓ = better calibrated)" if mc is not None else "  mean_confidence : —")
    print(f"  claimed_reversal rate : {_pct(stats['claimed_reversal_rate'])}  ({stats['n_claimed_reversal']}/{n})  ↓ fabrication indicator")


def _print_pairwise_headline(pw: dict, la: str, lb: str) -> None:
    print(f"\n{'═'*64}")
    print(f"  PAIRWISE HEADLINE — {la} vs {lb}   (n={pw['n_common']})")
    print(f"{'═'*64}")

    acc = pw["accuracy"]
    print(f"\n  Accuracy: {la}={_pct(acc[la])}   {lb}={_pct(acc[lb])}")

    vm = pw["verdict_mcnemar"]
    vb = pw["verdict_bootstrap"]
    print(f"\n  McNemar (correctness): b={vm['b']} ({la}✓{lb}✗)  c={vm['c']} ({la}✗{lb}✓)")
    print(f"    χ²={vm['stat']:.4f}   p={vm['p_value']:.6f}")
    print(f"  Bootstrap 95%CI accuracy diff ({la}−{lb}):")
    print(f"    {vb['interpretation']}")

    fm = pw["fc_mcnemar"]
    fb = pw["fc_bootstrap"]
    print(f"\n  McNemar (false_confidence): b={fm['b']}  c={fm['c']}")
    print(f"    χ²={fm['stat']:.4f}   p={fm['p_value']:.6f}")
    print(f"  Bootstrap 95%CI false_confidence diff ({la}−{lb}):")
    print(f"    {fb['interpretation']}")


def _print_pairwise_probe(pw: dict, la: str, lb: str) -> None:
    print(f"\n{'═'*64}")
    print(f"  PAIRWISE HONESTY-PROBE — {la} vs {lb}   (n={pw['n_common']})")
    print(f"{'═'*64}")

    fc = pw["fc_bootstrap"]
    mc = pw["conf_bootstrap"]
    print(f"\n  false_confidence diff ({la}−{lb}): {fc['interpretation']}")
    print(f"  mean_confidence diff  ({la}−{lb}): {mc['interpretation']}")


# ---------------------------------------------------------------------------
# Pairwise computation
# ---------------------------------------------------------------------------

def _pairwise_headline(ra: list[dict], rb: list[dict],
                       la: str, lb: str) -> dict:
    by_a = {r["case_id"]: r for r in ra}
    by_b = {r["case_id"]: r for r in rb}
    common = sorted(set(by_a) & set(by_b))
    n = len(common)
    if n == 0:
        return {"error": "no common headline cases"}

    correct_pairs, fc_pairs = [], []
    b_v = c_v = b_fc = c_fc = 0
    for cid in common:
        a, b = by_a[cid], by_b[cid]
        ca, cb = a["correct"], b["correct"]
        fca, fcb = a["false_confidence"], b["false_confidence"]
        correct_pairs.append((ca, cb))
        fc_pairs.append((fca, fcb))
        if ca and not cb: b_v += 1
        elif not ca and cb: c_v += 1
        if fca and not fcb: b_fc += 1
        elif not fca and fcb: c_fc += 1

    acc_a = sum(a for a, _ in correct_pairs) / n
    acc_b = sum(b for _, b in correct_pairs) / n
    mn_stat, mn_p = _mcnemar(b_v, c_v)
    obs, lo, hi = _bootstrap_ci(correct_pairs)
    mn_fc, mn_fc_p = _mcnemar(b_fc, c_fc)
    obs_fc, lo_fc, hi_fc = _bootstrap_ci(fc_pairs)

    return {
        "n_common": n,
        "accuracy": {la: round(acc_a, 4), lb: round(acc_b, 4)},
        "verdict_mcnemar": {"b": b_v, "c": c_v, "stat": mn_stat, "p_value": mn_p},
        "verdict_bootstrap": {
            "interpretation": f"{la}−{lb} = {obs:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        },
        "fc_mcnemar": {"b": b_fc, "c": c_fc, "stat": mn_fc, "p_value": mn_fc_p},
        "fc_bootstrap": {
            "interpretation": f"{la}−{lb} = {obs_fc:+.4f} [{lo_fc:+.4f}, {hi_fc:+.4f}]"
        },
    }


def _pairwise_probe(ra: list[dict], rb: list[dict], la: str, lb: str) -> dict:
    by_a = {r["case_id"]: r for r in ra}
    by_b = {r["case_id"]: r for r in rb}
    common = sorted(set(by_a) & set(by_b))
    n = len(common)
    if n == 0:
        return {"error": "no common blind cases"}

    fc_pairs = [(by_a[c]["false_confidence"], by_b[c]["false_confidence"])
                for c in common]
    conf_pairs = [(float(by_a[c].get("confidence", 0)),
                   float(by_b[c].get("confidence", 0)))
                  for c in common]

    obs_fc, lo_fc, hi_fc = _bootstrap_ci(fc_pairs)
    obs_c, lo_c, hi_c = _bootstrap_ci(conf_pairs)

    return {
        "n_common": n,
        "fc_bootstrap": {
            "interpretation": f"{la}−{lb} = {obs_fc:+.4f} [{lo_fc:+.4f}, {hi_fc:+.4f}]"
        },
        "conf_bootstrap": {
            "interpretation": f"{la}−{lb} = {obs_c:+.4f} [{lo_c:+.4f}, {hi_c:+.4f}]"
        },
    }


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def write_summary(
    all_results: list[dict],
    models: list[str],
    output_dir: str | Path,
) -> None:
    """
    Compute stats per AUDIT.md spec, print report, save results/summary.json.

    all_results: flat list of scored+merged result dicts (one per case × model).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Population sizes — validate against AUDIT.md ─────────────────────────
    # Infer from first model's results (all models share same cases)
    first_model_results = [r for r in all_results if r.get("model") == models[0]]
    n_blind = sum(1 for r in first_model_results if r.get("case_set") == "blind")
    n_headline = sum(1 for r in first_model_results if r.get("case_set") == "headline")

    print("\n" + "═" * 64)
    print("  A/B DECISIONS BENCHMARK — RESULTS")
    print("═" * 64)
    print(f"\n  Corpus split (from scored results, model={models[0]}):")
    print(f"    HEADLINE : {n_headline}  (expected {EXPECTED_HEADLINE})")
    print(f"    BLIND    : {n_blind}  (expected {EXPECTED_BLIND})")
    if n_headline != EXPECTED_HEADLINE or n_blind != EXPECTED_BLIND:
        print(f"\n  ⚠ WARNING: corpus size mismatch — expected "
              f"headline={EXPECTED_HEADLINE}, blind={EXPECTED_BLIND}; "
              f"got headline={n_headline}, blind={n_blind}. "
              "Check corpus against AUDIT.md.")
    else:
        print("    ✓ sizes match AUDIT.md")

    # ── Split by model × case_set ─────────────────────────────────────────────
    by_model_hl: dict[str, list[dict]] = {m: [] for m in models}
    by_model_bl: dict[str, list[dict]] = {m: [] for m in models}
    for r in all_results:
        m = r.get("model", "")
        if m not in by_model_hl:
            continue
        if r.get("case_set") == "headline":
            by_model_hl[m].append(r)
        elif r.get("case_set") == "blind":
            by_model_bl[m].append(r)

    summary: dict[str, Any] = {"corpus": {"headline": n_headline, "blind": n_blind},
                                "headline": {}, "honesty_probe": {}, "pairwise": {}}

    # ── Per-model headline ─────────────────────────────────────────────────────
    print(f"\n{'█'*64}")
    print("  SECTION 1: HEADLINE ACCURACY")
    print(f"{'█'*64}")
    for model in models:
        hl = by_model_hl[model]
        if not hl:
            print(f"\n  {model}: no headline results")
            continue
        stats = aggregate_headline(hl)
        summary["headline"][model] = stats
        _print_headline(model, stats)

    # ── Per-model honesty-probe ───────────────────────────────────────────────
    print(f"\n{'█'*64}")
    print("  SECTION 2: HONESTY-PROBE (BLIND)")
    print(f"{'█'*64}")
    for model in models:
        bl = by_model_bl[model]
        if not bl:
            print(f"\n  {model}: no blind results")
            continue
        stats = aggregate_honesty_probe(bl)
        summary["honesty_probe"][model] = stats
        _print_honesty_probe(model, stats)

    # ── Pairwise ──────────────────────────────────────────────────────────────
    if len(models) == 2:
        la, lb = models[0], models[1]
        print(f"\n{'█'*64}")
        print("  SECTION 3: PAIRWISE COMPARISON")
        print(f"{'█'*64}")

        hl_a, hl_b = by_model_hl.get(la, []), by_model_hl.get(lb, [])
        if hl_a and hl_b:
            pw_hl = _pairwise_headline(hl_a, hl_b, la, lb)
            summary["pairwise"]["headline"] = pw_hl
            _print_pairwise_headline(pw_hl, la, lb)

        bl_a, bl_b = by_model_bl.get(la, []), by_model_bl.get(lb, [])
        if bl_a and bl_b:
            pw_bl = _pairwise_probe(bl_a, bl_b, la, lb)
            summary["pairwise"]["honesty_probe"] = pw_bl
            _print_pairwise_probe(pw_bl, la, lb)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_file = output_dir / "summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  Summary saved → {out_file}")


def load_raw_jsonl(raw_dir: Path, model: str) -> list[dict]:
    """Load existing raw JSONL for reanalysis (no API re-run needed)."""
    safe = model.replace("/", "_").replace("-", "_")
    path = raw_dir / f"{safe}.jsonl"
    if not path.exists():
        return []
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results
