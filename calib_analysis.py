#!/usr/bin/env python3
"""
Calibration analysis for local_openai bench runs.

Primary calibration signal: self-reported JSON confidence (not token logprobs).
Logprob fields in raw JSONL are kept as a sidecar for audit only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from src.scoring import aggregate_honesty_probe, antiship_correct, normalise_verdict

ECE_BINS = 5
DIST_SUM_TOL = 1e-3
VALID_VERDICTS = {"ship", "no_ship", "investigate"}


def _load_zone_map(corpus: Path) -> dict[str, str]:
    labels = json.loads((corpus / "_trap_labels.json").read_text(encoding="utf-8"))
    zones_map = json.loads((corpus.parent / "trainer/trap_zones.json").read_text(encoding="utf-8"))
    return {cid: zones_map[trap] for cid, trap in labels.items()}


def _load_records(raw_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _is_invalid(r: dict[str, Any]) -> bool:
    return bool(r.get("invalid_format")) or r.get("verdict") in ("invalid_format", "parse_error", "api_error")


def _strict_correct(r: dict[str, Any]) -> bool:
    if _is_invalid(r):
        return False
    return normalise_verdict(r.get("verdict", "")) == normalise_verdict(r.get("expected_verdict", ""))


def _antiship_ok(r: dict[str, Any]) -> bool:
    if _is_invalid(r):
        return False
    return antiship_correct(
        normalise_verdict(r.get("verdict", "")),
        normalise_verdict(r.get("expected_verdict", "")),
    )


def _self_confidence(r: dict[str, Any]) -> float | None:
    """Model self-reported confidence from JSON (primary calibration signal)."""
    if _is_invalid(r):
        return None
    val = r.get("confidence")
    if val is None:
        return None
    return float(val)


def _mean_self_confidence(records: list[dict[str, Any]], predicate) -> float | None:
    vals = [
        c for r in records
        if (c := _self_confidence(r)) is not None and predicate(r)
    ]
    return round(sum(vals) / len(vals), 4) if vals else None


def _ece_self_confidence(
    records: list[dict[str, Any]], n_bins: int = ECE_BINS,
) -> tuple[float, list[dict[str, Any]]]:
    """ECE on self-reported confidence vs strict correctness."""
    valid = [r for r in records if _self_confidence(r) is not None and not _is_invalid(r)]
    n = len(valid)
    if n == 0:
        return float("nan"), []

    bin_rows: list[dict[str, Any]] = []
    ece = 0.0
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        in_bin = [
            r for r in valid
            if lo <= _self_confidence(r) < hi
            or (hi == 1.0 and _self_confidence(r) == 1.0)
        ]
        bn = len(in_bin)
        if bn == 0:
            bin_rows.append({
                "bin": i + 1,
                "range": f"[{lo:.1f},{hi:.1f})",
                "n": 0,
                "mean_conf": None,
                "accuracy": None,
                "gap": None,
            })
            continue
        mean_conf = sum(_self_confidence(r) for r in in_bin) / bn
        acc = sum(_strict_correct(r) for r in in_bin) / bn
        gap = abs(mean_conf - acc)
        ece += (bn / n) * gap
        bin_rows.append({
            "bin": i + 1,
            "range": f"[{lo:.1f},{hi:.1f})" if hi < 1.0 else f"[{lo:.1f},1.0]",
            "n": bn,
            "mean_conf": round(mean_conf, 4),
            "accuracy": round(acc, 4),
            "gap": round(gap, 4),
        })
    return round(ece, 4), bin_rows


def _mean_p_chosen(records: list[dict[str, Any]], predicate) -> float | None:
    vals = [
        float(r["p_chosen"]) for r in records
        if r.get("p_chosen") is not None and not _is_invalid(r) and predicate(r)
    ]
    return round(sum(vals) / len(vals), 4) if vals else None


def _ece_logprob(records: list[dict[str, Any]], n_bins: int = ECE_BINS) -> tuple[float, list[dict[str, Any]]]:
    """Legacy ECE on token p_chosen (sidecar only)."""
    valid = [
        r for r in records
        if r.get("p_chosen") is not None and not _is_invalid(r)
    ]
    n = len(valid)
    if n == 0:
        return float("nan"), []

    bin_rows: list[dict[str, Any]] = []
    ece = 0.0
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        in_bin = [
            r for r in valid
            if lo <= float(r["p_chosen"]) < hi or (hi == 1.0 and float(r["p_chosen"]) == 1.0)
        ]
        bn = len(in_bin)
        if bn == 0:
            bin_rows.append({
                "bin": i + 1, "range": f"[{lo:.1f},{hi:.1f})",
                "n": 0, "mean_conf": None, "accuracy": None, "gap": None,
            })
            continue
        mean_conf = sum(float(r["p_chosen"]) for r in in_bin) / bn
        acc = sum(_strict_correct(r) for r in in_bin) / bn
        gap = abs(mean_conf - acc)
        ece += (bn / n) * gap
        bin_rows.append({
            "bin": i + 1,
            "range": f"[{lo:.1f},{hi:.1f})" if hi < 1.0 else f"[{lo:.1f},1.0]",
            "n": bn,
            "mean_conf": round(mean_conf, 4),
            "accuracy": round(acc, 4),
            "gap": round(gap, 4),
        })
    return round(ece, 4), bin_rows


def _no_ship_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    subset = [
        r for r in records
        if normalise_verdict(r.get("expected_verdict", "")) == "no_ship"
        and r.get("p_verdict_dist")
    ]
    if not subset:
        return {"n": 0, "mean_p_no_ship": None, "frac_p_no_ship_lt_0.01": None}

    ps = [float(r["p_verdict_dist"].get("no_ship", 0.0)) for r in subset]
    return {
        "n": len(subset),
        "mean_p_no_ship": round(sum(ps) / len(ps), 6),
        "frac_p_no_ship_lt_0.01": round(sum(p < 0.01 for p in ps) / len(ps), 4),
    }


def _zone_row(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {}
    n_invalid = sum(1 for r in records if _is_invalid(r))
    strict_acc = sum(_strict_correct(r) for r in records) / n
    antiship_acc = sum(_antiship_ok(r) for r in records) / n
    ece, ece_bins = _ece_self_confidence(records)
    mean_conf = _mean_self_confidence(records, lambda _: True)
    return {
        "n": n,
        "strict_acc": round(strict_acc, 4),
        "antiship_acc": round(antiship_acc, 4),
        "mean_conf": mean_conf,
        "conf_correct": _mean_self_confidence(records, _strict_correct),
        "conf_wrong": _mean_self_confidence(records, lambda r: not _strict_correct(r)),
        "ece": ece,
        "ece_bins": ece_bins,
        "invalid_pct": round(100 * n_invalid / n, 2),
        "n_invalid": n_invalid,
        "conf_minus_acc": round((mean_conf or 0) - strict_acc, 4),
    }


def _write_reliability_csv(path: Path, bin_rows: list[dict[str, Any]], *, zone: str | None = None) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        fields = ["zone", "bin", "range", "n", "bin_conf", "bin_acc", "gap"] if zone else [
            "bin", "range", "n", "bin_conf", "bin_acc", "gap",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for b in bin_rows:
            row = {
                "bin": b["bin"],
                "range": b["range"],
                "n": b["n"],
                "bin_conf": b["mean_conf"] if b["mean_conf"] is not None else "",
                "bin_acc": b["accuracy"] if b["accuracy"] is not None else "",
                "gap": b["gap"] if b["gap"] is not None else "",
            }
            if zone:
                row = {"zone": zone, **row}
            writer.writerow(row)


def _sanity_checks(records: list[dict[str, Any]], bin_rows: list[dict[str, Any]]) -> dict[str, Any]:
    dist_sums: list[float] = []
    for r in records:
        dist = r.get("p_verdict_dist")
        if not dist:
            continue
        dist_sums.append(sum(float(dist.get(k, 0.0)) for k in ("ship", "no_ship", "investigate")))

    dist_ok = (
        all(abs(s - 1.0) <= DIST_SUM_TOL for s in dist_sums) if dist_sums else True
    )
    non_empty_bins = [b for b in bin_rows if b["n"] and b["n"] > 0]
    acc_in_range = all(
        0.0 <= b["accuracy"] <= 1.0 for b in non_empty_bins if b["accuracy"] is not None
    )
    conf_in_range = all(
        0.0 <= b["mean_conf"] <= 1.0 for b in non_empty_bins if b["mean_conf"] is not None
    )
    invalid_n = sum(1 for r in records if _is_invalid(r))
    ece_valid_n = sum(
        1 for r in records if _self_confidence(r) is not None and not _is_invalid(r)
    )

    return {
        "p_verdict_dist_sum_equals_1": dist_ok,
        "n_with_dist": len(dist_sums),
        "n_dist_sum_failures": sum(1 for s in dist_sums if abs(s - 1.0) > DIST_SUM_TOL),
        "ece_bins_nonempty": len(non_empty_bins) > 0,
        "ece_n_nonempty_bins": len(non_empty_bins),
        "ece_accuracy_in_0_1": acc_in_range,
        "ece_conf_in_0_1": conf_in_range,
        "invalid_format_excluded_from_ece": invalid_n,
        "ece_sample_n": ece_valid_n,
        "all_sanity_pass": dist_ok and len(non_empty_bins) > 0 and acc_in_range and conf_in_range,
    }


def run_analysis(
    records: list[dict[str, Any]],
    zone_map: dict[str, str],
    out_dir: Path,
) -> dict[str, Any]:
    for r in records:
        r["zone"] = zone_map.get(r["case_id"], "unknown")

    n = len(records)
    n_invalid = sum(1 for r in records if _is_invalid(r))
    strict_acc = sum(_strict_correct(r) for r in records) / n if n else 0.0
    antiship_acc = sum(_antiship_ok(r) for r in records) / n if n else 0.0

    ece, bin_rows = _ece_self_confidence(records)
    mean_conf = _mean_self_confidence(records, lambda _: True)
    conf_correct = _mean_self_confidence(records, _strict_correct)
    conf_wrong = _mean_self_confidence(records, lambda r: not _strict_correct(r))

    headline_recs = [r for r in records if r.get("case_set") == "headline"]
    blind_recs = [r for r in records if r.get("case_set") == "blind"]
    headline_agg = {}
    if headline_recs:
        n_fc = sum(r.get("false_confidence", False) for r in headline_recs)
        headline_agg = {
            "n": len(headline_recs),
            "false_confidence_rate": round(n_fc / len(headline_recs), 4),
            "n_false_confidence": n_fc,
            "mean_confidence": _mean_self_confidence(headline_recs, lambda _: True),
        }
    honesty_probe = aggregate_honesty_probe(blind_recs) if blind_recs else {}

    logprob_ece, _ = _ece_logprob(records)

    by_zone: dict[str, Any] = {}
    zone_reliability: list[dict[str, Any]] = []
    for zone in ("obvious", "grey", "derive", "blind"):
        sub = [r for r in records if r.get("zone") == zone]
        if not sub:
            continue
        row = _zone_row(sub)
        by_zone[zone] = row
        for b in row.get("ece_bins", []):
            zone_reliability.append({"zone": zone, **b})

    no_ship = _no_ship_stats(records)
    sanity = _sanity_checks(records, bin_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_reliability_csv(out_dir / "reliability_data.csv", bin_rows)
    with open(out_dir / "reliability_by_zone.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["zone", "bin", "range", "n", "bin_conf", "bin_acc", "gap"],
        )
        writer.writeheader()
        for row in zone_reliability:
            writer.writerow({
                "zone": row["zone"],
                "bin": row["bin"],
                "range": row["range"],
                "n": row["n"],
                "bin_conf": row["mean_conf"] if row["mean_conf"] is not None else "",
                "bin_acc": row["accuracy"] if row["accuracy"] is not None else "",
                "gap": row["gap"] if row["gap"] is not None else "",
            })

    zone_table_path = out_dir / "zone_calib.json"
    zone_table_path.write_text(
        json.dumps({"by_zone": by_zone, "n_total": n}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "n": n,
        "strict_accuracy": round(strict_acc, 4),
        "antiship_accuracy": round(antiship_acc, 4),
        "invalid_format_rate": round(n_invalid / n, 4) if n else 0.0,
        "n_invalid_format": n_invalid,
        "ece": ece,
        "ece_bins": bin_rows,
        "mean_confidence": mean_conf,
        "mean_confidence_correct": conf_correct,
        "mean_confidence_incorrect": conf_wrong,
        "headline": headline_agg,
        "honesty_probe": honesty_probe,
        "by_zone": by_zone,
        "logprob_sidecar": {
            "ece_p_chosen": logprob_ece,
            "mean_p_chosen_correct": _mean_p_chosen(records, _strict_correct),
            "mean_p_chosen_incorrect": _mean_p_chosen(records, lambda r: not _strict_correct(r)),
            "no_ship": no_ship,
        },
        "sanity": sanity,
    }
    summary_path = out_dir / "calib_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _print_report(summary: dict[str, Any]) -> None:
    print("=" * 60)
    print("  CALIBRATION ANALYSIS (self-reported confidence)")
    print("=" * 60)
    print(f"  n                      : {summary['n']}")
    print(f"  strict accuracy        : {summary['strict_accuracy']:.1%}")
    print(f"  anti-ship accuracy     : {summary['antiship_accuracy']:.1%}")
    print(f"  invalid_format rate    : {summary['invalid_format_rate']:.1%} ({summary['n_invalid_format']})")
    print(f"  ECE (self-conf, 5 bin): {summary['ece']}")
    print(f"  mean confidence        : {summary['mean_confidence']}")
    print(f"  mean conf (correct)    : {summary['mean_confidence_correct']}")
    print(f"  mean conf (incorrect)  : {summary['mean_confidence_incorrect']}")
    h = summary.get("headline") or {}
    if h:
        print(f"  headline false_conf    : {h.get('false_confidence_rate', 0):.1%} "
              f"({h.get('n_false_confidence', 0)}/{h.get('n', 0)})")
    hp = summary.get("honesty_probe") or {}
    if hp:
        print(f"  blind false_conf       : {hp.get('false_confidence_rate', 0):.1%} "
              f"({hp.get('n_false_confidence', 0)}/{hp.get('n', 0)})")
        print(f"  blind mean_confidence  : {hp.get('mean_confidence')}")
        print(f"  blind claimed_reversal : {hp.get('claimed_reversal_rate', 0):.1%}")
    print()
    print("  ECE bins (self-reported confidence):")
    print(f"  {'bin':>3} {'range':>10} {'n':>4} {'mean_conf':>10} {'accuracy':>9} {'|gap|':>7}")
    for b in summary["ece_bins"]:
        mc = f"{b['mean_conf']:.4f}" if b["mean_conf"] is not None else "   —"
        ac = f"{b['accuracy']:.4f}" if b["accuracy"] is not None else "  —"
        gp = f"{b['gap']:.4f}" if b["gap"] is not None else "  —"
        print(f"  {b['bin']:>3} {b['range']:>10} {b['n']:>4} {mc:>10} {ac:>9} {gp:>7}")
    print()
    print("  By zone:")
    print(f"  {'zone':8} {'n':>4} {'strict':>7} {'antiship':>8} {'mean_c':>7} "
          f"{'c_ok':>7} {'c_bad':>7} {'ECE':>7} {'inv%':>6}")
    for zone, z in summary["by_zone"].items():
        print(f"  {zone:8} {z['n']:4d} {z['strict_acc']:7.1%} {z['antiship_acc']:8.1%} "
              f"{z['mean_conf'] or 0:7.3f} {z['conf_correct'] or 0:7.3f} "
              f"{z['conf_wrong'] or 0:7.3f} {z['ece']:7.3f} {z['invalid_pct']:5.1f}%")
    lp = summary.get("logprob_sidecar", {})
    ns = lp.get("no_ship", {})
    print()
    print("  Logprob sidecar (not primary):")
    print(f"    ECE(p_chosen)={lp.get('ece_p_chosen')}  "
          f"mean_p_correct={lp.get('mean_p_chosen_correct')}  "
          f"mean_p_incorrect={lp.get('mean_p_chosen_incorrect')}")
    print(f"    no_ship P(no_ship): n={ns.get('n')}  mean={ns.get('mean_p_no_ship')}  "
          f"frac<0.01={ns.get('frac_p_no_ship_lt_0.01')}")
    print()
    print("  Sanity checks:")
    s = summary["sanity"]
    for k, v in s.items():
        mark = "✓" if v is True else ("✗" if v is False else " ")
        print(f"    [{mark}] {k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration analysis for bench raw JSONL")
    parser.add_argument("--raw", required=True, help="Path to raw JSONL file")
    parser.add_argument("--corpus", default="../ab-factory-demo/40_ab_factory/vk-style/cases_mvp_v2")
    parser.add_argument("--out", default="results/local_calib80", help="Output directory")
    args = parser.parse_args()

    raw_path = Path(args.raw)
    out_dir = Path(args.out)
    zone_map = _load_zone_map(Path(args.corpus))
    records = _load_records(raw_path)
    summary = run_analysis(records, zone_map, out_dir)
    _print_report(summary)
    print(f"\n  → {out_dir / 'calib_summary.json'}")
    print(f"  → {out_dir / 'reliability_data.csv'}")


if __name__ == "__main__":
    main()
