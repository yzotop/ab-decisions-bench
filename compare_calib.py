#!/usr/bin/env python3
"""Compare free vs SGR calibration by zone on the same case set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calib_analysis import _load_records, _load_zone_map, run_analysis

ZONES = ("obvious", "grey", "derive", "blind")
METRICS = (
    "strict_acc", "antiship_acc", "ece", "mean_conf",
    "conf_correct", "conf_wrong", "invalid_pct",
)


def _load_summary(raw: Path, corpus: Path, out: Path) -> dict:
    zone_map = _load_zone_map(corpus)
    records = _load_records(raw)
    return run_analysis(records, zone_map, out)


def _delta(free: float | None, sgr: float | None) -> str:
    if free is None or sgr is None:
        return "—"
    d = sgr - free
    sign = "+" if d > 0 else ""
    if abs(d) < 0.0001:
        return "0"
    if abs(d) < 0.01:
        return f"{sign}{d:.3f}"
    return f"{sign}{d:.2f}"


def _fmt(v: float | None, pct: bool = False) -> str:
    if v is None:
        return "—"
    if pct:
        return f"{v:.1%}"
    return f"{v:.3f}"


def compare(free_dir: Path, sgr_dir: Path, corpus: Path, out: Path) -> dict:
    free_raw = next((free_dir / "raw").glob("*.jsonl"))
    sgr_raw = next((sgr_dir / "raw").glob("*.jsonl"))
    free = _load_summary(free_raw, corpus, free_dir)
    sgr = _load_summary(sgr_raw, corpus, sgr_dir)

    rows = []
    for zone in ZONES:
        fz = free["by_zone"].get(zone, {})
        sz = sgr["by_zone"].get(zone, {})
        row = {"zone": zone}
        for m in METRICS:
            row[f"free_{m}"] = fz.get(m)
            row[f"sgr_{m}"] = sz.get(m)
            row[f"delta_{m}"] = (
                None if fz.get(m) is None or sz.get(m) is None else round(sz[m] - fz[m], 4)
            )
        rows.append(row)

    payload = {
        "free_n": free["n"],
        "sgr_n": sgr["n"],
        "free_overall": {
            "strict_acc": free["strict_accuracy"],
            "antiship_acc": free["antiship_accuracy"],
            "ece": free["ece"],
            "invalid_rate": free["invalid_format_rate"],
        },
        "sgr_overall": {
            "strict_acc": sgr["strict_accuracy"],
            "antiship_acc": sgr["antiship_accuracy"],
            "ece": sgr["ece"],
            "invalid_rate": sgr["invalid_format_rate"],
        },
        "by_zone": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _print_table(payload: dict) -> None:
    print("=" * 72)
    print("  FREE vs SGR — zone calibration comparison")
    print("=" * 72)
    fo, so = payload["free_overall"], payload["sgr_overall"]
    print(f"  Overall strict: free {fo['strict_acc']:.1%} → sgr {so['strict_acc']:.1%}  "
          f"(Δ {_delta(fo['strict_acc'], so['strict_acc'])})")
    print(f"  Overall ECE:    free {fo['ece']:.3f} → sgr {so['ece']:.3f}  "
          f"(Δ {_delta(fo['ece'], so['ece'])})")
    print()
    hdr = (
        f"{'zone':8} │ {'strict':^17} │ {'ECE':^17} │ "
        f"{'conf✓/✗ (free→sgr)':^24} │ inv%"
    )
    print(hdr)
    print("─" * len(hdr))
    for row in payload["by_zone"]:
        z = row["zone"]
        print(
            f"{z:8} │ "
            f"{_fmt(row['free_strict_acc'], True):>7}→{_fmt(row['sgr_strict_acc'], True):<7} "
            f"({_delta(row['free_strict_acc'], row['sgr_strict_acc']):>5}) │ "
            f"{row['free_ece'] or 0:6.3f}→{row['sgr_ece'] or 0:<6.3f} "
            f"({_delta(row['free_ece'], row['sgr_ece']):>5}) │ "
            f"{row['free_conf_wrong'] or 0:.2f}/{row['free_conf_correct'] or 0:.2f}"
            f"→{row['sgr_conf_wrong'] or 0:.2f}/{row['sgr_conf_correct'] or 0:.2f} │ "
            f"{row['free_invalid_pct']:.0f}→{row['sgr_invalid_pct']:.0f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare free vs SGR calibration")
    parser.add_argument("--free-dir", default="results/local_calib330")
    parser.add_argument("--sgr-dir", default="results/local_calib330_sgr")
    parser.add_argument("--corpus", default="../ab-factory-demo/40_ab_factory/vk-style/cases_mvp_v2")
    parser.add_argument("--out", default="results/local_calib330_sgr/free_vs_sgr.json")
    args = parser.parse_args()

    payload = compare(
        Path(args.free_dir), Path(args.sgr_dir), Path(args.corpus), Path(args.out),
    )
    _print_table(payload)
    print(f"\n  → {args.out}")


if __name__ == "__main__":
    main()
