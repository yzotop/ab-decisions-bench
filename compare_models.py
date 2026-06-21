#!/usr/bin/env python3
"""Cross-model comparison: Qwen vs Gemma, free vs SGR by zone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calib_analysis import _load_records, _load_zone_map, run_analysis

ZONES = ("obvious", "grey", "derive", "blind")
COLS = ("strict_acc", "antiship_acc", "ece", "conf_correct", "conf_wrong", "no_ship_frac_lt_0.01")


def _load_zone_table(results_dir: Path, corpus: Path) -> dict[str, dict]:
    raw = next((results_dir / "raw").glob("*.jsonl"))
    zone_map = _load_zone_map(corpus)
    records = _load_records(raw)
    summary = run_analysis(records, zone_map, results_dir)
    return summary["by_zone"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="../ab-factory-demo/40_ab_factory/vk-style/cases_mvp_v2")
    p.add_argument("--qwen-free", default="results/local_calib330")
    p.add_argument("--qwen-sgr", default="results/local_calib330_sgr")
    p.add_argument("--gemma-free", default="results/gemma_calib330")
    p.add_argument("--gemma-sgr", default="results/gemma_calib330_sgr")
    p.add_argument("--out", default="results/gemma_calib330/cross_model.json")
    args = p.parse_args()
    corpus = Path(args.corpus)

    data = {
        "Qwen_free": _load_zone_table(Path(args.qwen_free), corpus),
        "Qwen_sgr": _load_zone_table(Path(args.qwen_sgr), corpus),
        "Gemma_free": _load_zone_table(Path(args.gemma_free), corpus),
        "Gemma_sgr": _load_zone_table(Path(args.gemma_sgr), corpus),
    }
    Path(args.out).write_text(json.dumps(data, indent=2), encoding="utf-8")

    print("=" * 100)
    print("  QWEN vs GEMMA — pattern check (strict_acc / ECE / conf✓ / conf✗ / no_ship_blind%)")
    print("=" * 100)
    for zone in ZONES:
        print(f"\n  [{zone}]")
        print(f"  {'model_mode':14} {'strict':>7} {'antiship':>8} {'ECE':>7} {'conf✓':>7} {'conf✗':>7} {'no<1%':>7} {'inv%':>5}")
        for label, key in [("Qwen free", "Qwen_free"), ("Qwen SGR", "Qwen_sgr"),
                           ("Gemma free", "Gemma_free"), ("Gemma SGR", "Gemma_sgr")]:
            z = data[key].get(zone, {})
            ns = z.get("no_ship_frac_lt_0.01")
            ns_s = f"{ns:.0%}" if ns is not None else "  —"
            cc = z.get("conf_correct")
            cw = z.get("conf_wrong")
            print(f"  {label:14} {z.get('strict_acc',0):7.1%} {z.get('antiship_acc',0):8.1%} "
                  f"{z.get('ece',0):7.3f} {cc or 0:7.3f} {cw or 0:7.3f} {ns_s:>7} {z.get('invalid_pct',0):4.0f}%")
    print(f"\n  → {args.out}")


if __name__ == "__main__":
    main()
