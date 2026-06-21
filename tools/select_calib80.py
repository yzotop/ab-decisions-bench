#!/usr/bin/env python3
"""Stratified case selection for calibration semi-run (~80 cases)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO.parent / "ab-factory-demo/40_ab_factory/vk-style/cases_mvp_v2"
TRAP_ZONES = CORPUS.parent / "trainer/trap_zones.json"
OUT = REPO / "data/calib80_cases.json"

# Zone targets (~80 total, representative mix per bench plan).
ZONE_TARGETS = {
    "obvious": 28,
    "grey": 16,
    "derive": 24,
    "blind": 8,
}


def _round_robin_select(by_trap: dict[str, list[str]], n: int) -> list[str]:
    traps = sorted(by_trap.keys())
    indices = {t: 0 for t in traps}
    picked: list[str] = []
    while len(picked) < n:
        progressed = False
        for trap in traps:
            if len(picked) >= n:
                break
            idx = indices[trap]
            pool = by_trap[trap]
            if idx < len(pool):
                picked.append(pool[idx])
                indices[trap] += 1
                progressed = True
        if not progressed:
            break
    return picked


def main() -> None:
    labels = json.loads((CORPUS / "_trap_labels.json").read_text(encoding="utf-8"))
    zones_map = json.loads(TRAP_ZONES.read_text(encoding="utf-8"))

    by_zone_trap: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for case_id, trap in sorted(labels.items()):
        zone = zones_map[trap]
        by_zone_trap[zone][trap].append(case_id)

    selected: list[str] = []
    zone_counts: dict[str, int] = {}
    trap_mix: dict[str, dict[str, int]] = {}

    for zone, target in ZONE_TARGETS.items():
        picked = _round_robin_select(by_zone_trap[zone], target)
        if len(picked) < target:
            raise SystemExit(f"Not enough cases in zone {zone}: need {target}, got {len(picked)}")
        selected.extend(picked)
        zone_counts[zone] = len(picked)
        trap_mix[zone] = {}
        for cid in picked:
            trap = labels[cid]
            trap_mix[zone][trap] = trap_mix[zone].get(trap, 0) + 1

    payload = {
        "n": len(selected),
        "zone_targets": ZONE_TARGETS,
        "zone_counts": zone_counts,
        "trap_mix": trap_mix,
        "cases": selected,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(selected)} cases → {OUT}")
    for zone, cnt in zone_counts.items():
        print(f"  {zone}: {cnt} ({len(trap_mix[zone])} trap types)")


if __name__ == "__main__":
    main()
