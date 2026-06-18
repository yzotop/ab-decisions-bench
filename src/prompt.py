"""
Prompt builder.

Converts a case (contract.json + data.csv) into a single user-turn prompt.
Both models get the EXACT same string — only the runner passes a different
model ID.

Auto-cases have no case.md, so we build the narrative from contract fields.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def _load(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _load_csv(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fmt_pct(v: str | None) -> str:
    if v is None or v == "":
        return "—"
    try:
        return f"{float(v):+.1%}"
    except ValueError:
        return v


def _fmt_pval(v: str | None) -> str:
    if v is None or v == "":
        return "—"
    try:
        fv = float(v)
        return f"{fv:.4f}"
    except ValueError:
        return v


def build_prompt(case_dir: Path, mode: str = "free") -> str:
    """Build the model prompt for a single case directory."""
    contract = _load(case_dir / "contract.json")
    rows = _load_csv(case_dir / "data.csv")

    case_id = contract["case_id"]
    title = contract.get("title", case_id)
    domain = contract.get("domain", "—")
    duration = contract.get("time", {}).get("horizon_days", "?")
    pm = contract.get("primary_metric", {})
    pm_name = pm.get("name", "revenue")
    pm_mde = pm.get("mde_relative", 0.01)
    practical = contract.get("decision_framework", {}).get(
        "practical_threshold_relative", 0.005)
    alpha = contract.get("stats", {}).get("alpha", 0.05)

    guardrails = contract.get("guardrails", [])

    # ---- Results table (segment=all rows) ----
    all_rows = [r for r in rows if r.get("segment") == "all"]
    control_row = next((r for r in all_rows if r.get("variant") == "control"), {})
    test_row = next((r for r in all_rows if r.get("variant") != "control"), {})

    def cell(row: dict, col: str) -> str:
        v = row.get(col, "")
        try:
            fv = float(v)
            # large absolute numbers → plain int; small → 4 decimals
            return f"{fv:,.0f}" if fv > 100 else f"{fv:.4f}"
        except (ValueError, TypeError):
            return v or "—"

    results_lines = [
        "| Metric | Control | Test | Δ relative | p-value |",
        "|---|---|---|---|---|",
    ]
    for col in [pm_name, "ctr", "cpm", "fillrate", "dau", "retention"]:
        cv = control_row.get(col)
        if cv is None or cv == "":
            continue
        tv = test_row.get(col, "")
        eff = _fmt_pct(test_row.get(f"{col}_effect_relative"))
        pv = _fmt_pval(test_row.get(f"{col}_p_value"))
        results_lines.append(f"| {col} | {cell(control_row, col)} | {cell(test_row, col)} | {eff} | {pv} |")

    results_table = "\n".join(results_lines)

    # ---- Segment rows (if present) ----
    segment_section = ""
    segments = contract.get("segments", [])
    if segments:
        seg_lines = [
            "",
            "### Segment breakdown",
            "",
            f"| Segment | {pm_name} Δ | p-value |",
            "|---|---|---|",
        ]
        for seg in segments:
            seg_test = next(
                (r for r in rows if r.get("segment") == seg and r.get("variant") != "control"),
                None,
            )
            if seg_test:
                eff = _fmt_pct(seg_test.get(f"{pm_name}_effect_relative"))
                pv = _fmt_pval(seg_test.get(f"{pm_name}_p_value"))
                seg_lines.append(f"| {seg} | {eff} | {pv} |")
        segment_section = "\n".join(seg_lines)

    # ---- Guardrail spec ----
    guard_lines = []
    for g in guardrails:
        name = g.get("name", "?")
        max_drop = g.get("max_drop_relative")
        direction = g.get("direction", "neutral")
        if max_drop is not None:
            guard_lines.append(f"- {name}: direction={direction}, max_drop={max_drop:.1%}")
        else:
            guard_lines.append(f"- {name}: direction={direction}")
    guardrails_text = "\n".join(guard_lines) if guard_lines else "- (none specified)"

    notes = contract.get("notes", "")
    notes_section = f"\n## Notes\n{notes}\n" if notes else ""

    # ---- Assemble prompt (body only; task block appended by mode) ----
    prompt = f"""You are a senior analytics decision-maker. Review the A/B experiment below and return a decision.

## Experiment: {title}
- Domain: {domain}
- Duration: {duration} days
- Primary metric: {pm_name} (direction: up, MDE: {pm_mde:.1%})
- Practical significance threshold: {practical:.1%}
- Statistical significance level (α): {alpha}

### Guardrails
{guardrails_text}
{notes_section}
### Results
{results_table}{segment_section}
"""

    _FREE_TASK = """## Task

Based on the experiment description and data above, decide whether to ship, not ship, or investigate further.

Return ONLY a JSON object with this exact schema — no other text:

```json
{
  "verdict": "ship | no_ship | investigate",
  "mechanism": "one of: guardrail_violation | practically_small | segment_conflict | long_term_reversal | not_significant | none",
  "confidence": <float 0.0–1.0>,
  "rationale": "<one paragraph, ≤120 words explaining the key reason for this verdict>"
}
```"""

    _SGR_TASK = """## Task
Reason through the checks IN ORDER. Fill each check with one short finding (≤20 words). Do not skip a step. Your verdict must follow from what the checks reveal — not the other way round.
Return ONLY a JSON object with this exact schema — no other text:
```json
{
  "checks": {
    "significance":  "<is the primary metric's effect significant at alpha? compare p-value to alpha>",
    "practical":     "<is the effect above the practical threshold? compare delta to threshold>",
    "guardrails":    "<is any guardrail breached beyond its max_drop? name it, or 'none'>",
    "segments":      "<do segments contradict the headline or hide a negative? name it, or 'none'>",
    "reversal_risk": "<could the headline reverse over a longer horizon — and is that signal even visible in this data?>"
  },
  "verdict": "ship | no_ship | investigate",
  "mechanism": "one of: guardrail_violation | practically_small | segment_conflict | long_term_reversal | not_significant | none",
  "confidence": <float 0.0-1.0>,
  "rationale": "<one paragraph, <=120 words, consistent with the checks above>"
}
```"""

    if mode == "sgr":
        prompt += _SGR_TASK
    else:
        prompt += _FREE_TASK

    return prompt


if __name__ == "__main__":
    import sys
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    print(build_prompt(p))
