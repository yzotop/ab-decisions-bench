# ab-decisions-bench
Status: active

MVP benchmark for comparing LLM A/B decision quality across model versions.

## Purpose

Run identical A/B test cases through two Claude models and measure:
- Accuracy on `ship / no_ship / investigate` verdict
- Trap-type accuracy (guardrail violations, practically-small, segment conflict, etc.)
- Confidence calibration
- Disagreement cases (where models differ)

## Corpus

Source: [ab-factory-demo](https://github.com/yzotop/ab-factory-demo) — 100 auto-generated cases + 5 hand-authored.
Each case: `contract.json` (experiment spec) + `data.csv` (results) + `truth.json` (ground truth verdict).

Copy or symlink corpus to `data/corpus/`:
```bash
ln -s /path/to/ab-factory-demo/40_ab_factory/vk-style/cases_auto data/corpus
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in ANTHROPIC_API_KEY
```

## Models under comparison

| Model | Role |
|---|---|
| `claude-opus-4-8` | New version |
| `claude-opus-4-7` | Baseline |

## Run (Phase 2)

```bash
# Smoke test — 5 cases
python run_bench.py --limit 5

# Full benchmark
python run_bench.py
```

## Phase status

| Phase | Status |
|---|---|
| 1 — Corpus analysis + skeleton | ✓ Done |
| 2 — Run + scoring | TODO |
| 3 — Analysis + report | TODO |

## Project structure

```
ab-decisions-bench/
├── data/
│   └── corpus/          ← symlink or copy of cases_auto/
├── src/
│   ├── config.py        ← model list, API params
│   ├── prompt.py        ← case → prompt builder
│   ├── runner.py        ← API call + JSON parse + retry
│   ├── scoring.py       ← TODO Phase 2
│   └── analysis.py      ← TODO Phase 2
├── results/
│   ├── raw/             ← per-case per-model JSON (gitignored)
│   └── summary.json     ← aggregated scores (committed)
├── run_bench.py         ← entry point
├── requirements.txt
├── .env.example
└── README.md
```
