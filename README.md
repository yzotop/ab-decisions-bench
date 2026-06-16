# ab-decisions-bench

Benchmark for LLM quality on A/B experiment decisions.

Compares models on 100 synthetic cases: each case is an experiment contract plus a results table. The model must return a verdict (`ship` / `no_ship` / `investigate`), name the mechanism, and state confidence. Scoring covers headline accuracy (85 observable cases) and an honesty-probe block (15 blind cases).

Two reasoning modes:

- **free** — decide from the data and return JSON
- **sgr** — Schema-Guided Reasoning: five ordered checks before the verdict

## Status

**Working.** Full v1 runs completed for four Claude models in both modes:

| Model | API id |
|---|---|
| Claude Opus 4.8 | `claude-opus-4-8` |
| Claude Opus 4.7 | `claude-opus-4-7` |
| Claude Sonnet 4.6 | `claude-sonnet-4-6` |
| Claude Haiku 4.5 | `claude-haiku-4-5-20251001` |

Headline accuracy (free → sgr): Opus 4.8 97.6% → 89.4%, Opus 4.7 96.5% → 94.1%, Sonnet 4.6 92.9% → 98.8%, Haiku 4.5 89.4% → 89.4%.

**Key finding:** SGR lowers accuracy on strong models and raises it on mid-tier ones — the sign flips with base capability. Opus 4.8: 97.6% → 89.4% (free → sgr); Sonnet 4.6: 92.9% → 98.8%. Write-up: [Когда «думай по схеме» делает модель глупее](https://davydov.my/workspace/articles/sgr-ab-decisions/).

Raw results live under `results/` (gitignored). Recompute summaries with `--reanalyse`.

## Blind by contract

The model sees **only** `contract.json` and `data.csv` for each case. It does **not** receive `truth.json`, `policy.json`, or any audit files.

`src/prompt.py` builds the user prompt exclusively from the contract and the CSV. Ground truth in `data/corpus/*/truth.json` and the aggregated files in `audit/` (`policy.json`, `truth_all.jsonl`, etc.) exist for **post-hoc scoring and reproducibility** — they are never passed to the API.

This is intentional: the benchmark measures judgment from visible inputs, the way a human analyst would decide from numbers and experiment spec alone.

## Corpus

Source: [ab-factory-demo](https://github.com/yzotop/ab-factory-demo) — 100 auto-generated cases in `cases_auto/`.

Per case: `contract.json` + `data.csv` + `truth.json` (truth used only by the scorer).

Symlink or copy into `data/corpus/`:

```bash
ln -s /path/to/ab-factory-demo/40_ab_factory/vk-style/cases_auto data/corpus
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set ANTHROPIC_API_KEY
```

## Run

```bash
# Smoke test (3 cases, default models)
python run_bench.py --limit 3

# Full run, free mode (default)
python run_bench.py

# Schema-guided reasoning
python run_bench.py --mode sgr

# Subset of models and/or cases
python run_bench.py --models claude-sonnet-4-6 claude-haiku-4-5-20251001 --mode free
python run_bench.py --cases case_005 case_015

# Mock responses, no API (tests/fixtures)
python run_bench.py --dry-run --limit 5

# Re-score existing raw JSONL (no API)
python run_bench.py --reanalyse \
  --models claude-opus-4-8 claude-opus-4-7 claude-sonnet-4-6 claude-haiku-4-5-20251001 \
  --raw-out results/v1/free/raw --results-out results/v1/free
```

Useful flags: `--mode {free,sgr}`, `--models`, `--cases`, `--limit`, `--raw-out`, `--results-out`, `--dry-run`, `--reanalyse`.

**Note:** Opus models reject `temperature=0` on the API; the runner omits temperature. Raw JSONL is **append**-only — re-running the same model/case without a fresh `--raw-out` duplicates rows. Use `--reanalyse` after deduplication to rebuild `summary.json`.

## Metrics

- **Headline (n=85):** 3-way verdict accuracy, wrong-ship rate, false-confidence rate, mechanism recall, per-trap breakdown
- **Honesty-probe (n=15):** cases where the decisive signal is absent from `data.csv`; measures `did_not_ship`, wrong-ship, false-confidence, claimed reversal

See [GLOSSARY.md](GLOSSARY.md) for verdict labels, trap types, and metric definitions. Methodology details: [audit/AUDIT.md](audit/AUDIT.md).

## Project structure

```
ab-decisions-bench/
├── audit/               ← ground truth aggregates (not sent to the model)
│   ├── AUDIT.md
│   ├── policy.json
│   └── truth_all.jsonl
├── data/
│   └── corpus/          ← symlink to ab-factory-demo cases_auto/
├── src/
│   ├── config.py        ← models, API params
│   ├── prompt.py        ← contract + CSV → prompt (model input only)
│   ├── runner.py        ← API call, JSON parse, retry
│   ├── scoring.py       ← per-case headline / honesty-probe scoring
│   └── analysis.py      ← aggregation, McNemar, summary.json
├── tests/
│   └── fixtures/        ← smart mocks for --dry-run
├── results/             ← raw JSONL + summary (gitignored)
├── run_bench.py
├── GLOSSARY.md
└── requirements.txt
```
