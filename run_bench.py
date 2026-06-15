"""
Benchmark entry point — per AUDIT.md spec.

Usage:
    python run_bench.py --limit 3        # smoke test (3 cases × 2 models)
    python run_bench.py                  # full run (100 × 2)
    python run_bench.py --reanalyse      # skip API, reload results/raw/*.jsonl
    python run_bench.py --dry-run ...    # mock responses, no API

Guard: exits with clear message if ANTHROPIC_API_KEY is not set (skipped for --dry-run).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from src.config import CORPUS_DIR, MODELS

load_dotenv()


def _require_api_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        print(
            "\n[ERROR] ANTHROPIC_API_KEY not set.\n"
            "  1. Copy .env.example → .env\n"
            "  2. Fill in your key.\n"
            "  3. Re-run.\n"
            "\nAll code compiled successfully — ready for smoke test once key is added."
        )
        sys.exit(1)


# ── JSONL helpers ────────────────────────────────────────────────────────────

def _raw_path(raw_dir: Path, model: str) -> Path:
    safe = model.replace("/", "_").replace("-", "_")
    return raw_dir / f"{safe}.jsonl"


def _append_raw(raw_dir: Path, model: str, record: dict) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(_raw_path(raw_dir, model), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="A/B Decisions Benchmark")
    parser.add_argument("--corpus", default=CORPUS_DIR)
    parser.add_argument("--raw-out", default="results/raw", dest="raw_out")
    parser.add_argument("--results-out", default="results", dest="results_out")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit cases (3 → smoke test)")
    parser.add_argument("--cases", nargs="+", default=None,
                        help="Run only these case dir names (overrides --limit)")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--mode", choices=["free", "sgr"], default="free",
                        help="Prompt mode: free-form or schema-guided reasoning")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use mock responses from tests/fixtures (no API)")
    parser.add_argument("--reanalyse", action="store_true",
                        help="Skip API; load results/raw/*.jsonl and recompute")
    args = parser.parse_args()

    if not args.dry_run and not args.reanalyse:
        _require_api_key()

    from src.runner import run_case
    from src.scoring import score_case
    from src.analysis import load_raw_jsonl, write_summary
    if args.dry_run:
        from tests.fixtures.mock_responses import mock_run_case

    raw_out = Path(args.raw_out)
    results_out = Path(args.results_out)

    # ── Reanalysis mode ───────────────────────────────────────────────────────
    if args.reanalyse:
        print("Reanalysis mode — loading from", raw_out)
        all_results: list[dict] = []
        for model in args.models:
            records = load_raw_jsonl(raw_out, model)
            print(f"  {model}: {len(records)} records")
            all_results.extend(records)
        write_summary(all_results, args.models, results_out)
        return

    # ── Normal run ────────────────────────────────────────────────────────────
    corpus_path = Path(args.corpus)
    case_dirs = sorted(p for p in corpus_path.iterdir() if p.is_dir())
    if args.cases:
        wanted = set(args.cases)
        case_dirs = [p for p in case_dirs if p.name in wanted]
        missing = wanted - {p.name for p in case_dirs}
        if missing:
            print(f"  WARN: cases not found: {sorted(missing)}")
    elif args.limit:
        case_dirs = case_dirs[: args.limit]

    print(f"Corpus : {corpus_path} ({len(case_dirs)} cases)")
    print(f"Models : {args.models}")
    print(f"Mode   : {args.mode}")
    if args.dry_run:
        print("Dry-run: mock responses (no API)")
    print(f"Raw out: {raw_out}")
    print()

    all_results = []
    total_input_tokens = total_output_tokens = 0

    for case_dir in case_dirs:
        truth_path = case_dir / "truth.json"
        if not truth_path.exists():
            print(f"  SKIP {case_dir.name} — no truth.json")
            continue
        with open(truth_path, encoding="utf-8") as f:
            truth = json.load(f)

        for model in args.models:
            print(f"  {case_dir.name} × {model} ...", end=" ", flush=True)

            if args.dry_run:
                result = mock_run_case(case_dir, model, mode=args.mode, truth=truth)
            else:
                result = run_case(case_dir, model, mode=args.mode)
            scored = score_case(result, truth)
            full = {**result, **scored}

            # Token accounting
            usage = result.get("usage", {})
            in_tok = usage.get("input_tokens", 0)
            out_tok = usage.get("output_tokens", 0)
            total_input_tokens += in_tok
            total_output_tokens += out_tok

            # Build raw JSONL record (all fields needed for reanalysis)
            raw_record = {
                "case_id":       result["case_id"],
                "model":         model,
                "mode":          args.mode,
                "checks":        result.get("checks"),
                "raw_response":  result["raw_response"],
                "verdict":       result["verdict"],
                "mechanism":     result["mechanism"],
                "confidence":    result["confidence"],
                "rationale":     result["rationale"],
                "latency_s":     result["latency_s"],
                "retries":       result["retries"],
                "usage":         usage,
                # scoring fields (needed for --reanalyse)
                "case_set":      scored["case_set"],
                "expected_verdict": scored["expected_verdict"],
                "key_reasons":   scored["key_reasons"],
                **{k: v for k, v in scored.items()
                   if k not in ("case_set", "expected_verdict", "key_reasons")},
            }
            _append_raw(raw_out, model, raw_record)

            # Inline progress
            cs = scored.get("case_set", "?")
            if cs == "headline":
                ok = "✓" if scored["correct"] else "✗"
                ws = " [WRONG_SHIP!]" if scored["wrong_ship"] else ""
                fc = " [FALSE_CONF]" if scored["false_confidence"] else ""
                print(f"{ok} [{cs}] v={result['verdict']!r:<12} exp={scored['expected_verdict']!r:<12}"
                      f" conf={result['confidence']:.2f}  {result['latency_s']:.1f}s{ws}{fc}"
                      f"  tok={in_tok}+{out_tok}")
            else:
                dns = "✓" if scored["did_not_ship"] else "✗ WRONG_SHIP"
                fc = " [FALSE_CONF]" if scored["false_confidence"] else ""
                cr = " [CLAIMED_REV]" if scored["claimed_reversal"] else ""
                print(f"{dns} [blind] v={result['verdict']!r:<12}"
                      f" conf={result['confidence']:.2f}  {result['latency_s']:.1f}s{fc}{cr}"
                      f"  tok={in_tok}+{out_tok}")

            all_results.append(full)

    n_calls = len(case_dirs) * len(args.models)
    print(f"\nDone. {len(all_results)}/{n_calls} results."
          f"  Tokens used: {total_input_tokens} in + {total_output_tokens} out")
    write_summary(all_results, args.models, results_out)


if __name__ == "__main__":
    main()
