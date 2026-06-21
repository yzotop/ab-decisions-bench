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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from src.config import CORPUS_DIR, MODELS
from src.scoring import VERDICT_LABELS, normalise_verdict

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


def _write_raw_batch(raw_dir: Path, model: str, records: list[dict]) -> None:
    """Write all records for one model atomically (mode 'w')."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(_raw_path(raw_dir, model), "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _init_raw_file(raw_dir: Path, model: str) -> None:
    """Create/truncate raw JSONL at run start."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    _raw_path(raw_dir, model).write_text("", encoding="utf-8")


def _append_raw_record(raw_dir: Path, model: str, record: dict) -> None:
    """Append one result line immediately (survives mid-run crashes)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(_raw_path(raw_dir, model), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def _build_raw_record(
    result: dict,
    scored: dict,
    model: str,
    mode: str,
) -> dict:
    usage = result.get("usage", {})
    record = {
        "case_id":       result["case_id"],
        "model":         model,
        "mode":          mode,
        "checks":        result.get("checks"),
        "raw_response":  result["raw_response"],
        "verdict":       result["verdict"],
        "mechanism":     result["mechanism"],
        "confidence":    result["confidence"],
        "rationale":     result["rationale"],
        "latency_s":     result["latency_s"],
        "retries":       result["retries"],
        "usage":         usage,
        "case_set":      scored["case_set"],
        "expected_verdict": scored["expected_verdict"],
        "key_reasons":   scored["key_reasons"],
        **{k: v for k, v in scored.items()
           if k not in ("case_set", "expected_verdict", "key_reasons")},
    }
    if result.get("provider"):
        record["provider"] = result["provider"]
    if result.get("base_url"):
        record["base_url"] = result["base_url"]
    if "model_verdict" in result:
        record["model_verdict"] = result["model_verdict"]
    if "p_verdict_dist" in result:
        record["p_verdict_dist"] = result["p_verdict_dist"]
    if "p_chosen" in result:
        record["p_chosen"] = result["p_chosen"]
    if "verdict_first_token" in result:
        record["verdict_first_token"] = result["verdict_first_token"]
    if "verdict_top_logprobs" in result:
        record["verdict_top_logprobs"] = result["verdict_top_logprobs"]
    if "raw_logprobs" in result:
        record["raw_logprobs"] = result["raw_logprobs"]
    if "p_verdict_dist_censored" in result:
        record["p_verdict_dist_censored"] = result["p_verdict_dist_censored"]
    if "topk_floor_logprob" in result:
        record["topk_floor_logprob"] = result["topk_floor_logprob"]
    if result.get("invalid_format"):
        record["invalid_format"] = True
    if result.get("parsed_verdict_raw"):
        record["parsed_verdict_raw"] = result["parsed_verdict_raw"]
    return record


def _format_progress(
    case_name: str,
    model: str,
    result: dict,
    scored: dict,
    in_tok: int,
    out_tok: int,
) -> str:
    cs = scored.get("case_set", "?")
    if cs == "headline":
        ok = "✓" if scored["correct"] else "✗"
        ws = " [WRONG_SHIP!]" if scored["wrong_ship"] else ""
        fc = " [FALSE_CONF]" if scored["false_confidence"] else ""
        return (
            f"{case_name} × {model} ... {ok} [{cs}] "
            f"v={result['verdict']!r:<12} exp={scored['expected_verdict']!r:<12}"
            f" conf={result['confidence']:.2f}  {result['latency_s']:.1f}s{ws}{fc}"
            f"  tok={in_tok}+{out_tok}"
        )
    dns = "✓" if scored["did_not_ship"] else "✗ WRONG_SHIP"
    fc = " [FALSE_CONF]" if scored["false_confidence"] else ""
    cr = " [CLAIMED_REV]" if scored["claimed_reversal"] else ""
    return (
        f"{case_name} × {model} ... {dns} [blind] "
        f"v={result['verdict']!r:<12} conf={result['confidence']:.2f}  "
        f"{result['latency_s']:.1f}s{fc}{cr}  tok={in_tok}+{out_tok}"
    )


def _run_sampling_task(
    case_dir: Path,
    model: str,
    mode: str,
    truth: dict,
    n_samples: int,
    *,
    dry_run: bool,
    run_case_fn,
    mock_run_case_fn,
    score_case_fn,
) -> dict:
    """Run the same case N times; record empirical verdict distribution."""
    from collections import Counter

    counts: Counter[str] = Counter()
    confidences: list[float] = []
    latencies: list[float] = []
    first_result: dict | None = None
    total_in = total_out = 0

    for _ in range(n_samples):
        if dry_run:
            result = mock_run_case_fn(case_dir, model, mode=mode, truth=truth)
        else:
            result = run_case_fn(case_dir, model, mode=mode)
        if first_result is None:
            first_result = result
        v = normalise_verdict(result.get("verdict", ""))
        counts[v] += 1
        confidences.append(float(result.get("confidence", 0.0)))
        latencies.append(float(result.get("latency_s", 0.0)))
        usage = result.get("usage", {})
        total_in += usage.get("input_tokens", 0)
        total_out += usage.get("output_tokens", 0)

    assert first_result is not None
    mode_verdict, _ = counts.most_common(1)[0]
    rep = {
        **first_result,
        "verdict": mode_verdict,
        "confidence": round(sum(confidences) / len(confidences), 4),
        "latency_s": round(sum(latencies), 3),
    }
    scored = score_case_fn(rep, truth)
    full = {**rep, **scored}
    raw_record = _build_raw_record(rep, scored, model, mode)
    raw_record["sampling"] = True
    raw_record["n_samples"] = n_samples
    raw_record["verdict_counts"] = {k: counts.get(k, 0) for k in VERDICT_LABELS}
    raw_record["verdict_freq"] = {
        k: round(counts.get(k, 0) / n_samples, 4) for k in VERDICT_LABELS
    }
    raw_record["sample_confidences"] = confidences
    return {
        "full": full,
        "raw_record": raw_record,
        "model": model,
        "in_tok": total_in,
        "out_tok": total_out,
        "progress": (
            f"{case_dir.name} × {model} ... [{n_samples}×] "
            f"freq={raw_record['verdict_freq']}  {sum(latencies):.1f}s"
        ),
    }


def _run_one_task(
    case_dir: Path,
    model: str,
    mode: str,
    truth: dict,
    *,
    dry_run: bool,
    run_case_fn,
    mock_run_case_fn,
    score_case_fn,
) -> dict:
    if dry_run:
        result = mock_run_case_fn(case_dir, model, mode=mode, truth=truth)
    else:
        result = run_case_fn(case_dir, model, mode=mode)
    scored = score_case_fn(result, truth)
    full = {**result, **scored}
    usage = result.get("usage", {})
    in_tok = usage.get("input_tokens", 0)
    out_tok = usage.get("output_tokens", 0)
    raw_record = _build_raw_record(result, scored, model, mode)
    return {
        "full": full,
        "raw_record": raw_record,
        "model": model,
        "in_tok": in_tok,
        "out_tok": out_tok,
        "progress": _format_progress(
            case_dir.name, model, result, scored, in_tok, out_tok,
        ),
    }


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
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers (default 4, max 5)")
    parser.add_argument("--provider", choices=["anthropic", "local_openai"], default="anthropic",
                        help="API provider (default: anthropic)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1",
                        help="OpenAI-compatible base URL for local_openai")
    parser.add_argument("--local-model", default="mlx-community/Qwen2.5-7B-Instruct-4bit",
                        help="Model id for local_openai provider")
    parser.add_argument("--samples", type=int, default=1,
                        help="Repeat each case N times (temp=0.7 sampling); writes verdict_freq")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use mock responses from tests/fixtures (no API)")
    parser.add_argument("--reanalyse", action="store_true",
                        help="Skip API; load results/raw/*.jsonl and recompute")
    args = parser.parse_args()

    if args.samples < 1:
        parser.error("--samples must be >= 1")
    if args.samples > 1 and args.workers > 1:
        parser.error("--samples requires --workers 1 (sequential sampling)")

    if args.workers < 1 or args.workers > 5:
        parser.error("--workers must be between 1 and 5")

    if not args.dry_run and not args.reanalyse and args.provider == "anthropic":
        _require_api_key()

    from src.runner import run_case, _get_client
    from src.local_runner import run_case_local_openai
    from src.scoring import score_case
    from src.analysis import load_raw_jsonl, write_summary
    mock_run_case_fn = None
    if args.dry_run:
        from tests.fixtures.mock_responses import mock_run_case
        mock_run_case_fn = mock_run_case

    if args.provider == "local_openai":
        args.models = [args.local_model]

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

    tasks: list[tuple[Path, str, dict]] = []
    for case_dir in case_dirs:
        truth_path = case_dir / "truth.json"
        if not truth_path.exists():
            print(f"  SKIP {case_dir.name} — no truth.json")
            continue
        with open(truth_path, encoding="utf-8") as f:
            truth = json.load(f)
        for model in args.models:
            tasks.append((case_dir, model, truth))

    print(f"Corpus : {corpus_path} ({len(case_dirs)} cases)")
    print(f"Provider: {args.provider}")
    print(f"Models : {args.models}")
    if args.provider == "local_openai":
        print(f"Base URL: {args.base_url}")
    print(f"Mode   : {args.mode}")
    if args.samples > 1:
        print(f"Samples: {args.samples} per case (verdict frequency)")
    print(f"Workers: {args.workers}")
    if args.dry_run:
        print("Dry-run: mock responses (no API)")
    print(f"Raw out: {raw_out}")
    print()

    if not tasks:
        print("No tasks to run.")
        return

    if not args.dry_run and args.provider == "anthropic":
        _get_client()

    run_case_fn = run_case
    if args.provider == "local_openai" and not args.dry_run:
        base_url = args.base_url
        run_case_fn = lambda case_dir, model, mode="free": run_case_local_openai(
            case_dir, model, base_url=base_url, mode=mode,
        )

    total = len(tasks)
    all_results: list[dict] = []
    raw_by_model: dict[str, list[dict]] = {m: [] for m in args.models}
    total_input_tokens = total_output_tokens = 0

    if not args.dry_run and not args.reanalyse:
        for model in args.models:
            _init_raw_file(raw_out, model)

    run_task_fn = _run_sampling_task if args.samples > 1 else _run_one_task

    if args.workers == 1:
        # Strictly sequential — avoids concurrent requests to local mlx server.
        done = 0
        for case_dir, model, truth in tasks:
            if args.samples > 1:
                item = run_task_fn(
                    case_dir, model, args.mode, truth, args.samples,
                    dry_run=args.dry_run,
                    run_case_fn=run_case_fn,
                    mock_run_case_fn=mock_run_case_fn,
                    score_case_fn=score_case,
                )
            else:
                item = run_task_fn(
                    case_dir, model, args.mode, truth,
                    dry_run=args.dry_run,
                    run_case_fn=run_case_fn,
                    mock_run_case_fn=mock_run_case_fn,
                    score_case_fn=score_case,
                )
            done += 1
            all_results.append(item["full"])
            raw_by_model[item["model"]].append(item["raw_record"])
            if not args.dry_run:
                _append_raw_record(raw_out, item["model"], item["raw_record"])
            total_input_tokens += item["in_tok"]
            total_output_tokens += item["out_tok"]
            print(f"  [{done}/{total}] {item['progress']}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    _run_one_task,
                    case_dir,
                    model,
                    args.mode,
                    truth,
                    dry_run=args.dry_run,
                    run_case_fn=run_case_fn,
                    mock_run_case_fn=mock_run_case_fn,
                    score_case_fn=score_case,
                )
                for case_dir, model, truth in tasks
            ]
            done = 0
            for fut in as_completed(futures):
                item = fut.result()
                done += 1
                all_results.append(item["full"])
                raw_by_model[item["model"]].append(item["raw_record"])
                if not args.dry_run:
                    _append_raw_record(raw_out, item["model"], item["raw_record"])
                total_input_tokens += item["in_tok"]
                total_output_tokens += item["out_tok"]
                print(f"  [{done}/{total}] {item['progress']}")

    for model in args.models:
        records = raw_by_model[model]
        records.sort(key=lambda r: r["case_id"])
        if args.dry_run:
            _write_raw_batch(raw_out, model, records)

    print(f"\nDone. {len(all_results)}/{total} results."
          f"  Tokens used: {total_input_tokens} in + {total_output_tokens} out")
    write_summary(all_results, args.models, results_out)


if __name__ == "__main__":
    main()
