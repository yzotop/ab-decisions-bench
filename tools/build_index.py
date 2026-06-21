#!/usr/bin/env python3
"""Build results/INDEX.md from run_manifest.json + summary/calib_summary files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
INDEX_PATH = RESULTS / "INDEX.md"

HEADER = (
    "> Автогенерация из run_manifest + summary, не редактировать руками. "
    "Пересобрать: `python tools/build_index.py`\n"
)

SECTION_BREAK = (
    "\n---\n\n"
    "**↓ другая шкала / корпус — в лоб со строками выше не сравнивать**\n"
)

WARNING = """
## Сравнимость

Единая «лидерборд»-таблица намеренно **не используется**: соседние строки из разных секций провоцируют ложное сравнение.

**Три оси несравнимости (между секциями и частично внутри):**

| # | Ось | Что ломает сравнение |
|---|-----|----------------------|
| 1 | **Шкала confidence** | `self_reported` (Gemma, Claude) vs `logprob` / `p_chosen` (Qwen MLX). ECE и пары conf_corr/conf_wrong живут на разных шкалах — **не класть logprob-ECE Qwen в один ряд с self-ECE Gemma**. |
| 2 | **Corpus** | `cases_auto` (v1), `cases_auto_v2` (v2), `calib330` / `cases_mvp_v2 + calib330_cases.json` (локальные) — разный состав, n и сложность. Headline между corpus **не сравнивать**. |
| 3 | **Пайплайн / артефакты** | Claude: `summary.json` без ECE, без zone (`obvious/grey/derive/blind`), без пары conf_corr/conf_wrong. Локальные: полный `calib_summary.json` + `zone_calib.json`. |

**Qwen conf_corr/conf_wrong** — это `mean_p_chosen_correct` / `mean_p_chosen_incorrect` (logprob на финальном verdict-токене), **не** self-reported. Ранние заметки с инверсией ~0.82/0.89 относились к другой шкале/срезу; в calib330 free актуальная пара **0.91/0.93 (logp, p_chosen)**.

Сравнимо **внутри секции** (одна шкала + один corpus): Qwen free vs SGR; Claude v1 vs v2 между собой по знакам, не по абсолютам с локальными.
"""

COLUMNS = [
    "run_id",
    "model",
    "backend",
    "quant",
    "mode",
    "corpus",
    "conf_signal",
    "temp",
    "n",
    "ECE",
    "false_conf%",
    "conf_corr/conf_wrong",
]

SECTIONS: list[tuple[str, str, frozenset[str]]] = [
    (
        "## Локальные · self_reported шкала",
        "Gemma · calib330 · `confidence` в JSON. Сравнимо только внутри секции.",
        frozenset({"gemma_calib330"}),
    ),
    (
        "## Локальные · logprob шкала",
        "Qwen MLX · `p_chosen` на verdict-токене. **free vs sgr** на одном corpus — сравнимы.",
        frozenset({"local_calib330", "local_calib330_sgr"}),
    ),
    (
        "## Claude API · self_reported · другой корpus",
        "v1 (`cases_auto`) и v2 (`cases_auto_v2`). Модели и режимы сравнимы **между Claude-строками**, не с локальными.",
        frozenset({"v1/free", "v1/sgr", "v2/free", "v2/sgr"}),
    ),
]


@dataclass
class RawRow:
    run_id: str
    model: str
    backend: str
    quant: str
    mode: str
    corpus: str
    conf_signal: str
    temp: str
    n: str
    ece: float | None
    false_conf: float | None
    conf_corr: float | None
    conf_wrong: float | None
    scale: str  # "self" | "logp" | "none"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{100 * value:.1f}%"


def _fmt_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _fmt_conf_pair(corr: float | None, wrong: float | None, scale: str) -> str:
    if corr is None and wrong is None:
        return "—"
    pair = f"{_fmt_float(corr)}/{_fmt_float(wrong)}"
    if scale == "self":
        return f"{pair} (self)"
    if scale == "logp":
        return f"{pair} (logp, p_chosen)"
    return pair


def _find_manifests() -> list[Path]:
    manifests: list[Path] = []
    if not RESULTS.is_dir():
        return manifests
    for path in sorted(RESULTS.rglob("run_manifest.json")):
        if "_archive" in path.parts:
            continue
        manifests.append(path)
    return manifests


def _run_id(run_dir: Path) -> str:
    return str(run_dir.relative_to(RESULTS))


def _models_from_manifest(manifest: dict[str, Any]) -> list[str]:
    raw = manifest.get("model") or ""
    return [m.strip() for m in str(raw).split(",") if m.strip()]


def _headline_false_conf(summary: dict[str, Any], model: str) -> float | None:
    headline = summary.get("headline")
    if not isinstance(headline, dict):
        return None
    block = headline.get(model)
    if isinstance(block, dict):
        val = block.get("false_confidence_rate")
        return float(val) if val is not None else None
    val = headline.get("false_confidence_rate")
    return float(val) if val is not None else None


def _headline_n(summary: dict[str, Any], model: str, fallback: Any) -> str:
    headline = summary.get("headline")
    if isinstance(headline, dict):
        block = headline.get(model)
        if isinstance(block, dict) and block.get("n") is not None:
            return str(block["n"])
    if fallback is not None:
        return str(fallback)
    return "—"


def _infer_scale(manifest: dict[str, Any], calib: dict[str, Any] | None) -> str:
    signal = str(manifest.get("confidence_signal") or "").lower()
    if signal == "logprob" or signal == "logp":
        return "logp"
    if signal == "self_reported" or signal == "self":
        return "self"
    if calib and "mean_confidence" in calib:
        return "self"
    if calib and "mean_p_chosen_correct" in calib:
        return "logp"
    return "none"


def _metrics_from_calib(
    calib: dict[str, Any],
) -> tuple[float | None, float | None, float | None, float | None]:
    """Return (ece, false_conf, conf_corr, conf_wrong)."""
    if "mean_confidence" in calib or "mean_confidence_correct" in calib:
        ece = calib.get("ece")
        fc = None
        headline = calib.get("headline")
        if isinstance(headline, dict):
            fc = headline.get("false_confidence_rate")
        return (
            float(ece) if ece is not None else None,
            float(fc) if fc is not None else None,
            float(calib["mean_confidence_correct"]) if calib.get("mean_confidence_correct") is not None else None,
            float(calib["mean_confidence_incorrect"]) if calib.get("mean_confidence_incorrect") is not None else None,
        )

    ece = calib.get("ece")
    return (
        float(ece) if ece is not None else None,
        None,
        float(calib["mean_p_chosen_correct"]) if calib.get("mean_p_chosen_correct") is not None else None,
        float(calib["mean_p_chosen_incorrect"]) if calib.get("mean_p_chosen_incorrect") is not None else None,
    )


def _raw_rows_for_run(run_dir: Path, manifest: dict[str, Any]) -> list[RawRow]:
    calib = _load_json(run_dir / "calib_summary.json")
    summary = _load_json(run_dir / "summary.json")
    models = _models_from_manifest(manifest)
    if not models:
        models = ["—"]

    run_id = _run_id(run_dir)
    scale = _infer_scale(manifest, calib)
    calib_metrics = _metrics_from_calib(calib) if calib else None
    multi = len(models) > 1

    rows: list[RawRow] = []
    for model in models:
        ece: float | None = None
        fc: float | None = None
        conf_corr: float | None = None
        conf_wrong: float | None = None

        if calib_metrics and not multi:
            ece, fc, conf_corr, conf_wrong = calib_metrics

        if summary and fc is None:
            fc = _headline_false_conf(summary, model)

        if calib_metrics and not multi and fc is None and calib:
            headline = calib.get("headline")
            if isinstance(headline, dict) and headline.get("false_confidence_rate") is not None:
                fc = float(headline["false_confidence_rate"])

        if summary:
            n = _headline_n(summary, model, manifest.get("n_cases"))
        else:
            n = str(manifest.get("n_cases") or "—")

        rows.append(
            RawRow(
                run_id=run_id,
                model=model,
                backend=str(manifest.get("backend") or "—"),
                quant=str(manifest.get("quant") or "—"),
                mode=str(manifest.get("mode") or "—"),
                corpus=str(manifest.get("corpus") or "—"),
                conf_signal=str(manifest.get("confidence_signal") or "—"),
                temp=str(manifest.get("temperature") if manifest.get("temperature") is not None else "—"),
                n=n,
                ece=ece,
                false_conf=fc,
                conf_corr=conf_corr,
                conf_wrong=conf_wrong,
                scale=scale,
            )
        )
    return rows


def _render_row(row: RawRow) -> dict[str, str]:
    return {
        "run_id": row.run_id,
        "model": row.model,
        "backend": row.backend,
        "quant": row.quant,
        "mode": row.mode,
        "corpus": row.corpus,
        "conf_signal": row.conf_signal,
        "temp": row.temp,
        "n": row.n,
        "ECE": _fmt_float(row.ece),
        "false_conf%": _fmt_pct(row.false_conf),
        "conf_corr/conf_wrong": _fmt_conf_pair(row.conf_corr, row.conf_wrong, row.scale),
    }


def _anticalib_label(corr: float | None, wrong: float | None, scale: str) -> str:
    if corr is None or wrong is None:
        return "—"
    yes = wrong > corr
    tag = {"self": "self", "logp": "logp, p_chosen"}.get(scale, scale)
    word = "да" if yes else "нет"
    return f"{word} ({tag})"


def _sgr_ece_label(free: RawRow | None, sgr: RawRow | None, scale: str) -> str:
    if free is None or sgr is None or free.ece is None or sgr.ece is None:
        return "—"
    delta = sgr.ece - free.ece
    sign = "+" if delta >= 0 else ""
    worsens = "да" if delta > 0 else ("нет" if delta < 0 else "≈0")
    tag = {"self": "self", "logp": "logp"}.get(scale, scale)
    return f"{worsens}, ΔECE {sign}{delta:.2f} ({tag})"


def _pick_row(rows: list[RawRow], run_id: str) -> RawRow | None:
    matches = [r for r in rows if r.run_id == run_id]
    return matches[0] if matches else None


def _pattern_table(all_rows: list[RawRow]) -> str:
    gemma = _pick_row(all_rows, "gemma_calib330")
    qwen_free = _pick_row(all_rows, "local_calib330")
    qwen_sgr = _pick_row(all_rows, "local_calib330_sgr")

    antic_qwen = _anticalib_label(
        qwen_free.conf_corr if qwen_free else None,
        qwen_free.conf_wrong if qwen_free else None,
        "logp",
    )
    antic_gemma = _anticalib_label(
        gemma.conf_corr if gemma else None,
        gemma.conf_wrong if gemma else None,
        "self",
    )

    sgr_qwen = _sgr_ece_label(qwen_free, qwen_sgr, "logp")
    sgr_gemma = "— (SGR не гонялся)"

    lines = [
        "## Кросс-модельные паттерны (знаки, не абсолюты)",
        "",
        "Единственная таблица, где **межмодельное** сравнение осмысленно: только **знак/наличие** эффекта, не величины.",
        "",
        "| паттерн | Qwen | Gemma | Claude (v2) |",
        "| --- | --- | --- | --- |",
        f"| conf_wrong > conf_correct (антикалибровка) | {antic_qwen} | {antic_gemma} | — (нет conf_corr/wrong в summary) |",
        f"| SGR ухудшает ECE | {sgr_qwen} | {sgr_gemma} | — (нет ECE в summary) |",
        "",
    ]
    return "\n".join(lines)


def _render_table(rows: list[RawRow]) -> list[str]:
    lines = [
        "| " + " | ".join(COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(COLUMNS)) + " |",
    ]
    for row in rows:
        rendered = _render_row(row)
        lines.append("| " + " | ".join(rendered[col] for col in COLUMNS) + " |")
    return lines


def build_index() -> str:
    all_rows: list[RawRow] = []
    for manifest_path in _find_manifests():
        manifest = _load_json(manifest_path)
        if not manifest:
            continue
        all_rows.extend(_raw_rows_for_run(manifest_path.parent, manifest))

    lines = [HEADER, "", "# Results index", "", _pattern_table(all_rows), ""]

    section_outputs: list[list[str]] = []
    for title, subtitle, run_ids in SECTIONS:
        section_rows = sorted(
            (r for r in all_rows if r.run_id in run_ids),
            key=lambda r: (r.run_id, r.model),
        )
        if not section_rows:
            continue
        block = [title, "", subtitle, ""] + _render_table(section_rows)
        section_outputs.append(block)

    for i, block in enumerate(section_outputs):
        lines.extend(block)
        if i < len(section_outputs) - 1:
            lines.append(SECTION_BREAK)

    lines.append(WARNING)
    return "\n".join(lines) + "\n"


def main() -> None:
    text = build_index()
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(text, encoding="utf-8")
    print(f"Wrote {INDEX_PATH.relative_to(ROOT)} ({text.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
