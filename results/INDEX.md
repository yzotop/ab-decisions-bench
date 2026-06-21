> Автогенерация из run_manifest + summary, не редактировать руками. Пересобрать: `python tools/build_index.py`


# Results index

## Кросс-модельные паттерны (знаки, не абсолюты)

Единственная таблица, где **межмодельное** сравнение осмысленно: только **знак/наличие** эффекта, не величины.

| паттерн | Qwen | Gemma | Claude (v2) |
| --- | --- | --- | --- |
| conf_wrong > conf_correct (антикалибровка) | да (logp, p_chosen) | да (self) | — (нет conf_corr/wrong в summary) |
| SGR ухудшает ECE | да, ΔECE +0.21 (logp) | — (SGR не гонялся) | — (нет ECE в summary) |


## Локальные · self_reported шкала

Gemma · calib330 · `confidence` в JSON. Сравнимо только внутри секции.

| run_id | model | backend | quant | mode | corpus | conf_signal | temp | n | ECE | false_conf% | conf_corr/conf_wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma_calib330 | mlx-community/gemma-2-9b-it-4bit | mlx | 4bit | free | cases_mvp_v2 + data/calib330_cases.json | self_reported | 0.7 | 295 | 0.56 | 54.6% | 0.73/0.86 (self) |

---

**↓ другая шкала / корпус — в лоб со строками выше не сравнивать**

## Локальные · logprob шкала

Qwen MLX · `p_chosen` на verdict-токене. **free vs sgr** на одном corpus — сравнимы.

| run_id | model | backend | quant | mode | corpus | conf_signal | temp | n | ECE | false_conf% | conf_corr/conf_wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local_calib330 | mlx-community/Qwen2.5-7B-Instruct-4bit | mlx | 4bit | free | cases_mvp_v2 + data/calib330_cases.json | logprob | 0 | 295 | 0.42 | 32.2% | 0.91/0.93 (logp, p_chosen) |
| local_calib330_sgr | mlx-community/Qwen2.5-7B-Instruct-4bit | mlx | 4bit | sgr | cases_mvp_v2 + data/calib330_cases.json | logprob | 0 | 295 | 0.63 | 46.4% | 0.95/0.97 (logp, p_chosen) |

---

**↓ другая шкала / корпус — в лоб со строками выше не сравнивать**

## Claude API · self_reported · другой корpus

v1 (`cases_auto`) и v2 (`cases_auto_v2`). Модели и режимы сравнимы **между Claude-строками**, не с локальными.

| run_id | model | backend | quant | mode | corpus | conf_signal | temp | n | ECE | false_conf% | conf_corr/conf_wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1/free | claude-haiku-4-5-20251001 | anthropic-api | none | free | cases_auto | self_reported | not passed | 85 | — | 0.0% | — |
| v1/free | claude-opus-4-7 | anthropic-api | none | free | cases_auto | self_reported | not passed | 85 | — | 2.4% | — |
| v1/free | claude-opus-4-8 | anthropic-api | none | free | cases_auto | self_reported | not passed | 85 | — | 0.0% | — |
| v1/free | claude-sonnet-4-6 | anthropic-api | none | free | cases_auto | self_reported | not passed | 85 | — | 0.0% | — |
| v1/sgr | claude-haiku-4-5-20251001 | anthropic-api | none | sgr | cases_auto | self_reported | not passed | 85 | — | 0.0% | — |
| v1/sgr | claude-opus-4-7 | anthropic-api | none | sgr | cases_auto | self_reported | not passed | 85 | — | 0.0% | — |
| v1/sgr | claude-opus-4-8 | anthropic-api | none | sgr | cases_auto | self_reported | not passed | 85 | — | 0.0% | — |
| v1/sgr | claude-sonnet-4-6 | anthropic-api | none | sgr | cases_auto | self_reported | not passed | 85 | — | 0.0% | — |
| v2/free | claude-haiku-4-5-20251001 | anthropic-api | none | free | cases_auto_v2 | self_reported | not passed | 240 | — | 0.0% | — |
| v2/free | claude-sonnet-4-6 | anthropic-api | none | free | cases_auto_v2 | self_reported | not passed | 240 | — | 0.4% | — |
| v2/sgr | claude-haiku-4-5-20251001 | anthropic-api | none | sgr | cases_auto_v2 | self_reported | not passed | 240 | — | 0.0% | — |
| v2/sgr | claude-sonnet-4-6 | anthropic-api | none | sgr | cases_auto_v2 | self_reported | not passed | 240 | — | 0.0% | — |

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

