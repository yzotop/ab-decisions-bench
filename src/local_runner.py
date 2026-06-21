"""
Local OpenAI-compatible runner (mlx_lm.server, vLLM, etc.).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from .prompt import build_prompt
from .runner import _extract_json, _validate_verdict
from .config import SGR_MAX_TOKENS
from .verdict_logprobs import attach_p_chosen, extract_verdict_logprobs

# 2 retries after the first attempt (3 total) — enough for weak local models.
LOCAL_MAX_RETRIES = 2

# mlx_lm.server caps top_logprobs at 11 (not 20).
LOCAL_TOP_LOGPROBS = 11
LOCAL_MAX_TOKENS_FREE = 180

# Calibration runs: sample from the distribution, not greedy argmax.
LOCAL_TEMPERATURE = 0.7

def _api_params(mode: str) -> dict[str, Any]:
    max_tokens = SGR_MAX_TOKENS if mode == "sgr" else LOCAL_MAX_TOKENS_FREE
    return {
        "temperature": LOCAL_TEMPERATURE,
        "logprobs": True,
        "top_logprobs": LOCAL_TOP_LOGPROBS,
        "max_tokens": max_tokens,
    }


_client_cache: dict[str, OpenAI] = {}


def _chat_messages(model: str, prompt_text: str) -> list[dict[str, str]]:
    system = (
        "You are an expert A/B test analyst. "
        "Always respond with ONLY a valid JSON object matching the requested schema."
    )
    # Gemma mlx server rejects role=system; fold into user turn.
    if "gemma" in model.lower():
        return [{"role": "user", "content": f"{system}\n\n{prompt_text}"}]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt_text},
    ]


def _get_openai_client(base_url: str) -> OpenAI:
    if base_url not in _client_cache:
        _client_cache[base_url] = OpenAI(base_url=base_url, api_key="x")
    return _client_cache[base_url]


def _logprobs_to_dict(choice) -> list[dict]:
    content = []
    lp = getattr(choice, "logprobs", None)
    if not lp or not lp.content:
        return content
    for tc in lp.content:
        content.append({
            "token": tc.token,
            "logprob": tc.logprob,
            "top_logprobs": [
                {"token": t.token, "logprob": t.logprob}
                for t in (tc.top_logprobs or [])
            ],
        })
    return content


def _logprob_fields(logprob_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "p_verdict_dist": logprob_info.get("p_verdict_dist"),
        "p_chosen": logprob_info.get("p_chosen"),
        "verdict_first_token": logprob_info.get("verdict_first_token"),
        "verdict_logprob_idx": logprob_info.get("verdict_logprob_idx"),
        "raw_logprobs": logprob_info.get("raw_logprobs"),
        "p_verdict_dist_censored": logprob_info.get("p_verdict_dist_censored"),
        "verdict_top_logprobs": logprob_info.get("verdict_top_logprobs"),
        "topk_floor_logprob": logprob_info.get("topk_floor_logprob"),
    }


def _parsed_verdict_raw(raw_response: str) -> str | None:
    try:
        parsed = _extract_json(raw_response)
    except (json.JSONDecodeError, ValueError):
        return None
    verdict = str(parsed.get("verdict", "")).strip().lower()
    return verdict or None


def _invalid_format_result(
    *,
    case_id: str,
    model: str,
    base_url: str,
    mode: str,
    raw_response: str,
    logprobs_content: list[dict],
    last_error: Exception | None,
    attempt: int,
    latency_s: float,
    usage: dict,
) -> dict[str, Any]:
    logprob_info = extract_verdict_logprobs(logprobs_content)
    parsed_raw = _parsed_verdict_raw(raw_response)
    return {
        "case_id": case_id,
        "model": model,
        "provider": "local_openai",
        "base_url": base_url,
        "mode": mode,
        "verdict": "invalid_format",
        "model_verdict": "invalid_format",
        "parsed_verdict_raw": parsed_raw,
        "mechanism": "none",
        "confidence": 0.0,
        "rationale": f"Invalid response after {attempt + 1} attempts: {last_error}",
        "invalid_format": True,
        "p_chosen": None,
        "raw_response": raw_response,
        "logprobs_content": logprobs_content,
        "latency_s": round(latency_s, 3),
        "retries": attempt,
        "usage": usage,
        **_logprob_fields(logprob_info),
    }


def run_case_local_openai(
    case_dir: Path,
    model: str,
    *,
    base_url: str = "http://127.0.0.1:8080/v1",
    mode: str = "free",
    retry_delay_s: float = 1.0,
) -> dict[str, Any]:
    """Run one case against a local OpenAI-compatible endpoint."""
    case_id = case_dir.name
    prompt_text = build_prompt(case_dir, mode=mode)
    client = _get_openai_client(base_url)

    last_error: Exception | None = None
    raw_response = ""
    logprobs_content: list[dict] = []
    last_latency_s = -1.0
    last_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

    for attempt in range(LOCAL_MAX_RETRIES + 1):
        t0 = time.monotonic()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=_chat_messages(model, prompt_text),
                **_api_params(mode),
            )
            last_latency_s = time.monotonic() - t0
            choice = response.choices[0]
            raw_response = choice.message.content or ""
            logprobs_content = _logprobs_to_dict(choice)

            if response.usage:
                last_usage = {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                }

            parsed = _extract_json(raw_response)
            validated = _validate_verdict(parsed)
            logprob_info = extract_verdict_logprobs(logprobs_content)
            logprob_info = attach_p_chosen(validated, logprob_info)

            return {
                "case_id": case_id,
                "model": model,
                "provider": "local_openai",
                "base_url": base_url,
                "mode": mode,
                **validated,
                "model_verdict": validated["verdict"],
                "raw_response": raw_response,
                "logprobs_content": logprobs_content,
                "latency_s": round(last_latency_s, 3),
                "retries": attempt,
                "usage": last_usage,
                **_logprob_fields(logprob_info),
            }

        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            if attempt < LOCAL_MAX_RETRIES:
                time.sleep(retry_delay_s)
            continue
        except Exception as e:
            last_error = e
            # Transient transport errors: retry without counting toward format failures.
            if attempt < LOCAL_MAX_RETRIES:
                time.sleep(retry_delay_s)
                continue
            return {
                "case_id": case_id,
                "model": model,
                "provider": "local_openai",
                "base_url": base_url,
                "mode": mode,
                "verdict": "api_error",
                "model_verdict": "api_error",
                "mechanism": "none",
                "confidence": 0.0,
                "rationale": f"API error after {attempt + 1} attempts: {last_error}",
                "raw_response": raw_response,
                "logprobs_content": logprobs_content,
                "latency_s": round(last_latency_s, 3),
                "retries": attempt,
                "usage": last_usage,
                **_logprob_fields(extract_verdict_logprobs(logprobs_content)),
            }

    return _invalid_format_result(
        case_id=case_id,
        model=model,
        base_url=base_url,
        mode=mode,
        raw_response=raw_response,
        logprobs_content=logprobs_content,
        last_error=last_error,
        attempt=LOCAL_MAX_RETRIES,
        latency_s=last_latency_s,
        usage=last_usage,
    )
