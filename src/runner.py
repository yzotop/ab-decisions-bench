"""
API runner.

Calls Anthropic API for one (case, model) pair and returns a parsed verdict.
NOT executed in Phase 1 — write-only until Phase 2.

Identical parameters for both models; only the model string differs.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

from .config import API_PARAMS, MAX_RETRIES, SGR_MAX_TOKENS
from .prompt import build_prompt

load_dotenv()

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set. Copy .env.example → .env and fill in your key.")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def _extract_json(text: str) -> dict:
    """
    Extract the first JSON object from the model response.
    Handles code-fenced (```json ... ```) and bare JSON.
    """
    # Try code-fence first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    # Try bare JSON object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"No JSON object found in response:\n{text[:300]}")


def _validate_verdict(data: dict) -> dict:
    """Light validation and normalisation of parsed response."""
    VALID_VERDICTS = {"ship", "no_ship", "investigate"}
    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in VALID_VERDICTS:
        raise ValueError(f"Invalid verdict '{verdict}'. Must be one of {VALID_VERDICTS}.")

    confidence = data.get("confidence")
    try:
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence {confidence} out of [0, 1]")
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid confidence: {e}") from e

    return {
        "verdict": verdict,
        "mechanism": str(data.get("mechanism", "none")).strip().lower(),
        "confidence": round(confidence, 4),
        "rationale": str(data.get("rationale", "")).strip(),
        "checks": data.get("checks") if isinstance(data.get("checks"), dict) else None,
    }


def run_case(
    case_dir: Path,
    model: str,
    *,
    mode: str = "free",
    retry_delay_s: float = 1.0,
) -> dict[str, Any]:
    """
    Run one case against one model.

    Returns:
        {
            "case_id": str,
            "model": str,
            "verdict": "ship" | "no_ship" | "investigate",
            "mechanism": str,
            "confidence": float,
            "rationale": str,
            "raw_response": str,
            "latency_s": float,
            "retries": int,
        }
    """
    case_id = case_dir.name
    prompt_text = build_prompt(case_dir, mode=mode)
    client = _get_client()
    params = dict(API_PARAMS)
    if mode == "sgr":
        params["max_tokens"] = SGR_MAX_TOKENS

    last_error: Exception | None = None
    raw_response = ""

    for attempt in range(MAX_RETRIES + 1):
        t0 = time.monotonic()
        try:
            message = client.messages.create(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                system=(
                    "You are an expert A/B test analyst. "
                    "Always respond with ONLY a valid JSON object matching the requested schema."
                ),
                **params,
            )
            latency_s = time.monotonic() - t0
            raw_response = message.content[0].text if message.content else ""
            usage = {
                "input_tokens": message.usage.input_tokens if message.usage else 0,
                "output_tokens": message.usage.output_tokens if message.usage else 0,
            }

            parsed = _extract_json(raw_response)
            validated = _validate_verdict(parsed)

            return {
                "case_id": case_id,
                "model": model,
                "mode": mode,
                **validated,
                "raw_response": raw_response,
                "latency_s": round(latency_s, 3),
                "retries": attempt,
                "usage": usage,
            }

        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(retry_delay_s)
            continue

    # All retries exhausted
    return {
        "case_id": case_id,
        "model": model,
        "mode": mode,
        "verdict": "parse_error",
        "mechanism": "none",
        "confidence": 0.0,
        "rationale": f"Failed to parse after {MAX_RETRIES + 1} attempts: {last_error}",
        "raw_response": raw_response,
        "latency_s": -1.0,
        "retries": MAX_RETRIES,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
