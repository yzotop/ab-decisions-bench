"""
Benchmark configuration.

Both models run with IDENTICAL parameters — only the model string differs.
"""

from __future__ import annotations

MODELS = [
    "claude-opus-4-8",
    "claude-opus-4-7",
]

# Shared API call parameters
API_PARAMS = {
    "max_tokens": 512,       # compact JSON response; ~300 tokens typical
    "temperature": 0,        # deterministic for reproducibility
}

# SGR reasoning is longer; 512 truncates JSON
SGR_MAX_TOKENS: int = 1024

# Corpus path (relative to repo root)
CORPUS_DIR = "data/corpus"

# Output schema the model must return
VERDICT_SCHEMA = {
    "verdict": "ship | no_ship | investigate",
    "mechanism": "trap code from taxonomy or 'none' (e.g. guardrail_violation, practically_small, segment_conflict, long_term_reversal, not_significant)",
    "confidence": "float 0..1",
    "rationale": "one paragraph, <=120 words",
}

# Retry policy for JSON parse failures
MAX_RETRIES = 2

# False-confidence threshold: wrong_ship AND confidence >= this → false_confidence flag
FC_THRESHOLD: float = 0.7
