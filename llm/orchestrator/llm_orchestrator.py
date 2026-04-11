"""
llm_orchestrator.py
-------------------
The single entry point for the entire LLM narrative pipeline.

Wires together all LLM components in the correct order:
    1. Load static prompts  (system_prompt.md + feature_glossary.md)
    2. Build user prompt    (user_prompt_builder.py)
    3. Call Groq API        (llm_client.py)
    4. Parse response       (narrative.py)
    5. Return NarrativeResult to loan_assessment.py
"""

import os
from pathlib import Path
from functools import lru_cache

from llm.data.applicant_data import ApplicantFeatures
from llm.data.prediction_data import PredictionResult
from llm.prompts.user_prompt_builder import build_user_prompt
from llm.client.llm_client import generate_narrative
from llm.narrative.narrative import parse_narrative, NarrativeResult

_PROMPTS_DIR      = Path(__file__).resolve().parent.parent / "prompts"
_SYSTEM_PROMPT    = _PROMPTS_DIR / "system_prompt.md"
_FEATURE_GLOSSARY = _PROMPTS_DIR / "feature_glossary.md"


@lru_cache(maxsize=1)
def _load_system_prompt() -> str:
    if not _SYSTEM_PROMPT.exists():
        raise FileNotFoundError(
            f"system_prompt.md not found at {_SYSTEM_PROMPT}"
        )
    if not _FEATURE_GLOSSARY.exists():
        raise FileNotFoundError(
            f"feature_glossary.md not found at {_FEATURE_GLOSSARY}"
        )
    system_text   = _SYSTEM_PROMPT.read_text(encoding="utf-8").strip()
    glossary_text = _FEATURE_GLOSSARY.read_text(encoding="utf-8").strip()
    return (
        system_text
        + "\n\n---\n\n"
        + "## Feature Reference Glossary\n\n"
        + "Use the definitions below when explaining any risk driver "
        + "mentioned in the borrower data:\n\n"
        + glossary_text
    )


def run_narrative_pipeline(
    features:   ApplicantFeatures,
    prediction: PredictionResult,
) -> NarrativeResult:
    system_prompt = _load_system_prompt()
    user_prompt   = build_user_prompt(features, prediction)
    raw_narrative = generate_narrative(system_prompt, user_prompt)
    result        = parse_narrative(raw_narrative, verdict=prediction.verdict)
    return result
