"""
llm_client.py
-------------
Handles all communication with the Groq API.
Takes the assembled system prompt and user prompt,
sends them to the LLM, and returns the raw response string.

This is the ONLY file in the project that touches the Groq SDK.
All other files are pure Python with no external API dependencies.

Setup required:
    pip install groq
    export GROQ_API_KEY="your_key_here"
    OR add to .env:  GROQ_API_KEY=your_key_here

Usage:
    from llm.client.llm_client import generate_narrative
    raw = generate_narrative(system_prompt, user_prompt)
"""

import os
from groq import Groq

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MODEL        = "llama-3.3-70b-versatile"   # best quality on free Groq tier
MAX_TOKENS   = 1024                         # enough for 3-section narrative
TEMPERATURE  = 0.3                          # low = consistent, professional tone


# ─────────────────────────────────────────────────────────────────────────────
# CLIENT — initialised once at module load
# ─────────────────────────────────────────────────────────────────────────────

def _get_client() -> Groq:
    """
    Create and return a Groq client.
    Reads GROQ_API_KEY from environment variables.
    Raises a clear error if the key is missing.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable not set.\n"
            "Add it to your .env file or run:\n"
            "  export GROQ_API_KEY='your_key_here'"
        )
    return Groq(api_key=api_key)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def generate_narrative(system_prompt: str, user_prompt: str) -> str:
    """
    Send system + user prompts to Groq and return the raw response string.

    Args:
        system_prompt : str — full system message (system_prompt.md + feature_glossary.md)
        user_prompt   : str — borrower-specific prompt from user_prompt_builder.py

    Returns:
        str — raw LLM response text (will be parsed by narrative.py)

    Raises:
        RuntimeError — wraps any Groq API error with a clean message
    """
    client = _get_client()

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return response.choices[0].message.content

    except Exception as e:
        raise RuntimeError(f"Groq API call failed: {e}") from e