"""
narrative.py
------------
Parses the raw LLM response string into a structured NarrativeResult
object with named sections.

The LLM returns markdown-formatted text with bold headers like:
    **Assessment Summary**
    **Key Strengths**
    etc.

This file splits that raw text into a clean dataclass so
loan_assessment.py can render each section individually in Streamlit
instead of dumping one raw block of text.

Handles all 3 verdict zones:
    APPROVE       → Assessment Summary, Key Strengths, Monitoring Notes
    REJECT        → Risk Assessment, Primary Risk Factors, Risk Summary
    MANUAL REVIEW → Assessment Overview, Risk Signals,
                    Mitigating Factors, Recommended Verification Steps

Usage:
    from llm.narrative.narrative import parse_narrative

    result = parse_narrative(raw_llm_response, verdict="REJECT")
    # result.sections → OrderedDict of {section_title: content}
    # result.raw      → original unmodified string (fallback)
"""

import re
from dataclasses import dataclass, field
from collections import OrderedDict


# ─────────────────────────────────────────────────────────────────────────────
# SECTION HEADERS PER VERDICT ZONE
# These must match exactly what system_prompt.md instructs the LLM to produce
# ─────────────────────────────────────────────────────────────────────────────

APPROVE_SECTIONS = [
    "Assessment Summary",
    "Key Strengths",
    "Monitoring Notes",
]

REJECT_SECTIONS = [
    "Risk Assessment",
    "Primary Risk Factors",
    "Risk Summary",
]

MANUAL_SECTIONS = [
    "Assessment Overview",
    "Risk Signals",
    "Mitigating Factors",
    "Recommended Verification Steps",
]

SECTION_MAP = {
    "APPROVE":       APPROVE_SECTIONS,
    "REJECT":        REJECT_SECTIONS,
    "MANUAL REVIEW": MANUAL_SECTIONS,
}


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NarrativeResult:
    """
    Parsed LLM narrative for one borrower assessment.

    Attributes:
        verdict   str          — APPROVE / REJECT / MANUAL REVIEW
        sections  OrderedDict  — {section_title: content_string}
                                  content is cleaned text, ready to display
        raw       str          — original LLM response (fallback if parse fails)
        parsed    bool         — True if sections were extracted successfully
                                  False if we fell back to raw display
    """
    verdict:  str                        = ""
    sections: OrderedDict                = field(default_factory=OrderedDict)
    raw:      str                        = ""
    parsed:   bool                       = False


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _clean_content(text: str) -> str:
    """
    Strip leading/trailing whitespace from a section's content.
    Preserves internal newlines so bullet points stay intact.
    """
    return text.strip()


def _extract_sections(
    raw: str,
    expected_sections: list[str],
) -> OrderedDict:
    """
    Split raw LLM text into named sections using bold markdown headers.

    The LLM is instructed to use **Section Title** format.
    This function finds each header and extracts the text that follows
    until the next header (or end of string).

    Args:
        raw               : full LLM response string
        expected_sections : list of section title strings to look for

    Returns:
        OrderedDict — {title: content} for every section found
        Empty OrderedDict if no sections matched
    """
    result = OrderedDict()

    # Build a regex pattern that matches any of the expected headers
    # Handles: **Title**, ## Title, ### Title (LLM sometimes varies)
    escaped = [re.escape(s) for s in expected_sections]
    header_pattern = r"(?:\*\*|#{1,3}\s*)(" + "|".join(escaped) + r")(?:\*\*)?"

    # Find all header positions
    matches = list(re.finditer(header_pattern, raw, re.IGNORECASE))

    if not matches:
        return result  # parse failed — caller will use raw fallback

    for i, match in enumerate(matches):
        title = match.group(1).strip()

        # Content starts after the header line
        content_start = match.end()

        # Content ends at the next header (or end of string)
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)

        content = _clean_content(raw[content_start:content_end])
        result[title] = content

    return result


def _fallback_result(raw: str, verdict: str) -> NarrativeResult:
    """
    Return a NarrativeResult with parsed=False when section
    extraction fails. The full raw text is preserved so Streamlit
    can still display something meaningful.
    """
    return NarrativeResult(
        verdict  = verdict,
        sections = OrderedDict({"Full Assessment": raw.strip()}),
        raw      = raw,
        parsed   = False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def parse_narrative(raw: str, verdict: str) -> NarrativeResult:
    """
    Parse the raw LLM response into a structured NarrativeResult.

    Args:
        raw     : str — raw string returned by llm_client.generate_narrative()
        verdict : str — "APPROVE" | "REJECT" | "MANUAL REVIEW"

    Returns:
        NarrativeResult with populated sections if parsing succeeded,
        or fallback NarrativeResult with parsed=False if it failed.
    """
    if not raw or not raw.strip():
        return _fallback_result("(No narrative generated)", verdict)

    expected_sections = SECTION_MAP.get(verdict.upper(), [])

    if not expected_sections:
        # Unknown verdict — return raw without attempting to parse
        return _fallback_result(raw, verdict)

    sections = _extract_sections(raw, expected_sections)

    if not sections:
        # LLM response didn't match expected format — use fallback
        return _fallback_result(raw, verdict)

    return NarrativeResult(
        verdict  = verdict,
        sections = sections,
        raw      = raw,
        parsed   = True,
    )