"""
backend/graph/relationship_extractor.py

Extracts relationships between legal entities (Sections, Judgments,
Notifications, Circulars, Amendments) from raw chunk text.

Strategy: High-efficiency, safe regex matching first for deterministic legal-drafting phrases.
LLM-based extraction is used as a fallback or enhancement step in Phase 4.

Output format — each extracted relationship is a dict:
{
    "source_text": str,       # raw matched text for the source entity
    "source_type": str,       # NodeLabel value
    "source_act": str | None, # act name captured alongside the source Section,
                               # if present in text (e.g. AMENDED_BY only)
    "rel_type": str,          # RelType value
    "target_text": str,       # raw matched text for the target entity
    "target_type": str,       # NodeLabel value
    "target_act": str | None, # reserved for future patterns that capture an
                               # act name on the target side (always None today)
    "confidence": float,      # 0-1, regex matches default to 0.9
    "extraction_method": str, # "regex" | "llm"
    "doc_id": int,
    "chunk_index": int,
}
"""

import re
import logging
import warnings
from typing import List, Dict, Any, Tuple, Optional

from backend.graph.schema import NodeLabel, RelType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strict, Backtracking-Safe Pattern Definitions
# ---------------------------------------------------------------------------

# Captures Indian case formatting: "X v. Y", "X vs Y", "X v/s Y", "X versus Y" (case-insensitive)
# Bounds are tightly controlled to prevent runaway matching on raw text blocks.
CASE_NAME_RAW = r"\b[A-Z0-9][\w\.\&\s']{2,50}\s+(?:v\.|v/s|vs\.?|versus)\s+[A-Z0-9][\w\.\&\s']{2,50}\b"

# Captures nested subsections, e.g., "Section 16(2)(c)(ii)" or "Section 54(8)(a)"
# CORRECTED: [ivxIVX] replaces [i|v|x] to prevent literal pipe matches inside character classes.
SECTION_REF_RAW = r"\bSec(?:tion)?\s+\d+[A-Za-z]?(?:\(\d+\))?(?:\([a-z]\))?(?:\([ivxIVX]+\))?\b"

# Captures "Notification No. 12/2017-Central Tax" or similar formats (typically uses a slash)
NOTIFICATION_REF_RAW = r"\bNotification\s+No\.?\s*[0-9]+/[0-9]+[\w\-\s]*\b"

# Captures Finance Acts or Central/State Amendment Acts.
# CORRECTED: Matches standard Indian citation format "Act No. 12 of 2017" while preserving slash support [1].
ACT_AMENDMENT_REF_RAW = r"\b(?:Act\s+No\.?\s*\d+(?:\s+of\s+|\s*/\s*)\d{4}|Finance\s+Act,?\s*\d{4})\b"

# Captures an act name mentioned next to a Section reference, e.g. "the CGST
# Act, 2017" or "the Central Goods and Services Tax Act" (year optional —
# GST drafting very often omits it when the act is contextually obvious).
ACT_NAME_RAW = r"[\w\s]{1,50}Act(?:,?\s*\d{4})?"


# Pre-compile patterns. 
# Wildcard tokens are bounded [^.]{1,80} instead of lazy star patterns to prevent CPU locks.
RELATIONSHIP_PATTERNS: List[Tuple[str, str, str, re.Pattern]] = [
    (
        RelType.AMENDED_BY,
        NodeLabel.SECTION,
        NodeLabel.AMENDMENT,
        re.compile(
            rf"({SECTION_REF_RAW})(?:\s+of\s+the\s+({ACT_NAME_RAW}))?\s+"
            rf"(?:was\s+|stands\s+|has\s+been\s+)?(?:amended|substituted|modified)\s+by\s+"
            rf"({NOTIFICATION_REF_RAW}|{ACT_AMENDMENT_REF_RAW})",
            re.IGNORECASE,
        ),
    ),
    (
        RelType.OVERRULED_BY,
        NodeLabel.JUDGMENT,
        NodeLabel.JUDGMENT,
        re.compile(
            rf"({CASE_NAME_RAW})\s+(?:was\s+|is\s+|stands\s+)?overrul(?:ed|ing)\s+by\s+({CASE_NAME_RAW})",
            re.IGNORECASE,
        ),
    ),
    (
        RelType.CITED_BY,
        NodeLabel.JUDGMENT,
        NodeLabel.JUDGMENT,
        re.compile(
            rf"(?:as\s+held\s+in|relying\s+on|following\s+the\s+decision\s+in|cited\s+in)\s+({CASE_NAME_RAW})",
            re.IGNORECASE,
        ),
    ),
    (
        RelType.CROSS_REFERENCED,
        NodeLabel.SECTION,
        NodeLabel.SECTION,
        re.compile(
            rf"({SECTION_REF_RAW})\s+read\s+with\s+({SECTION_REF_RAW})",
            re.IGNORECASE,
        ),
    ),
    (
        RelType.AFFECTS,
        NodeLabel.NOTIFICATION,
        NodeLabel.SECTION,
        re.compile(
            rf"({NOTIFICATION_REF_RAW})[^.]{{1,80}}?(?:amends|affects|inserts|substitutes)[^.]{{1,80}}?({SECTION_REF_RAW})",
            re.IGNORECASE,
        ),
    ),
]


def clean_extracted_text(text: Optional[str]) -> Optional[str]:
    """
    Normalizes extracted entity text to ensure consistency in Neo4j.
    Removes trailing punctuation, unifies whitespaces, and standardizes casing.
    """
    if not text:
        return None
    
    # Replace multiple whitespaces/newlines with a single space
    cleaned = re.sub(r"\s+", " ", text).strip()
    
    # Strip common trailing noise from regex boundary matches
    cleaned = cleaned.rstrip(",.;:-")
    
    # Map abbreviations (e.g., "Sec 16" -> "Section 16")
    if cleaned.lower().startswith("sec "):
        cleaned = "Section" + cleaned[3:]
        
    return cleaned


def extract_relationships(
    text: str, doc_id: int, chunk_index: int
) -> List[Dict[str, Any]]:
    """
    Scans text using high-performance regex patterns.
    """
    results: List[Dict[str, Any]] = []

    # Safeguard against extraordinarily long chunks that could slow regex evaluation
    if not text or len(text) > 10000:
         logger.warning(f"Chunk size limit exceeded ({len(text)} chars) for doc_id {doc_id}. Skipping regex extraction.")
         return results

    for rel_type, source_type, target_type, pattern in RELATIONSHIP_PATTERNS:
        for match in pattern.finditer(text):
            groups = match.groups()
            source_act: Optional[str] = None
            target_act: Optional[str] = None

            if rel_type == RelType.CITED_BY:
                # Only one capture group: the cited judgment. The citing
                # judgment is implicit ("this document") and resolved
                # downstream by graph_builder.py using doc_id.
                source_text = None
                target_text = clean_extracted_text(groups[0])
            elif rel_type == RelType.AMENDED_BY:
                # Three capture groups: (section, act_name_optional, amendment_ref)
                source_text = clean_extracted_text(groups[0])
                source_act = clean_extracted_text(groups[1]) if groups[1] else None
                target_text = clean_extracted_text(groups[2])
            else:
                source_text = clean_extracted_text(groups[0])
                target_text = clean_extracted_text(groups[1])

            # Skip matches if clean parsing resulted in empty properties
            if rel_type != RelType.CITED_BY and not source_text:
                continue
            if not target_text:
                continue

            results.append(
                {
                    "source_text": source_text,
                    "source_type": source_type,
                    "source_act": source_act,
                    "rel_type": rel_type,
                    "target_text": target_text,
                    "target_type": target_type,
                    "target_act": target_act,
                    "confidence": 0.9,
                    "extraction_method": "regex",
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                }
            )

    return results


def extract_relationships_llm(
    text: str, doc_id: int, chunk_index: int
) -> List[Dict[str, Any]]:
    """
    Fallback placeholder for Phase 4.
    Throws a runtime warning to notify developers if fallback is active but unimplemented.
    """
    warnings.warn(
        "LLM relationship extraction fallback is active but not yet implemented (Phase 4 Task).",
        RuntimeWarning,
        stacklevel=2
    )
    logger.error(
        f"Skipped LLM fallback execution for doc_id {doc_id}, chunk_index {chunk_index}. "
        "LLM client is not yet initialized."
    )
    return []


def extract_all(
    text: str, doc_id: int, chunk_index: int, use_llm_fallback: bool = False
) -> List[Dict[str, Any]]:
    """
    Main orchestration entry point.
    Runs regex matches. If nothing is found and fallback is active,
    delegates processing to the LLM extraction framework.
    """
    results = extract_relationships(text, doc_id, chunk_index)

    if not results and use_llm_fallback:
        results = extract_relationships_llm(text, doc_id, chunk_index)

    return results