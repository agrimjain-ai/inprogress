import re
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from backend.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class ExtractedMetadata:
    doc_type: str
    court_name: Optional[str]
    case_number: Optional[str]
    date: Optional[str]
    circular_number: Optional[str]
    notification_number: Optional[str]
    parties: List[str] = field(default_factory=list)
    section_refs: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Converts the dataclass instance to a standard dictionary."""
        return asdict(self)


# ==============================================================================
# PRE-COMPILED PATTERNS
# ==============================================================================

COURT_PATTERNS = [
    re.compile(r"(Supreme Court of India)", re.IGNORECASE),
    re.compile(r"(High Court of \w+)", re.IGNORECASE),
    re.compile(r"(GST Appellate Authority[^,\n]*)", re.IGNORECASE),
    re.compile(r"(Authority for Advance Ruling[^,\n]*)", re.IGNORECASE),
    re.compile(r"(Appellate Authority for Advance Ruling[^,\n]*)", re.IGNORECASE),
    re.compile(r"(Customs[,\s]*Excise and Service Tax Appellate Tribunal[^,\n]*)", re.IGNORECASE),
    re.compile(r"(National Anti-Profiteering Authority)", re.IGNORECASE),
]

CASE_NUMBER_PATTERNS = [
    re.compile(r"(W\.?P\.?\s*\(C\)?\s*No\.?\s*\d+/\d{4})", re.IGNORECASE),
    re.compile(r"(Civil Appeal\s+No\.?\s*\d+\s+of\s+\d{4})", re.IGNORECASE),
    re.compile(r"(Tax Appeal\s+No\.?\s*\d+\s+of\s+\d{4})", re.IGNORECASE),
    re.compile(r"(GST Appeal\s+No\.?\s*\d+/\d{4})", re.IGNORECASE),
    re.compile(r"(Advance Ruling\s+No\.?\s*[A-Z]{2,}/AAAR?/\d+/\d{4})", re.IGNORECASE),
]

DATE_PATTERNS = [
    re.compile(
        r"\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"),
    re.compile(
        r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b",
        re.IGNORECASE,
    ),
]

CIRCULAR_PATTERN = re.compile(
    r"Circular\s+No\.?\s*(\d+/\d+(?:/\d+)?(?:-[A-Z]+(?:/GST)?)?)", re.IGNORECASE
)

NOTIFICATION_PATTERN = re.compile(
    r"Notification\s+No\.?\s*(\d+/\d{4}(?:-[A-Z]+(?:\s*\(Rate\))?)?)", re.IGNORECASE
)

SECTION_PATTERN = re.compile(
    r"(?:Section|Rule|Article)\s+\d+[A-Z]?\s*(?:\(\d+\))*(?:\([a-z]+\))*", re.IGNORECASE
)

# PRODUCTION OPTIMIZATION: Broad '\s' inside the character class has been replaced with 
# explicit spaces/tabs to prevent matching across multiple paragraphs, mitigating 
# catastrophic backtracking when lookaheads fail to find 'vs/versus' in massive text files.
PARTY_PATTERN = re.compile(
    r"(?:M/s\.?|Commissioner|Union of India|State of \w+)\s+[A-Z][a-zA-Z \t&()\-\.]{3,50}(?=\s+(?:vs?\.?|versus|appellant|respondent))",
    re.IGNORECASE,
)


# ==============================================================================
# HELPER UTILITIES
# ==============================================================================

def _clean_whitespace(text: Optional[str]) -> Optional[str]:
    """
    Normalizes consecutive whitespace, tabs, non-breaking spaces, and
    newlines into single spaces. Removes leading and trailing spaces.
    """
    if not text:
        return None
    # Replaces non-breaking spaces (\xa0) and standard spacing/newlines
    cleaned = re.sub(r"[\s\xa0]+", " ", text)
    return cleaned.strip()


def _deduplicate_list(items: List[str]) -> List[str]:
    """
    Deduplicates a list of strings while preserving original order
    and filtering out empty strings.
    """
    seen = set()
    unique = []
    for item in items:
        cleaned = item.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique.append(cleaned)
    return unique


def _search_patterns(text: str, patterns: List[re.Pattern]) -> Optional[str]:
    """
    Searches a list of regex patterns and returns the first cleaned match.
    Defensively retrieves the first capture group if it exists; otherwise,
    falls back to the full match.
    """
    for pattern in patterns:
        try:
            match = pattern.search(text)
            if match:
                val = match.group(1) if match.groups() else match.group(0)
                return _clean_whitespace(val)
        except Exception as e:
            logger.warning("Regex pattern match failed defensively: %s", e)
    return None


# ==============================================================================
# EXTRACTION IMPLEMENTATION
# ==============================================================================

def extract_court_name(text: str) -> Optional[str]:
    return _search_patterns(text, COURT_PATTERNS)


def extract_case_number(text: str) -> Optional[str]:
    return _search_patterns(text, CASE_NUMBER_PATTERNS)


def extract_date(text: str) -> Optional[str]:
    return _search_patterns(text, DATE_PATTERNS)


def extract_circular_number(text: str) -> Optional[str]:
    try:
        match = CIRCULAR_PATTERN.search(text)
        return _clean_whitespace(match.group(1)) if match else None
    except Exception as e:
        logger.warning("Circular pattern processing encountered an error: %s", e)
        return None


def extract_notification_number(text: str) -> Optional[str]:
    try:
        match = NOTIFICATION_PATTERN.search(text)
        return _clean_whitespace(match.group(1)) if match else None
    except Exception as e:
        logger.warning("Notification pattern processing encountered an error: %s", e)
        return None


def extract_all_section_refs(text: str) -> List[str]:
    try:
        matches = SECTION_PATTERN.findall(text)
        cleaned_matches = [_clean_whitespace(m) for m in matches if m]
        return _deduplicate_list([m for m in cleaned_matches if m])
    except Exception as e:
        logger.error("Failed to extract section references: %s", e)
        return []


def extract_parties(text: str) -> List[str]:
    try:
        matches = PARTY_PATTERN.findall(text)
        cleaned_matches = [_clean_whitespace(m) for m in matches if m]
        # Cap at 4 parties to prevent metadata bloating
        return _deduplicate_list([m for m in cleaned_matches if m])[:4]
    except Exception as e:
        logger.error("Failed to extract parties: %s", e)
        return []


def extract_metadata(
    full_text: str,
    doc_type: str,
    chunks: List[Chunk],
    header_char_limit: int = 3000
) -> ExtractedMetadata:
    """
    Extracts structured metadata from the document text.
    
    Uses a configurable character range from the start of the document for 
    header-level fields (court, case number, date, circulars, notifications, parties) 
    since these reliably appear at the beginning of legal documents.
    
    Processes chunk inputs defensively to collect and deduplicate section references.
    """
    logger.info("Extracting metadata for doc_type: %s", doc_type)

    # Defensive input checks
    if not full_text:
        logger.warning("Empty or null full_text passed to extract_metadata.")
        full_text = ""

    header_text = full_text[:header_char_limit]

    # Collect section references from parsed chunks
    all_refs = []
    if chunks:
        for chunk in chunks:
            # Defensively check that the chunk object contains a valid text field
            if chunk and hasattr(chunk, 'text') and chunk.text:
                refs = extract_all_section_refs(chunk.text)
                all_refs.extend(refs)
    else:
        logger.info("No chunks provided; falling back to full text for section references.")
        all_refs = extract_all_section_refs(full_text)

    unique_refs = _deduplicate_list(all_refs)

    return ExtractedMetadata(
        doc_type=doc_type,
        court_name=extract_court_name(header_text),
        case_number=extract_case_number(header_text),
        date=extract_date(header_text),
        circular_number=extract_circular_number(header_text),
        notification_number=extract_notification_number(header_text),
        parties=extract_parties(header_text),
        section_refs=unique_refs
    )