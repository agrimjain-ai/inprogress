import re
import logging
from dataclasses import dataclass
from typing import List
import tiktoken  # Production-grade tokenizer
import pysbd   # Robust sentence boundary disambiguation

# ==============================================================================
# PRODUCTION NOTE ON DEPENDENCIES:
# This module requires `tiktoken` (for precise token counting) and `pysbd` 
# (for robust sentence splitting that doesn't break on legal abbreviations like 
# "v.", "Ltd.", "No.", "Sec.", etc.). Install them via:
# pip install tiktoken pysbd
# ==============================================================================

# Attempt to import the document schemas from your parsing pipeline;
# falls back to local definition if run in a standalone environment.
try:
    from backend.ingestion.parser import ParsedDocument, PageData
except ImportError:
    @dataclass(frozen=True)
    class PageData:
        page_number: int
        text: str

    @dataclass
    class ParsedDocument:
        filename: str
        doc_type: str
        raw_text: str
        pages: List[PageData]
        total_pages: int

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: int
    text: str
    token_count: int
    page_number: int
    section_ref: str


# Pre-compile regular expressions globally to optimize execution speed.
# Removed case-insensitivity on nouns (like Section, Rule) to avoid matching generic text.
SECTION_PATTERNS = [
    re.compile(r"Section\s+\d+[A-Z]?\s*(?:\(\d+\))*(?:\([a-z]+\))*"),
    re.compile(r"Rule\s+\d+[A-Z]?\s*(?:\(\d+\))*(?:\([a-z]+\))*"),
    re.compile(r"Article\s+\d+[A-Z]?"),
    re.compile(r"Notification\s+No\.?\s*\d+/\d+(?:-[A-Z]+)?", re.IGNORECASE),
    re.compile(r"Circular\s+No\.?\s*\d+/\d+(?:/\d+)?(?:-[A-Z]+)?", re.IGNORECASE),
]

# Initialize the sentence segmenter globally
segmenter = pysbd.Segmenter(language="en", clean=False)

# Initialize the tiktoken encoding mapping globally (targeting OpenAI standard cl100k_base)
try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    logger.warning("Failed to load tiktoken encoding; falling back to rough character calculation. Error: %s", e)
    tokenizer = None


def count_tokens(text: str) -> int:
    """
    Accurately counts tokens using the target tokenizer.
    Falls back to a safe approximation if the tokenizer fails to load.
    """
    if tokenizer:
        return len(tokenizer.encode(text, disallowed_special=()))
    
    # Fallback heuristic: 1 token ≈ 3.5 characters (slightly padded for safety)
    return int(len(text) / 3.5) + 1


def extract_section_ref(text: str) -> str:
    """
    Finds the most prominent legal section reference in a chunk.
    Returns the first match found, or empty string if none.
    """
    for pattern in SECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0).strip()
    return ""


def split_into_sentences(text: str) -> List[str]:
    """
    Splits text into sentences while respecting common legal abbreviations
    (e.g., 'vs.', 'Ltd.', 'No.', 'Sec.') using PySBD.
    """
    if not text.strip():
        return []
    
    try:
        return [sentence.strip() for sentence in segmenter.segment(text) if sentence.strip()]
    except Exception as e:
        logger.error("PySBD sentence splitting failed, falling back to basic regex. Error: %s", e)
        # Fallback split on period + space + capital letter
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [p.strip() for p in parts if p.strip()]


def split_oversized_sentence(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    """
    Forcefully splits an exceptionally long sentence that exceeds max_tokens 
    into smaller, token-based overlapping sub-sentences.
    """
    if not tokenizer:
        char_max = max_tokens * 4
        char_overlap = overlap_tokens * 4
        parts = []
        start = 0
        while start < len(text):
            end = start + char_max
            parts.append(text[start:end])
            start += (char_max - char_overlap)
        return parts

    tokens = tokenizer.encode(text, disallowed_special=())
    parts = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        parts.append(tokenizer.decode(chunk_tokens))
        step = max_tokens - overlap_tokens
        if step <= 0:
            step = 1
        start += step
    return parts


def chunk_document(
    parsed_doc: ParsedDocument,
    max_tokens: int = 512,
    overlap_tokens: int = 50
) -> List[Chunk]:
    """
    Splits a ParsedDocument into overlapping chunks using a sentence-boundary aware algorithm.

    Strategy:
    1. Work page by page to preserve page number metadata.
    2. Split each page into sentences using PySBD to preserve semantic boundaries.
    3. Accumulate sentences until we hit the max_tokens limit.
    4. Automatically split and wrap any singular sentence that is larger than max_tokens.
    5. Slide the window by backtracking sentence index dynamically to achieve the desired overlap_tokens.
    6. Extract section references from each chunk for metadata mapping.
    """
    logger.info("Chunking document: %s", parsed_doc.filename)
    chunks: List[Chunk] = []
    chunk_id_counter = 1

    # Enforce safe parameter ranges
    if max_tokens <= overlap_tokens:
        logger.warning(
            "max_tokens (%d) is <= overlap_tokens (%d). Adjusting overlap to 10%% of limit.", 
            max_tokens, 
            overlap_tokens
        )
        overlap_tokens = max(1, max_tokens // 10)

    for page in parsed_doc.pages:
        sentences = split_into_sentences(page.text)
        if not sentences:
            continue

        # Pre-process sentences to resolve single sentence sizes & avoid redundant token counting
        sentence_data = []
        for s in sentences:
            tokens = count_tokens(s)
            if tokens > max_tokens:
                sub_sentences = split_oversized_sentence(s, max_tokens, overlap_tokens)
                for sub_s in sub_sentences:
                    sentence_data.append((sub_s, count_tokens(sub_s)))
            else:
                sentence_data.append((s, tokens))

        idx = 0
        n_sentences = len(sentence_data)

        while idx < n_sentences:
            current_sentences = []
            current_tokens = 0
            start_idx = idx

            # Accumulate sentences until we hit the token limit
            while idx < n_sentences:
                s_text, s_tokens = sentence_data[idx]
                if current_tokens + s_tokens <= max_tokens:
                    current_sentences.append(s_text)
                    current_tokens += s_tokens
                    idx += 1
                else:
                    # Safeguard: if a singular sentence somehow still overflows, append and move forward
                    if not current_sentences:
                        current_sentences.append(s_text)
                        current_tokens += s_tokens
                        idx += 1
                    break

            chunk_text = " ".join(current_sentences)
            section_ref = extract_section_ref(chunk_text)

            chunks.append(
                Chunk(
                    chunk_id=chunk_id_counter,
                    text=chunk_text,
                    token_count=current_tokens,
                    page_number=page.page_number,
                    section_ref=section_ref
                )
            )
            chunk_id_counter += 1

            # Exit the window if we reached the end of the page's sentences
            if idx >= n_sentences:
                break

            # Calculate backtrack overlap boundary using sentence sizes
            overlap_sum = 0
            backtrack_count = 0
            for back_idx in range(idx - 1, start_idx - 1, -1):
                _, s_tokens = sentence_data[back_idx]
                if overlap_sum + s_tokens <= overlap_tokens:
                    overlap_sum += s_tokens
                    backtrack_count += 1
                else:
                    # Ensure progress if a single sentence in the overlap area is too large
                    if backtrack_count == 0:
                        backtrack_count = 1
                    break

            # Backtrack the pointer for the next iteration window
            if backtrack_count > 0:
                idx = idx - backtrack_count

    return chunks