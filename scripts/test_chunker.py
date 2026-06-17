import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.ingestion.parser import parse_document
from backend.ingestion.chunker import chunk_document

parsed = parse_document("/app/test_docs/sample.pdf", "sample.pdf")
chunks = chunk_document(parsed)

# Basic stats
print(f"Total chunks     : {len(chunks)}")
print(f"Avg token count  : {sum(c.token_count for c in chunks) // len(chunks)}")
print(f"Max token count  : {max(c.token_count for c in chunks)}")
print(f"Min token count  : {min(c.token_count for c in chunks)}")

# Section ref extraction
refs = [c.section_ref for c in chunks if c.section_ref]
print(f"\nChunks with section refs : {len(refs)} / {len(chunks)}")
print(f"Sample refs found        : {refs[:5]}")

# Overlap check
print(f"\n--- Overlap Check (chunks 0 and 1) ---")
print(f"End of chunk 0   : ...{chunks[0].text[-120:]}")
print(f"Start of chunk 1 : {chunks[1].text[:120]}...")

# Spot check chunk
print(f"\n--- Chunk 0 ---")
print(f"Page     : {chunks[0].page_number}")
print(f"Tokens   : {chunks[0].token_count}")
print(f"Section  : {chunks[0].section_ref}")
print(f"Text     : {chunks[0].text[:400]}")

# Token boundary check — no chunk should exceed 512
violations = [c for c in chunks if c.token_count > 512]
print(f"\nChunks exceeding 512 tokens : {len(violations)}")
if violations:
    for v in violations[:3]:
        print(f"  chunk_id={v.chunk_id} tokens={v.token_count}")