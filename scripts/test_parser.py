import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.ingestion.parser import parse_document

result = parse_document("/app/test_docs/sample.pdf", "sample.pdf")

print(f"Filename     : {result.filename}")
print(f"Doc type     : {result.doc_type}")
print(f"Total pages  : {result.total_pages}")
print(f"Total chars  : {len(result.raw_text)}")
print(f"\nFirst 500 characters:")
print(result.raw_text[:500])
print(f"\nPage 1 preview:")
print(result.pages[0].text[:300] if result.pages else "No pages found")