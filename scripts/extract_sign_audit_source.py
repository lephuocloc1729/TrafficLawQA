"""Extract auditable page text from the official PDF; never touches LawDB."""
import json
import re
from pathlib import Path
from pypdf import PdfReader

OUT = Path('outputs/sign_coverage_audit/2026-10-04/sources')
pdf = OUT / '51-bgtvt-kem.pdf'
pages = []
for i, p in enumerate(PdfReader(pdf).pages):
    raw = p.extract_text()
    # This PDF encodes inter-word spaces as NUL; extraction introduces spaces
    # inside words. Keep raw text for checking every normalization decision.
    cleaned = '\n'.join(re.sub(r'[ \t]+', '', line).replace('\x00', ' ') if '\x00' in line else line for line in raw.splitlines())
    pages.append({'pdf_page': i + 1, 'text': cleaned, 'raw_text': raw})
(OUT / 'official_pages.json').write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding='utf-8')
(OUT / 'official_text.txt').write_text('\n'.join(f'\n=== PDF PAGE {p["pdf_page"]} ===\n{p["text"]}' for p in pages), encoding='utf-8')
print(f'Extracted {len(pages)} pages')
