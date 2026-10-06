"""Apply the documented NUL-space cleanup to an existing source extraction."""
import json
import re
from pathlib import Path
p = Path('outputs/sign_coverage_audit/2026-10-04/sources/official_pages.json')
pages = json.loads(p.read_text())
for page in pages:
    raw = page.get('raw_text', page['text'])
    page['raw_text'] = raw
    page['text'] = '\n'.join(re.sub(r'[ \t]+', '', line).replace('\x00',' ') if '\x00' in line else line for line in raw.splitlines())
p.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding='utf-8')
p.with_name('official_text.txt').write_text('\n'.join(f'\n=== PDF PAGE {x["pdf_page"]} ===\n{x["text"]}' for x in pages), encoding='utf-8')
