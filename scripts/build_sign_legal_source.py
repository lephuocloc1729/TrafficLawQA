"""Build a source-first sign register. Every extraction retains page evidence."""
from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path

OUT = Path('outputs/sign_coverage_audit/2026-10-04')
URL = 'https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf'
CODE = re.compile(r'(?<![\w.])((?:R[.,]E|S[.,][GH])[.,]\s*\d{1,2}|(?:DP|IE|P|W|R|I|S)[.,]?\s*\d{3})(?:\s*([a-z])(?=\W|$))?', re.I)

def norm(s):
    s = re.sub(r'\s+','',s).replace(',','.')
    m = re.fullmatch(r'([A-Za-z.]+)\.?([0-9]+)([a-z]?)',s,re.I)
    if not m: return s
    return m[1].rstrip('.').upper()+'.'+m[2]+m[3].lower()

def family(s): return re.sub(r'[a-z]$','',s)

def mentions(text):
    found = []
    for m in CODE.finditer(text):
        base = norm(m[1])
        code = norm(m[0])
        tail = text[m.end():]
        group = re.match(r'\s*\(\s*([a-z](?:\s*[,;]\s*[a-z])+|[a-z]\s*[-–]\s*[a-z])\s*\)',tail)
        if group:
            letters = re.findall(r'[a-z]',group[1])
            if re.search('[-–]',group[1]):
                letters = [chr(x) for x in range(ord(letters[0]),ord(letters[-1])+1)]
            for ch in letters:
                found.append({'code':base+ch,'raw':m[0]+group[0], 'start':m.start(),'end':m.end()+group.end(),'expansion':'explicit_group'})
        else:
            found.append({'code':code,'raw':m[0],'start':m.start(),'end':m.end(),'expansion':None})
            # Explicit shorthand e.g. I.407b,c. Do not expand unspecified ranges.
            suffix = re.match(r'(?:\s*,\s*[a-z](?=\W|$))+', tail) if re.search('[a-z]$',code) else None
            if suffix:
                for ch in re.findall('[a-z]',suffix[0]):
                    found.append({'code':family(code)+ch,'raw':m[0]+suffix[0],'start':m.start(),'end':m.end()+suffix.end(),'expansion':'explicit_suffix_list'})
    return found

def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    pages = json.loads((OUT/'sources/official_pages.json').read_text())
    occurrences, sections, figures = [], [], []
    current = None
    for page in pages:
        p, text = page['pdf_page'],page['text']
        for m in mentions(text):
            occurrences.append({**m,'pdf_page':p,'context':re.sub(r'\s+',' ',text[max(0,m['start']-100):m['end']+180])})
        for line in text.splitlines():
            line = line.strip()
            if re.fullmatch(r'\d+',line) or not line: continue
            header = re.match(r'^([B-FP]\.\d+[a-z]?)\.?\s+((?:Biển|Báo|Chỉ|Biểu|Tiêu).*)',line)
            # Appendix headings only within their actual pages, not references.
            if header and (80 <= p <= 157 or p >= 345):
                current = {'section':header[1], 'start_page':p,'lines':[],'pages':[]}
                sections.append(current)
            if current and (80 <= p <= 157 or p >= 345):
                current['lines'].append(line)
                if p not in current['pages']: current['pages'].append(p)
            if re.match(r'^(?:[a-z]\)\s*)?Hình\s+(?:[A-Z]\.\d+|\d+)',line):
                figures.append({'pdf_page':p,'caption':line,'codes':[m['code'] for m in mentions(line)]})
    definitions = defaultdict(list)
    for s in sections:
        text = '\n'.join(s['lines'])
        # Heading stops at a clause marker or descriptive opening; keep wrapped title.
        title_lines = []
        for i,line in enumerate(s['lines']):
            if i and (re.match(r'^(?:[a-z]\)|- |Để |Biển |Các |Ở |Khi |Trong |Trên |Theo |Tùy |Tại |Hình )',line) or len(title_lines)>=4): break
            title_lines.append(line)
        heading = ' '.join(title_lines)
        prefix = {'B':('P.','DP.'),'C':('W.',),'D':('R.',),'E':('I.',),'F':('S.',),'P':('IE.',)}[s['section'][0]]
        head_codes = [m['code'] for m in mentions(heading) if m['code'].startswith(prefix)]
        families = {family(c) for c in head_codes}
        all_codes = {m['code'] for m in mentions(text) if m['code'].startswith(prefix) and (not families or family(m['code']) in families)}
        for code in all_codes:
            definitions[code].append({'section':s['section'],'pages':s['pages'],'heading':heading,'text':text,'role':'design_specification' if s['section'].startswith('P.') else 'meaning_and_use'})
    # Technical appendix M gives exact per-variant names independent of LawDB.
    names = defaultdict(list)
    for page in pages:
        p,text = page['pdf_page'],page['text']
        for m in re.finditer(r'Biển\s+số\s+([^\n:]+)\s*:\s*([^\n]+)',text,re.I):
            codes = mentions(m[1])
            for code in codes:
                names[code['code']].append({'name':m[2].strip(), 'pdf_page':p,'source':'caption_colon'})
                if 214 <= p <= 338 and code['code'] not in definitions:
                    definitions[code['code']].append({'section':'M','pages':[p],'heading':m[0],'text':m[0],'role':'design_specification'})
        # Exact quotation tied to a code; name does not imply legal scope.
        for m in mentions(text):
            tail = text[m['end']:]
            q = re.match(r'\s*["“]([^"”\n]+(?:\n[^"”]+)?)["”]',tail)
            if q:
                names[m['code']].append({'name':re.sub(r'\s+',' ',q[1]).strip(),'pdf_page':p,'source':'quoted_name'})
    save('legal_sections.json', sections)
    save('official_code_occurrences.json', occurrences)
    save('official_figure_register.json', figures)
    save('legal_definitions_draft.json',dict(definitions))
    save('official_names.json',dict(names))
    print('Source:',len(sections),'sections;',len(definitions),'defined code candidates;',len(set(x['code'] for x in occurrences)),'all code candidates;',len(figures),'figure captions')
    print('Only referenced:',sorted(set(x['code'] for x in occurrences)-definitions.keys()))
    print('No extracted name:',[c for c in definitions if c not in names])

if __name__=='__main__': main()
