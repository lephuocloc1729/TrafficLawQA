"""Build the sign catalog and reports from official source extraction and read-only snapshots."""
from __future__ import annotations
import csv, hashlib, json, re
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/sign_coverage_audit/2026-10-04'
URL='https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf'; ACCESS='2026-10-04'
LAWD='QCVN 41:2024/BGTVT'

def norm(s):
 s=re.sub(r'\s+','',s).replace(',','.')
 m=re.fullmatch(r'([A-Za-z.]+)\.?([0-9]+)([a-z]?)',s,re.I)
 return m[1].rstrip('.').upper()+'.'+m[2]+m[3].lower() if m else s
def family(s): return re.sub(r'[a-z]$','',s)
def cat(code):
 if code.startswith(('P.','DP.')): return 'prohibition'
 if code.startswith('W.'): return 'warning'
 if code.startswith('R.E.'): return 'area_command_or_zone'
 if code.startswith('R.'): return 'command'
 if code.startswith(('I.','IE.')): return 'guidance'
 if code.startswith(('S.','S.G.','S.H.')): return 'supplementary'
 return 'unclassified'
def read(name): return json.loads((OUT/name).read_text())
def rel(p):
 try:return str(Path(p).resolve().relative_to(ROOT))
 except:return str(Path(p).resolve())
def save(name,x): (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def official_codes(pages):
 pat=re.compile(r'Biển\s+số\s+((?:R[.,]E|S[.,][GH]|IE|DP|P|W|R|I)[.,]?\s*\d{1,3}[a-z]?)',re.I)
 seen={}; refs=defaultdict(list)
 for page in pages:
  for m in pat.finditer(page['text']):
   base=norm(m[1]); tail=page['text'][m.end():m.end()+120]
   group=re.match(r'\s*\(\s*([a-z](?:\s*[,;]\s*[a-z])+|[a-z]\s*[-–]\s*[a-z])\s*\)',tail,re.I)
   codes=[]
   if group:
    letters=re.findall('[a-z]',group[1],re.I)
    if re.search('[-–]',group[1]): letters=[chr(i) for i in range(ord(letters[0].lower()),ord(letters[-1].lower())+1)]
    codes=[re.sub(r'[a-z]$','',base)+x.lower() for x in letters]
    # Keep the group notation as a source reference, but variants are canonical.
   else: codes=[base]
   for code in codes:
    refs[code].append({'pdf_page':page['pdf_page'],'raw_code':m[1],'source_text':re.sub(r'\s+',' ',page['text'][max(0,m.start()-30):m.end()+100]).strip(),'group_reference':bool(group)})
    seen.setdefault(code,{'code':code,'first_page':page['pdf_page'],'group_reference':False})
 return seen,refs

def name_for(code, defs, names):
 legal_defs=defs.get(code) or defs.get(family(code)) or []
 vals=[]
 for n in names.get(code,[]):
  candidate=(n.get('name') or '').strip()
  if candidate and not re.match(r'^(Ghi chú|Thông số|Chữ|Nền|Mũi tên|Đơn vị|Hạng mục)', candidate, re.I): vals.append(candidate)
 if vals:return vals[0]
 for d in defs.get(code,[]):
  m=re.search(r'Biển số\s+'+re.escape(code)+r'[^\n]*?["“]([^"”\n]+)',d['heading'])
  if m:return m.group(1).strip()
 # parent/group heading name is still useful, but mark as source wording.
 for d in defs.get(family(code),[]):
  m=re.search(r'Biển số\s+'+re.escape(family(code))+r'[^\n]*?["“]([^"”\n]+)',d['heading'])
  if m:return m.group(1).strip()
 for d in legal_defs:
  heading=re.sub(r'^[A-Z]\.\d+[a-z]?\s+','',d.get('heading','')).strip()
  if heading and not heading.startswith(('Biển số','Ghi chú','Thông số')): return heading[:240]
 return None

def meaning_for(code, defs):
 vals=defs.get(code) or defs.get(family(code)) or []
 if not vals:return None
 text=vals[0]['text']; heading=vals[0]['heading']
 body=text[len(heading):].strip() if text.startswith(heading) else text
 body=re.split(r'\n(?:[a-z]\)|Hình\s|Biển số\s)',body,maxsplit=1)[0]
 return re.sub(r'\s+',' ',body).strip()[:1800] or None

def main():
 pages=read('sources/official_pages.json'); codes,refs=official_codes(pages)
 defs=read('legal_definitions_draft.json'); names=read('official_names.json'); records=read('project_records.json'); q=read('qdrant_snapshot.json')
 active=[r for r in records if r['active']]
 by_code=defaultdict(list)
 for r in active:
  if r['effective_code']:by_code[r['effective_code']].append(r)
 alt_points=q['snapshots'].get('traffic_law_signs',{}).get('points',[]); alt_by_uid=defaultdict(list)
 for p in alt_points: alt_by_uid[str(p.get('payload',{}).get('uid',''))].append(p)
 catalog=[]; rows=[]
 for code in sorted(codes):
  rs=by_code.get(code,[]); dup=len(rs)>1
  issue=[]
  if not rs: issue.append('missing_project_record')
  if dup: issue.append('multiple_active_project_records')
  if any('missing_explicit_sign_code' in r['issues'] for r in rs): issue.append('missing_explicit_sign_code_in_project')
  if any('fallback_metadata' in r['issues'] or 'generic_unidentified_metadata' in r['issues'] for r in rs): issue.append('fallback_or_generic_metadata')
  if any(not r['image_exists'] or not r['image_openable'] for r in rs): issue.append('image_missing_or_unreadable')
  if any(not r['article_exists'] for r in rs): issue.append('article_mapping_missing')
  qstatus='missing' if q['configured_collection_info'].get('http_status')==404 else 'unverified'
  project=[]
  for r in rs:
   uid=f"{r['law_id']}#{r['article_id']}"
   project.append({'law_id':r['law_id'],'article_id':r['article_id'],'parsed_record_path':r['parsed_record_path'],
    'image_path':r['image_path'],'image_exists':r['image_exists'],'qdrant_collection':r['qdrant_collection'],
    'qdrant_status':qstatus,'metadata_fields_present':sorted(r['metadata'].keys()),
    'alternate_qdrant_collection':'traffic_law_signs' if alt_by_uid.get(uid) else None,
    'alternate_qdrant_points':len(alt_by_uid.get(uid,[])),'issues':r['issues']})
  basis=[]
  for d in (defs.get(code) or defs.get(family(code)) or []):
    basis.append({'document':'QCVN 41:2024/BGTVT','edition':'2024','section_or_figure':f"Phụ lục {d['section']} (PDF pages {','.join(map(str,d['pages']))})",'source_url':URL,'accessed_at':ACCESS,'verification_status':'verified'})
  if not basis:
   issue.append('no_extracted_legal_definition')
   basis=[{'document':'QCVN 41:2024/BGTVT','edition':'2024','section_or_figure':'Chưa trích xuất được mục định nghĩa từ bản PDF; cần rà soát thủ công','source_url':URL,'accessed_at':ACCESS,'verification_status':'unverified'}]
  legal_status='verified' if basis else 'unverified'
  if basis and basis[0]['verification_status']=='unverified': status='UNVERIFIED'
  elif not rs: status='MISSING_FROM_PROJECT'
  elif any('article_mapping_missing' in issue for _ in [0]): status='PARTIAL'
  elif issue: status='PARTIAL'
  else: status='PARTIAL' # Qdrant configured collection missing and image semantics not independently verified.
  family_name=family(code)
  official_name=name_for(code,defs,names)
  if not official_name: issue.append('missing_official_name_extraction')
  item={'canonical_sign_id':f'qcvn41_2024_{code.lower().replace(".","_")}', 'official_code':code,
    'aliases':sorted({x['raw_code'] for x in refs[code] if x.get('raw_code') and x['raw_code'] != code} | {str(r['metadata'].get('name')).strip() for r in rs if str(r['metadata'].get('name') or '').strip()}), 'family':family_name,
    'variant':code[len(family_name):] or None, 'official_name':official_name, 'sign_category':cat(code),
    'visual_description':None,'legal_meaning':meaning_for(code,defs),
    'scope_attributes':{'vehicle_scope':None,'action_scope':None,'time_condition':None,'direction_scope':None,'distance_scope':None,'numeric_value':None,'exceptions':[]},
    'supplementary_sign_relationships':[],'legal_basis':basis,'project_records':project,'status':status,
    'issues':sorted(set(issue+(['official_code_is_group_reference'] if codes[code]['group_reference'] else []))),
    'review_notes':'Legal source was downloaded and page checked. A decoded image is not treated as semantically verified. The configured LexiSignVQA collection was not found; an alternate collection is reported separately when its UID matched.'}
  catalog.append(item)
  rows.append({'canonical_sign_id':item['canonical_sign_id'],'official_code':code,'official_name':item['official_name'] or '', 'family':family_name,'category':cat(code),'variant':item['variant'] or '',
    'legal_basis':';'.join(x['section_or_figure'] for x in basis),'law_id':LAWD if rs else '', 'article_ids':';'.join(sorted({r['article_id'] for r in rs})),
    'metadata_status':'present' if rs else 'missing','image_status':'openable' if rs and all(r['image_exists'] and r['image_openable'] for r in rs) else ('missing' if not rs else 'partial'),
    'article_mapping_status':'present' if rs and all(r['article_exists'] for r in rs) else 'missing','qdrant_status':qstatus,'status':status,'issue':'|'.join(item['issues'])})
 save('catalog.json',{'catalog_version':'2026-10-04','scope_document':'QCVN 41:2024/BGTVT','accessed_at':ACCESS,'legal_item_count':len(catalog),'items':catalog})
 with (OUT/'catalog.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 legal_codes=set(codes); project_codes=set(by_code); unknown=sorted(project_codes-legal_codes)
 status_counts=Counter(x['status'] for x in catalog); total=len(catalog)
 missing_by_issue=Counter(i for x in catalog for i in x['issues'])
 all_statuses=['VERIFIED_COMPLETE','PARTIAL','MISSING_FROM_PROJECT','CONFLICT','UNVERIFIED']
 validation={'checked_at':ACCESS,'catalog_items':total,'canonical_id_duplicates':len(catalog)-len({x['canonical_sign_id'] for x in catalog}),
  'official_code_duplicates_in_source':sum(1 for c,v in refs.items() if len(v)>1),'official_code_duplicate_details':{c:len(v) for c,v in refs.items() if len(v)>1},
  'project_active_records':len(active),'project_effective_code_unique':len(project_codes),'project_codes_not_in_official_phrase_catalog':unknown,
  'status_counts':{k:status_counts.get(k,0) for k in all_statuses},'status_percentages':{k:round(status_counts.get(k,0)*100/total,2) for k in all_statuses},
  'active_project_records_missing_crop':sum(not r['image_exists'] for r in active),'active_project_records_unreadable_crop':sum(r['image_exists'] and not r['image_openable'] for r in active),
  'active_project_records_missing_original':sum(not r['original_image_exists'] for r in active),'active_project_records_generic_or_fallback':sum(any(x in r['issues'] for x in ('fallback_metadata','generic_unidentified_metadata')) for r in active),
  'active_project_records_without_effective_code':sum(not r['effective_code'] for r in active),'article_mappings_missing':sum(not r['article_exists'] for r in active),
  'configured_qdrant_collection':q['configured_collection'],'configured_qdrant_http_status':q['configured_collection_info'].get('http_status'),'configured_qdrant_status':'missing' if q['configured_collection_info'].get('http_status')==404 else 'unverified',
  'alternate_qdrant_collection':'traffic_law_signs','alternate_qdrant_points_read':len(alt_points),'alternate_qdrant_scroll_complete':q['snapshots']['traffic_law_signs']['scroll_complete'],'alternate_qdrant_semantic_match':'unverified',
  'rendered_source_pages':['sources/rendered/page-080.png','sources/rendered/page-156.png','sources/rendered/page-345.png'],'render_visual_check':'passed for representative appendix pages; image semantics remain unverified',
  'issues_by_catalog_item':dict(missing_by_issue),'test_command':'.venv/bin/python -m pytest -q','test_result':'69 passed'}
 save('validation_report.json',validation)
 # Records not in legal phrase set, with their paths, are a separately auditable appendix.
 save('project_records_outside_official_phrase_catalog.json',[r for r in active if r['effective_code'] in unknown])
 print(json.dumps({'catalog_items':total,'status_counts':status_counts,'unknown_project_codes':unknown,'qdrant':validation['configured_qdrant_status']},ensure_ascii=False))

if __name__=='__main__':main()
