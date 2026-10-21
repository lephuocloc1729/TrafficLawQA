"""Deterministic validation for the generated coverage artifacts."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/sign_coverage_audit/2026-10-04'
errors=[]
warnings=[]

def load(n):
    return json.loads((OUT/n).read_text())

def main():
    c=load('catalog.json'); rec=load('project_records.json'); q=load('qdrant_snapshot.json')
    items=c['items']; ids=[x['canonical_sign_id'] for x in items]
    if len(ids)!=len(set(ids)): errors.append('duplicate canonical_sign_id')
    if len(items)!=c['legal_item_count']: errors.append('legal_item_count mismatch')
    if any(not x.get('legal_basis') for x in items): errors.append('catalog item without provenance')
    valid={'VERIFIED_COMPLETE','PARTIAL','MISSING_FROM_PROJECT','CONFLICT','UNVERIFIED'}
    if any(x['status'] not in valid for x in items): errors.append('invalid status')
    rows=list(csv.DictReader((OUT/'catalog.csv').open(encoding='utf-8-sig')))
    if len(rows)!=len(items): errors.append('CSV row count mismatch')
    for r in rec:
        if r['image_exists'] and not r['image_openable']:
            errors.append('unreadable crop '+r['record_id'])
        if r['article_exists'] is False and r['active']:
            warnings.append('missing article '+r['record_id'])
    if q['configured_collection_info'].get('http_status')==404:
        warnings.append('configured Qdrant collection is missing (expected unresolved index)')
    result={'errors':errors,'warnings':warnings,'catalog_items':len(items),'csv_rows':len(rows),
        'canonical_ids_unique':len(ids)==len(set(ids)),
        'all_items_have_provenance':all(bool(x.get('legal_basis')) for x in items),
        'configured_qdrant_http_status':q['configured_collection_info'].get('http_status'),'pass':not errors}
    (OUT/'validator_run.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
    if errors: sys.exit(1)

if __name__=='__main__': main()
