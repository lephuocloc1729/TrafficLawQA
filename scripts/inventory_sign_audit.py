"""Inventory every LawDB version and image without changing project input data."""
from __future__ import annotations
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.settings import settings
from src.core.lawdb.sign_metadata import build_sign_payload

OUT = ROOT / 'outputs/sign_coverage_audit/2026-10-04'
ACTIVE = Path(settings.lawdb_json_sign_parsing_path).resolve()

def rel(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())

def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def fingerprint(path):
    data = path.read_bytes()
    return {'path': rel(path), 'size_bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

def main():
    paths = sorted(set(p.resolve() for p in [
        *Path('data/raw/law_db').glob('*.json'), *Path('data/raw/law_db').glob('*.bak'),
        *Path('outputs').glob('*law*.json'), *Path('outputs/lawdb_manual_backups').glob('*.json')]))
    images = {}
    for folder in (Path(settings.lawdb_image_dir), Path(settings.lawdb_extracted_sign_dir)):
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.name.startswith('.'):
                continue
            f = fingerprint(path)
            try:
                with Image.open(path) as im:
                    f.update(format=im.format, width=im.width, height=im.height)
                    im.verify()
                with Image.open(path) as im:
                    im.load()
                f['openable'] = True
            except Exception as exc:
                f.update(openable=False, error=str(exc))
            images[rel(path)] = f
    save('image_inventory.json', list(images.values()))
    raw = json.loads(Path(settings.lawdb_json_path).read_text())
    raw_keys = {(l['id'], a['id']) for l in raw for a in l['articles']}
    records, articles, versions = [], [], []
    for path in paths:
        data = json.loads(path.read_text())
        summary = fingerprint(path)
        summary.update(active=path == ACTIVE, laws=[], records=0)
        for li, law in enumerate(data):
            ids = Counter(a.get('id') for a in law['articles'])
            summary['laws'].append({'law_id': law['id'], 'articles': len(law['articles']),
                'duplicate_article_ids': {k:v for k,v in ids.items() if v>1}})
            for ai, article in enumerate(law['articles']):
                ptr = f'/{li}/articles/{ai}'
                originals = article.get('images') or re.findall(r'<<IMAGE:\s*([^\s]+)\s*/IMAGE>>', article.get('text',''))
                originals = list(dict.fromkeys(originals))
                article_row = {'source_path': rel(path), 'json_pointer': ptr, 'active': path == ACTIVE,
                    'law_id':law['id'], 'article_id':article['id'], 'title': article.get('title'),
                    'has_text': bool(article.get('text','').strip()), 'image_names':originals,
                    'signs': article.get('signs',[]), 'detailed_signs_count':len(article.get('detailed_signs',[])),
                    'raw_article_exists':(law['id'],article['id']) in raw_keys}
                articles.append(article_row)
                for si, sign in enumerate(article.get('detailed_signs', [])):
                    crop = Path(settings.lawdb_extracted_sign_dir) / sign.get('image','')
                    origin_name = re.sub(r'_crop\d+(?=\.[^.]+$)', '', sign.get('image',''))
                    origin = Path(settings.lawdb_image_dir) / origin_name
                    payload = build_sign_payload(law['id'], article['id'], str(crop), sign)
                    issues = []
                    if not sign.get('sign_code'): issues.append('missing_explicit_sign_code')
                    if not payload['sign_code']: issues.append('no_effective_code')
                    if not sign.get('legal_conditions'): issues.append('missing_structured_legal_conditions')
                    if not crop.is_file(): issues.append('missing_crop')
                    elif not images.get(rel(crop),{}).get('openable'): issues.append('unreadable_crop')
                    if not origin.is_file(): issues.append('missing_inferred_original')
                    if origin_name not in originals: issues.append('crop_original_not_in_article_images')
                    if (law['id'],article['id']) not in raw_keys: issues.append('article_absent_from_raw')
                    if ids[article['id']] > 1: issues.append('ambiguous_duplicate_article_id')
                    fallback = bool(sign.get('__fallback') or 'Fallback metadata:' in sign.get('description',''))
                    generic = 'chưa xác định' in sign.get('name','').lower()
                    if fallback: issues.append('fallback_metadata')
                    if generic: issues.append('generic_unidentified_metadata')
                    record = {'record_id':f'{rel(path)}#{ptr}/detailed_signs/{si}',
                        'parsed_record_path': rel(path), 'json_pointer':f'{ptr}/detailed_signs/{si}',
                        'active':path == ACTIVE, 'law_id':law['id'], 'article_id':article['id'],
                        'article_title':article.get('title'), 'article_exists': (law['id'],article['id']) in raw_keys,
                        'metadata':sign, 'effective_payload':payload, 'effective_code':payload['sign_code'],
                        'image_path': rel(crop), 'image_exists':crop.is_file(),
                        'image_openable':images.get(rel(crop),{}).get('openable',False),
                        'original_image_path':rel(origin), 'original_image_exists':origin.is_file(),
                        'original_image_openable':images.get(rel(origin),{}).get('openable',False),
                        'image_identity_status':'unverified', 'image_identity_note':'Decode success is not semantic sign verification.',
                        'qdrant_collection':settings.collection_name, 'qdrant_status':'unverified',
                        'issues':issues}
                    records.append(record)
                    summary['records'] += 1
        versions.append(summary)
    code_groups = defaultdict(list)
    for rec in records:
        if rec['active'] and rec['effective_code']:
            code_groups[rec['effective_code']].append(rec['record_id'])
    snapshot = json.loads((OUT / 'qdrant_snapshot.json').read_text())
    configured_http = snapshot['configured_collection_info']['http_status']
    for rec in records:
        rec['qdrant_status'] = 'missing' if configured_http == 404 else 'unverified'
    save('project_records.json', records)
    save('article_inventory.json', articles)
    save('data_versions.json', versions)
    save('project_inventory_summary.json', {
        'active_path': rel(ACTIVE), 'records_all_versions':len(records),
        'active_records':sum(r['active'] for r in records),
        'active_issues':dict(Counter(i for r in records if r['active'] for i in r['issues'])),
        'duplicate_effective_codes':{k:v for k,v in code_groups.items() if len(v)>1},
        'images':len(images), 'unreadable_images':[p for p,i in images.items() if not i['openable']],
        'settings':{k:str(getattr(settings,k)) for k in ('lawdb_json_path','lawdb_json_preprocessing_path',
            'lawdb_json_sign_parsing_path','lawdb_image_dir','lawdb_extracted_sign_dir','collection_name')},
        'lawdb_symlink': {'path':'data/law_db','target':'raw/law_db','resolved':rel(Path('data/law_db'))},
        'hashes':[fingerprint(p) for p in [*paths,*Path('src/core/lawdb').glob('*.py'),
            Path('src/settings.py'),Path('src/deps/qdrant.py'),Path('src/core/symbolic/sign_normalizer.py'),
            Path('src/core/symbolic/traffic_rules.py'),Path('src/core/article_context.py')]]})
    print('Inventoried', len(records), 'records;', len(images), 'images')

if __name__ == '__main__':
    main()
