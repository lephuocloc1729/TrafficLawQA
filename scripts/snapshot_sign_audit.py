"""Read-only snapshot of configured Qdrant and payloads; no ingest or mutations."""
import json
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.settings import settings

OUT = Path('outputs/sign_coverage_audit/2026-10-04')

def request(path, body=None):
    req = urllib.request.Request(settings.qdrant_uri.rstrip('/') + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            return {'http_status': response.status, 'body': json.load(response)}
    except urllib.error.HTTPError as exc:
        return {'http_status': exc.code, 'body': json.loads(exc.read())}
    except Exception as exc:
        return {'http_status': None, 'error': type(exc).__name__ + ': ' + str(exc)}

result = {'checked_at': datetime.now(timezone.utc).isoformat(),
          'configured_collection': settings.collection_name,
          'collections': request('/collections'),
          'configured_collection_info': request('/collections/' + settings.collection_name),
          'snapshots': {}}
names = [x['name'] for x in result['collections'].get('body', {}).get('result', {}).get('collections', [])]
for name in names:
    info = request('/collections/' + name)
    # Other collections may belong to another project. Only sign collection is relevant.
    entry = {'info': info, 'points': [], 'scroll_complete': False}
    if name in (settings.collection_name, 'traffic_law_signs'):
        offset = None
        while True:
            body = {'limit': 256, 'with_payload': True, 'with_vector': True}
            if offset is not None:
                body['offset'] = offset
            page = request('/collections/' + name + '/points/scroll', body)
            if page.get('http_status') != 200:
                entry['error'] = page
                break
            data = page['body']['result']
            for point in data['points']:
                vector = point.pop('vector', None)
                point['vector_present'] = bool(vector)
                point['vector_dimensions'] = len(vector) if isinstance(vector, list) else None
                entry['points'].append(point)
            offset = data.get('next_page_offset')
            if offset is None:
                entry['scroll_complete'] = True
                break
    result['snapshots'][name] = entry
(OUT / 'qdrant_snapshot.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'configured_http': result['configured_collection_info']['http_status'],
                  'collections': {k: {'points_read': len(v['points']), 'complete': v['scroll_complete']} for k,v in result['snapshots'].items()}}, indent=2))
