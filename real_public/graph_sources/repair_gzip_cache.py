"""Decode saved actual gzip responses while preserving immutable wire bodies."""
from pathlib import Path
import gzip,hashlib,json
root=Path(__file__).resolve().parent/'raw'
count=0
for path in root.rglob('*.json'):
 payload=path.read_bytes()
 if payload[:2]!=b'\x1f\x8b':continue
 decoded=gzip.decompress(payload)
 parsed=json.loads(decoded)
 wire_path=path.with_name(path.stem+'.original_response.gz')
 wire_path.write_bytes(payload)
 path.write_bytes(decoded)
 metadata_path=path.with_name(path.stem+'.provenance.json')
 if metadata_path.exists():
  metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
  metadata['wire_response']={'filename':wire_path.name,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'encoding':'gzip'}
  metadata['bytes']=len(decoded);metadata['sha256']=hashlib.sha256(decoded).hexdigest()
  metadata['decoded_from_original_gzip']=True
  if isinstance(parsed,dict) and parsed.get('error'):metadata['api_error']=parsed['error']
  metadata_path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
 count+=1
print(json.dumps({'saved_wire_bodies_preserved_and_decoded':count}))
