"""Publishable local file manifest; no credentials or external upload."""
from pathlib import Path
import hashlib,json
from collect_wikidata import save_json,write_jsonl,stamp
root=Path(__file__).resolve().parent
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
# Improve the old collector's overly narrow wording; records themselves are unchanged.
path=root/'uncollected_targets.jsonl'
rows=[json.loads(line) for line in path.open(encoding='utf-8') if line.strip()]
for row in rows:row['reason']='not collected by bounded seed plus one-hop enrichment; may be further-hop or missing; not invented'
write_jsonl(path,rows)
summary=json.loads((root/'collection_summary.json').read_text(encoding='utf-8'))
summary['files'][path.name]={'bytes':path.stat().st_size,'sha256':digest(path)}
save_json(root/'collection_summary.json',summary)
files=[]
for p in sorted(root.rglob('*')):
    if not p.is_file() or '__pycache__' in p.parts or p.name=='file_manifest.json':continue
    files.append({'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)})
manifest={'created_at_utc':stamp(),'status':'real_official_structured_data_collected',
          'data_license':'CC0-1.0','collector_code_license':'MIT','is_synthetic':False,
          'file_count':len(files),'total_bytes':sum(f['bytes'] for f in files),
          'excluded_from_manifest':['__pycache__','file_manifest.json'],
          'files':files}
save_json(root/'file_manifest.json',manifest)
print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2))
print(json.dumps({'largest_files':sorted(files,key=lambda x:x['bytes'],reverse=True)[:8]},indent=2))
