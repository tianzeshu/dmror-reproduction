"""Read cached official responses and export a real, explicitly partial snapshot."""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib
from datetime import datetime,timezone
from collect_wikidata import PROPERTIES,item_target,write_jsonl,save_json

root=Path(__file__).resolve().parent
output=root/'checkpoint';output.mkdir(exist_ok=True)
domains=defaultdict(set)
for selector_file in root.glob('select_*_query.json'):
 name=selector_file.stem.replace('_query','')
 raw=root/'raw'/(name+'.json')
 if not raw.exists():continue
 domain=name.split('_',1)[1].title()
 try:result=json.loads(raw.read_text(encoding='utf-8'))
 except ValueError:continue
 for row in result.get('results',{}).get('bindings',[]):
  qid=row['item']['value'].split('/')[-1];domains[qid].add(domain)
entities={};origins={};raw_files=[]
for path in sorted(list((root/'raw').glob('seed_entities_*.json'))+list((root/'raw').glob('related_entities_*.json'))):
 if path.name.endswith('.provenance.json'):continue
 meta_path=path.with_name(path.stem+'.provenance.json')
 if not meta_path.exists():continue
 try:
  payload=path.read_bytes();metadata=json.loads(meta_path.read_text(encoding='utf-8'));result=json.loads(payload)
 except (ValueError,OSError):continue
 if metadata.get('status')!=200 or result.get('error') or hashlib.sha256(payload).hexdigest()!=metadata.get('sha256'):continue
 raw_files.append({'path':str(path.relative_to(root)),'sha256':metadata['sha256'],'bytes':len(payload),'source_url':metadata['source_url'],'retrieved_at_utc':metadata['retrieved_at_utc']})
 for qid,entity in result.get('entities',{}).items():
  if entity.get('missing') is not None:continue
  entities[qid]=entity;origins[qid]=metadata
nodes=[];edges=[]
for qid,entity in entities.items():
 meta=origins[qid]
 nodes.append({'qid':qid,'wikidata_url':'https://www.wikidata.org/wiki/'+qid,'labels':{k:v['value'] for k,v in entity.get('labels',{}).items()},
  'descriptions':{k:v['value'] for k,v in entity.get('descriptions',{}).items()},'domain_candidates':sorted(domains[qid]),
  'lastrevid':entity.get('lastrevid'),'modified':entity.get('modified'),'source_response_url':meta['source_url'],
  'retrieved_at_utc':meta['retrieved_at_utc'],'raw_response_sha256':meta['sha256'],'license':'CC0-1.0','is_synthetic':False})
 for prop,claims in entity.get('claims',{}).items():
  if prop not in PROPERTIES:continue
  for claim in claims:
   target=item_target(claim)
   if not target or claim.get('rank')=='deprecated':continue
   edges.append({'subject_qid':qid,'property_id':prop,'predicate':PROPERTIES[prop],'object_qid':target,'target_record_collected':target in entities,
    'statement_id':claim.get('id'),'rank':claim.get('rank'),'qualifiers':claim.get('qualifiers',{}),'references':claim.get('references',[]),
    'source_url':'https://www.wikidata.org/wiki/'+qid,'source_response_url':meta['source_url'],'raw_response_sha256':meta['sha256'],
    'retrieved_at_utc':meta['retrieved_at_utc'],'license':'CC0-1.0','is_synthetic':False})
write_jsonl(output/'entities.jsonl',nodes);write_jsonl(output/'relations.jsonl',edges)
unresolved=Counter(e['object_qid'] for e in edges if not e['target_record_collected'])
write_jsonl(output/'uncollected_targets.jsonl',[{'qid':qid,'statement_count':count,'reason':'not yet downloaded in partial snapshot'} for qid,count in sorted(unresolved.items())])
summary={'status':'partial_real_collection','snapshot_at_utc':datetime.now(timezone.utc).isoformat(),'is_synthetic':False,'license':'CC0-1.0',
 'entity_records':len(nodes),'relation_statement_records':len(edges),'resolved_relation_statement_records':sum(e['target_record_collected'] for e in edges),
 'uncollected_target_qids':len(unresolved),'domain_candidate_entity_counts':dict(Counter(d for n in nodes for d in n['domain_candidates'])),
 'property_counts':dict(Counter(e['predicate'] for e in edges)),'raw_responses':raw_files,
 'limitations':['Partial download; does not satisfy manuscript Table 1 totals.','Wikidata community facts are real collected records, not verified supplier dependencies.',
  'No invented entity, relationship, risk outcome, resilience value or causal failure path.','Current data cannot prove historical information availability.']}
summary['files']={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in output.glob('*.jsonl')}
save_json(output/'collection_summary.json',summary)
print(json.dumps({k:v for k,v in summary.items() if k not in {'raw_responses','files'}},ensure_ascii=False,indent=2))
