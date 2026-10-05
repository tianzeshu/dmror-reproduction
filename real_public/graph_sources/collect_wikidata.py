"""Acquire real CC0 Wikidata statements; never invent entities or relationships.

Use narrow official SPARQL selectors followed by batches of <=50 official API
entity records. A failed response is saved and reported as missing, not filled.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import gzip,hashlib,json,time,urllib.parse,urllib.request,urllib.error
from pathlib import Path

ROOT=Path(__file__).resolve().parent
UA='DMROR-RealData-Provenance/1.0 (research; https://github.com/tianzeshu/dmror-reproduction)'
PROPERTIES={'P31':'instance_of','P452':'industry','P17':'country','P159':'headquarters',
            'P355':'subsidiary','P127':'owned_by','P749':'parent_organization',
            'P1056':'product_or_material_produced','P176':'manufacturer','P206':'located_in_or_next_to_water',
            'P131':'located_in_administrative_entity','P276':'location','P495':'country_of_origin',
            'P527':'has_part','P361':'part_of','P138':'named_after','P137':'operator'}
API='https://www.wikidata.org/w/api.php'
SPARQL='https://query.wikidata.org/sparql'
SOURCES=[
 'https://www.wikidata.org/wiki/Wikidata:Data_access',
 'https://www.mediawiki.org/wiki/Wikimedia_APIs/Rate_limits',
 'https://www.mediawiki.org/wiki/Wikidata_query_service/User_Manual',
 'https://www.wikidata.org/wiki/Wikidata:REST_API',
]

def stamp():return datetime.now(timezone.utc).isoformat()
def sha(payload):return hashlib.sha256(payload).hexdigest()
def save_json(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

class Fetcher:
 def __init__(self,folder):
  self.folder=folder;folder.mkdir(parents=True,exist_ok=True);self.log=[]
 def fetch(self,name,url,timeout=35):
  path=self.folder/(name+'.json');meta_path=self.folder/(name+'.provenance.json')
  if path.exists() and meta_path.exists():
   metadata=json.loads(meta_path.read_text(encoding='utf-8'))
   if metadata['status']==200:
    cached=json.loads(path.read_text(encoding='utf-8'))
    if 'error' not in cached:
     self.log.append(metadata)
     return cached,metadata
  for attempt in range(3):
   start=stamp()
   try:
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':UA,'Accept':'application/json','Accept-Encoding':'gzip'}),timeout=timeout) as response:
     wire=response.read();status=response.status;headers=dict(response.headers)
     content_encoding=next((v for k,v in headers.items() if k.lower()=='content-encoding'),'')
     if str(content_encoding).lower()=='gzip' or wire[:2]==b'\x1f\x8b':
      wire_path=self.folder/(name+f'.attempt{attempt}.response.gz');wire_path.write_bytes(wire)
      payload=gzip.decompress(wire)
     else:
      payload=wire
   except urllib.error.HTTPError as error:
    payload=error.read();status=error.code;headers=dict(error.headers)
   except Exception as error:
    metadata={'source_url':url,'retrieved_at_utc':start,'status':None,'error':str(error),'request_name':name}
    self.log.append(metadata);save_json(meta_path,metadata)
    if attempt<2:
     time.sleep(5);continue
    return None,metadata
   path.write_bytes(payload)
   metadata={'source_url':url,'retrieved_at_utc':start,'status':status,'bytes':len(payload),'sha256':sha(payload),
             'request_name':name,'response_headers':{k:v for k,v in headers.items() if k.lower() in {'content-type','content-encoding','date','etag','last-modified','retry-after','x-database-lag'}}}
   save_json(meta_path,metadata);self.log.append(metadata)
   try:parsed=json.loads(payload)
   except ValueError:parsed=None
   if isinstance(parsed,dict) and parsed.get('error'):
    metadata['api_error']=parsed['error'];save_json(meta_path,metadata)
   attempt_path=self.folder/(name+f'.attempt{attempt}.json');attempt_path.write_bytes(payload)
   save_json(self.folder/(name+f'.attempt{attempt}.provenance.json'),metadata)
   retry=status in {429,503} or (isinstance(parsed,dict) and parsed.get('error',{}).get('code') in {'maxlag','ratelimited'})
   if retry and attempt<2:
    delay=max(5,min(60,int(headers.get('Retry-After','5'))))
    print(json.dumps({'request':name,'status':status,'backoff_seconds':delay}),flush=True);time.sleep(delay);continue
   return parsed if status==200 and isinstance(parsed,dict) and 'error' not in parsed else None,metadata
  return None,metadata

def action_url(**params):return API+'?'+urllib.parse.urlencode({'format':'json','maxlag':5,**params})
def choose_search(fetcher,term):
 name='search_'+term.replace(' ','_').replace('/','_')
 obj,meta=fetcher.fetch(name,action_url(action='wbsearchentities',search=term,language='en',limit=10))
 results=(obj or {}).get('search',[])
 exact=[x for x in results if x.get('label','').casefold()==term.casefold()]
 chosen=exact[0] if exact else None
 return {'term':term,'chosen_qid':chosen.get('id') if chosen else None,'results':[{'qid':x.get('id'),'label':x.get('label'),'description':x.get('description')} for x in results],
         'source_url':meta['source_url'],'raw_sha256':meta.get('sha256'),'status':meta['status']}

def qids(result,key='item'):
 return sorted({r[key]['value'].split('/')[-1] for r in (result or {}).get('results',{}).get('bindings',[]) if key in r},key=lambda x:int(x[1:]))

def batches(ids,size=50):
 for offset in range(0,len(ids),size):yield ids[offset:offset+size]

def get_entities(fetcher,ids,prefix,entities,origins):
 for index,group in enumerate(batches(ids)):
  obj,meta=fetcher.fetch(f'{prefix}_{index:04}',action_url(action='wbgetentities',ids='|'.join(group),props='labels|descriptions|aliases|claims|info',languages='en|zh',languagefallback=1),timeout=40)
  for qid,entity in (obj or {}).get('entities',{}).items():
   if entity.get('missing') is not None:continue
   entities[qid]=entity;origins[qid]=meta
  print(json.dumps({'stage':prefix,'batch':index,'requested':len(group),'collected_total':len(entities)}),flush=True)
  time.sleep(.5)

def item_target(statement):
 snak=statement.get('mainsnak',{})
 if snak.get('snaktype')!='value':return None
 value=snak.get('datavalue',{}).get('value')
 return value.get('id') if isinstance(value,dict) and value.get('entity-type')=='item' else None

def write_jsonl(path,values):
 with path.open('w',encoding='utf-8') as handle:
  for value in values:handle.write(json.dumps(value,ensure_ascii=False)+'\n')

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--limit',type=int,default=1500)
 parser.add_argument('--related-limit',type=int,default=5000)
 parser.add_argument('--output',type=Path,default=ROOT)
 args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 fetcher=Fetcher(args.output/'raw');selectors=[];domains=defaultdict(set);roles=defaultdict(set)
 terms=['automotive industry','automobile manufacturer','semiconductor industry','semiconductor manufacturer',
        'energy industry','oil and gas industry','electric power industry','silicon','copper','lithium','steel','aluminium']
 searches={term:choose_search(fetcher,term) for term in terms}
 save_json(args.output/'search_resolution.json',searches)
 domain_terms={'Auto':['automotive industry'],'Semi':['semiconductor industry'],'Energy':['energy industry','oil and gas industry','electric power industry']}
 class_terms={'Auto':['automobile manufacturer'],'Semi':['semiconductor manufacturer'],'Energy':[]}
 seed_ids=set()
 for domain,terms_for_domain in domain_terms.items():
  ids=[searches[t]['chosen_qid'] for t in terms_for_domain if searches[t]['chosen_qid']]
  classes=[searches[t]['chosen_qid'] for t in class_terms[domain] if searches[t]['chosen_qid']]
  patterns=[]
  if ids:patterns.append('{ VALUES ?industry { '+' '.join('wd:'+x for x in ids)+' } ?item wdt:P452 ?industry . }')
  if classes:patterns.append('{ VALUES ?class { '+' '.join('wd:'+x for x in classes)+' } ?item wdt:P31 ?class . }')
  if not patterns:continue
  query='SELECT DISTINCT ?item WHERE { '+' UNION '.join(patterns)+' } ORDER BY ?item LIMIT '+str(args.limit)
  name='select_'+domain.lower();save_json(args.output/(name+'_query.json'),{'query':query,'scope':'domain candidate with explicit industry/instance assertions; organization type must be checked','limit':args.limit})
  obj,meta=fetcher.fetch(name,SPARQL+'?'+urllib.parse.urlencode({'query':query,'format':'json'}),timeout=55)
  found=qids(obj)
  selectors.append({'domain':domain,'query':query,'source_url':meta['source_url'],'raw_sha256':meta.get('sha256'),'status':meta['status'],'count':len(found),'limit':args.limit,'possibly_truncated':len(found)==args.limit})
  for qid in found:domains[qid].add(domain);roles[qid].add('organization_candidate');seed_ids.add(qid)
  for qid in ids+classes:domains[qid].add(domain);roles[qid].add('industry_or_organization_class');seed_ids.add(qid)
  print(json.dumps({'domain':domain,'seed_count':len(found),'status':meta['status']}),flush=True)
  time.sleep(1)
 for term in ['silicon','copper','lithium','steel','aluminium']:
  qid=searches[term]['chosen_qid']
  if qid:roles[qid].add('material_seed');seed_ids.add(qid)
 save_json(args.output/'selectors.json',selectors)
 entities={};origins={};get_entities(fetcher,sorted(seed_ids,key=lambda x:int(x[1:])), 'seed_entities',entities,origins)
 related=set()
 for qid,entity in list(entities.items()):
  for prop,claims in entity.get('claims',{}).items():
   if prop not in PROPERTIES:continue
   for claim in claims:
    if claim.get('rank')=='deprecated':continue
    target=item_target(claim)
    if not target:continue
    related.add(target)
    if prop in {'P17','P159','P131','P276','P495'}:roles[target].add('geographic_object')
    if prop=='P452':roles[target].add('industry')
    if prop=='P1056':roles[target].add('product_or_material')
    if prop in {'P127','P355','P749','P176','P137'}:roles[target].add('organization_related')
    domains[target].update(domains[qid])
 related=sorted(related-set(entities),key=lambda x:int(x[1:]))
 requested_related=related[:args.related_limit]
 get_entities(fetcher,requested_related,'related_entities',entities,origins)
 node_records=[];relation_records=[];statement_records=[];unfetched_targets=Counter()
 for qid in sorted(entities,key=lambda x:int(x[1:])):
  entity=entities[qid];meta=origins[qid]
  labels={lang:item['value'] for lang,item in entity.get('labels',{}).items()}
  description={lang:item['value'] for lang,item in entity.get('descriptions',{}).items()}
  node_records.append({'qid':qid,'wikidata_url':'https://www.wikidata.org/wiki/'+qid,'labels':labels,'descriptions':description,
   'domain_candidates':sorted(domains[qid]),'roles_from_observed_claims':sorted(roles[qid]),'lastrevid':entity.get('lastrevid'),'modified':entity.get('modified'),
   'retrieved_at_utc':meta['retrieved_at_utc'],'raw_response_sha256':meta.get('sha256'),'source_response_url':meta['source_url'],'license':'CC0-1.0','is_synthetic':False})
  for prop,claims in entity.get('claims',{}).items():
   for claim in claims:
    statement_records.append({'subject_qid':qid,'property_id':prop,'statement':claim,'source_response_sha256':meta.get('sha256'),'retrieved_at_utc':meta['retrieved_at_utc']})
    if prop not in PROPERTIES or claim.get('rank')=='deprecated':continue
    target=item_target(claim)
    if not target:continue
    known=target in entities
    if not known:unfetched_targets[target]+=1
    relation_records.append({'subject_qid':qid,'property_id':prop,'predicate':PROPERTIES[prop],'object_qid':target,'target_record_collected':known,
     'statement_id':claim.get('id'),'rank':claim.get('rank'),'qualifiers':claim.get('qualifiers',{}),'references':claim.get('references',[]),
     'source_url':'https://www.wikidata.org/wiki/'+qid,'source_response_url':meta['source_url'],'raw_response_sha256':meta.get('sha256'),
     'retrieved_at_utc':meta['retrieved_at_utc'],'license':'CC0-1.0','is_synthetic':False})
 write_jsonl(args.output/'entities.jsonl',node_records);write_jsonl(args.output/'relations.jsonl',relation_records);write_jsonl(args.output/'statements.jsonl',statement_records)
 write_jsonl(args.output/'uncollected_targets.jsonl',[{'qid':qid,'referencing_statements':count,'reason':'not collected by bounded seed plus one-hop enrichment; may be further-hop or missing; not invented'} for qid,count in sorted(unfetched_targets.items())])
 summary={'status':'collected','retrieval_finished_at_utc':stamp(),'license':'CC0-1.0','official_documentation':SOURCES,'is_synthetic':False,
  'entity_records':len(node_records),'relation_statement_records':len(relation_records),'all_statement_records':len(statement_records),
  'resolved_relation_statement_records':sum(r['target_record_collected'] for r in relation_records),'uncollected_target_qids':len(unfetched_targets),
  'seed_qids':len(seed_ids),'related_requested':len(requested_related),'related_candidates':len(related),'selectors':selectors,
  'property_counts':dict(Counter(r['predicate'] for r in relation_records)),
  'role_counts':dict(Counter(role for record in node_records for role in record['roles_from_observed_claims'])),
  'domain_entity_counts':dict(Counter(domain for record in node_records for domain in record['domain_candidates'])),
  'failed_requests':[x for x in fetcher.log if x.get('status')!=200 or x.get('api_error')],
  'limitations':['Wikidata is a real community knowledge base, not a gold-standard supplier dependency/risk dataset.',
    'Current statements and modified times are not historical knowledge-availability dates.',
    'Only preserved non-deprecated item-valued statements become graph relations; no random relation or inferred supplier link is added.',
    'Industry query results are candidates; roles retain their claim-derived meanings instead of forcing all targets into the five paper types.',
    'Missing targets, corporate risk labels, resilience capacities, future failure events and causal paths remain missing.']}
 summary['files']={p.name:{'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in args.output.glob('*.jsonl')}
 save_json(args.output/'collection_summary.json',summary)
 save_json(args.output/'request_manifest.json',fetcher.log)
 print(json.dumps(summary,ensure_ascii=True),flush=True)

if __name__=='__main__':main()
