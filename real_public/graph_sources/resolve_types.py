"""Resolve a conservative five-type mapping from actual P31/P279 facts."""
from pathlib import Path
import json,urllib.parse
from collect_wikidata import Fetcher,action_url,save_json,SPARQL,item_target

root=Path(__file__).resolve().parent
fetcher=Fetcher(root/'raw'/'taxonomy')
terms=['business enterprise','company','manufactured good','chemical substance','chemical element','material',
       'industry','economic sector','geographic region','country','city','administrative territorial entity','human settlement','fictional entity']
resolution={}
for term in terms:
 obj,meta=fetcher.fetch('root_search_'+term.replace(' ','_'),action_url(action='wbsearchentities',search=term,language='en',limit=10))
 found=[row for row in (obj or {}).get('search',[]) if row.get('label','').casefold()==term.casefold()]
 resolution[term]={'exact_matches':[{'qid':x['id'],'label':x.get('label'),'description':x.get('description')} for x in found],
                   'source_url':meta['source_url'],'raw_sha256':meta.get('sha256')}
save_json(root/'type_root_searches.json',resolution)
print(json.dumps(resolution,ensure_ascii=True,indent=2))
