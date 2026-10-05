"""Independent raw-to-normalized integrity audit; exit 1 on a violated claim."""
from collections import Counter
import argparse, hashlib, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def rows(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def target(c):
    s=c.get('mainsnak',{})
    v=s.get('datavalue',{}).get('value')
    return v.get('id') if s.get('snaktype')=='value' and isinstance(v,dict) and v.get('entity-type')=='item' else None

def audit(root):
    errors=[];checks=Counter()
    raw={}; origins={}; raw_manifest=[]
    for folder in [root/'raw',root/'raw'/'taxonomy']:
        for path in sorted(folder.glob('*entities_*.json')):
            if '.attempt' in path.name or path.name.endswith('.provenance.json'):continue
            mp=path.with_name(path.stem+'.provenance.json')
            if not mp.exists():continue
            meta=json.loads(mp.read_text(encoding='utf-8'))
            obj=json.loads(path.read_text(encoding='utf-8'))
            if meta.get('status')!=200 or obj.get('error'):continue
            if digest(path)!=meta.get('sha256'):errors.append('raw_hash:'+path.name)
            raw_manifest.append({'path':str(path.relative_to(root)),'sha256':meta['sha256'],'bytes':path.stat().st_size,
                                 'source_url':meta['source_url'],'retrieved_at_utc':meta['retrieved_at_utc']})
            for q,e in obj.get('entities',{}).items():
                if e.get('missing') is not None:continue
                raw[q]=e;origins[q]=meta
    summary=json.loads((root/'normalized'/'summary.json').read_text(encoding='utf-8'))
    ns=list(rows(root/'normalized'/'nodes.jsonl')); ex=list(rows(root/'normalized'/'excluded_entities.jsonl'))
    ids={n['qid'] for n in ns}; exids={n['qid'] for n in ex}
    def check(ok,name):
        checks[name]+=1
        if not ok:errors.append(name)
    check(len(ids)==len(ns),'unique_node_ids');check(not ids&exids,'excluded_nodes_disjoint')
    for n in ns+ex:
        q=n['qid'];e=raw.get(q,{})
        check(bool(e) and e.get('id')==q,'actual_entity_id')
        check(n['labels']=={k:v['value'] for k,v in e.get('labels',{}).items()},'labels_exact_raw')
        check(n['is_synthetic'] is False and n['license']=='CC0-1.0','source_license_and_no_synthetic')
        check(n['knowledge_available_at'] is None and n['historical_prediction_eligible'] is False,'historical_unknown')
        check(n['risk_label'] is None,'risk_unknown')
        check(n['raw_response_sha256']==origins[q]['sha256'],'node_provenance')
        if q in ids:check(n['fictional_or_hypothetical'] is False,'no_explicit_fictional_node')
        else:check(n['fictional_or_hypothetical'] is True and n['entity_type']=='unclassified','excluded_fictional_evidence')
        for kind,paths in n['type_evidence'].items():
            for path in paths:
                current=q
                for step in path:
                    check(step['subject_qid']==current,'contiguous_type_path')
                    claims=raw.get(current,{}).get('claims',{}).get(step['property_id'],[])
                    actual=[c for c in claims if c.get('id')==step['statement_id'] and c.get('rank')!='deprecated']
                    check(len(actual)==1 and target(actual[0])==step['object_qid'],'actual_type_statement')
                    current=step['object_qid']
    facts=list(rows(root/'normalized'/'relations.jsonl'))
    statements=list(rows(root/'normalized'/'relation_statements.jsonl'))
    statement_map={r['statement_id']:r for r in statements}
    check(len(statement_map)==len(statements),'unique_statement_ids')
    for r in statements:
        check(r['subject_qid'] in ids and r['object_qid'] in ids,'closed_statement_endpoints')
        actual=[c for c in raw[r['subject_qid']]['claims'].get(r['property_id'],[]) if c.get('id')==r['statement_id']]
        check(len(actual)==1,'statement_exists_in_raw')
        if len(actual)==1:
            c=actual[0]
            check(target(c)==r['object_qid'] and c.get('rank')==r['rank'],'statement_target_rank')
            check(c.get('qualifiers',{})==r['qualifiers'] and c.get('references',[])==r['references'],'qualifiers_references_exact')
        check(r['is_supplier_dependency'] is False and r['is_causal_failure_edge'] is False,'no_invented_supply_or_causality')
        check(r['knowledge_available_at'] is None and r['statement_publication_time'] is None,'no_inferred_publication_time')
    keys=set()
    for r in facts:
        k=(r['subject_qid'],r['property_id'],r['object_qid'])
        check(k not in keys,'unique_graph_fact');keys.add(k)
        check(r['subject_qid'] in ids and r['object_qid'] in ids,'closed_fact_endpoints')
        check(all(i in statement_map for i in r['statement_ids']),'fact_statement_links')
    labels=list(rows(root/'normalized'/'risk_labels.jsonl'))
    check({l['qid'] for l in labels}==ids and len(labels)==len(ids),'label_entity_coverage')
    check(all(l['label'] is None and l['label_code']==-1 and l['gold_label'] is False for l in labels),'all_labels_unknown')
    check(summary['strict_type_counts']==dict({t:sum(n['entity_type']==t for n in ns) for t in summary['strict_type_counts']}),'exact_type_summary')
    check(summary['usable_real_entity_count']==len(ns) and summary['downloaded_entity_count']==len(ns)+len(ex),'exact_entity_summary')
    check(summary['unique_graph_facts']==len(facts) and summary['statement_records']==len(statements),'exact_relation_summary')
    check(summary['observed_risk_labels']==0 and summary['supplier_dependency_edges']==0,'no_gold_or_supplies_claim')
    for name,info in summary['files'].items():check(digest(root/'normalized'/name)==info['sha256'],'normalized_file_hash')
    return {'status':'passed' if not errors else 'failed','errors':errors,'checks':dict(checks),
            'usable_entities':len(ns),'excluded_fictional_entities':len(ex),'closed_facts':len(facts),
            'closed_statements':len(statements),'strict_type_counts':summary['strict_type_counts'],
            'observed_risk_labels':0,'supplier_dependency_edges':0,'raw_provenance_manifest':raw_manifest}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=ROOT);p.add_argument('--output',type=Path)
    a=p.parse_args();report=audit(a.data_root)
    destination=a.output or a.data_root/'normalized'/'integrity_report.json'
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in {'raw_provenance_manifest','checks'}},ensure_ascii=True,indent=2))
    sys.exit(0 if report['status']=='passed' else 1)
if __name__=='__main__':main()
