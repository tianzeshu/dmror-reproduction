"""Conservative, evidence-linked five-type export of the collected CC0 snapshot.

Only official entity responses supply nodes or statements. P31/P279 paths supply
type evidence; a role in a relationship is never itself a type assertion.
Taxonomy expansion is bounded and any unresolved classification stays unknown.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict, deque
import hashlib, json, time
from pathlib import Path
from collect_wikidata import Fetcher, action_url, batches, item_target, save_json, stamp, write_jsonl

ROOT = Path(__file__).resolve().parent
# These meanings were resolved against official wbsearchentities responses.
# Q268592 is the group-of-firms sense of "industry", not the unrelated motto.
TYPE_ROOTS = {
    'firm': ['Q783794'],
    'product': ['Q1836700'],
    'material': ['Q214609', 'Q79529', 'Q11344'],
    'industry': ['Q8148', 'Q268592', 'Q3958441'],
    'region': ['Q6256', 'Q515', 'Q56061', 'Q486972'],
    'fictional': ['Q14897293'],
}

def sha(data): return hashlib.sha256(data).hexdigest()
def load_jsonl(path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip(): yield json.loads(line)

def load_official_entities(folder, prefixes):
    entities, origins, files = {}, {}, []
    for prefix in prefixes:
        for path in sorted(folder.glob(prefix + '*.json')):
            if '.attempt' in path.name or path.name.endswith('.provenance.json'): continue
            meta_path = path.with_name(path.stem + '.provenance.json')
            if not meta_path.exists(): continue
            try:
                payload = path.read_bytes()
                meta = json.loads(meta_path.read_text(encoding='utf-8'))
                result = json.loads(payload)
            except (ValueError, OSError): continue
            if meta.get('status') != 200 or result.get('error') or sha(payload) != meta.get('sha256'): continue
            files.append({'path': str(path.relative_to(ROOT)), **meta})
            for qid, entity in result.get('entities', {}).items():
                if entity.get('missing') is not None or not qid.startswith('Q'): continue
                entities[qid], origins[qid] = entity, meta
    return entities, origins, files

def claims_to_qids(entity, prop):
    return [(item_target(c), c) for c in entity.get('claims', {}).get(prop, [])
            if c.get('rank') != 'deprecated' and item_target(c)]

def collect_taxonomy(base_entities, depth_limit, entity_limit):
    """Follow actual P279 arcs; stop at known roots or the explicit budget."""
    folder = ROOT / 'raw' / 'taxonomy'
    fetcher = Fetcher(folder)
    tax, origins, _ = load_official_entities(folder, ['class_entities_'])
    cached_at_start = len(tax)
    combined = {**base_entities, **tax}
    roots = {q for ids in TYPE_ROOTS.values() for q in ids}
    frontier = roots | {q for e in base_entities.values() for q, _ in claims_to_qids(e, 'P31')}
    # A material/product category may have a P279 path even though it is a class.
    frontier |= {q for e in base_entities.values() for q, _ in claims_to_qids(e, 'P279')}
    seen, missing, requested = set(), set(), set()
    for depth in range(depth_limit + 1):
        todo = sorted(frontier - seen, key=lambda q: int(q[1:]))
        seen.update(todo)
        need = [q for q in todo if q not in combined]
        available = max(0, entity_limit - cached_at_start - len(requested))
        allowed, omitted = need[:available], need[available:]
        missing.update(omitted)
        for group in batches(allowed):
            # Names depend on QIDs, so rerunning after additional collection reuses exact responses.
            key = sha('|'.join(group).encode())[:16]
            obj, meta = fetcher.fetch('class_entities_' + key,
                action_url(action='wbgetentities', ids='|'.join(group),
                           props='labels|descriptions|claims|info', languages='en|zh', languagefallback=1), timeout=45)
            requested.update(group)
            returned = set()
            for qid, entity in (obj or {}).get('entities', {}).items():
                if entity.get('missing') is not None: continue
                returned.add(qid); combined[qid] = entity; tax[qid] = entity; origins[qid] = meta
            missing.update(set(group) - returned)
            print(json.dumps({'stage':'taxonomy','depth':depth,'requested_this_run':len(requested),
                              'taxonomy_entities_cached':len(tax),'missing':len(missing)}), flush=True)
            time.sleep(.5)
        next_frontier = set()
        for qid in todo:
            if qid in roots: continue
            for target, _ in claims_to_qids(combined.get(qid, {}), 'P279'): next_frontier.add(target)
        if depth == depth_limit:
            missing.update(next_frontier - seen)
        frontier = next_frontier
        if not frontier - seen: break
    save_json(ROOT/'taxonomy_collection_summary.json', {
        'retrieved_at_utc':stamp(),'root_qids':TYPE_ROOTS,'depth_limit':depth_limit,
        'total_taxonomy_entity_budget':entity_limit,'taxonomy_entity_records':len(tax),
        'unexpanded_or_missing_qids':sorted(missing),
        'request_manifest':fetcher.log,'is_synthetic':False,'license':'CC0-1.0'})
    return combined, origins, missing

def find_path(start, targets, entities, max_depth=12):
    """Return one real subclass proof, not an inferred or generated taxonomy arc."""
    queue = deque([(start, [])]); seen = {start}
    while queue:
        qid, proof = queue.popleft()
        if qid in targets: return proof
        if len(proof) >= max_depth: continue
        for target, statement in claims_to_qids(entities.get(qid, {}), 'P279'):
            if target in seen: continue
            seen.add(target)
            queue.append((target, proof + [{'subject_qid':qid,'property_id':'P279','object_qid':target,
                                           'statement_id':statement.get('id')}]))
    return None

def classify(qid, entity, taxonomy):
    matches = defaultdict(list)
    for class_qid, claim in claims_to_qids(entity, 'P31'):
        for kind, roots in TYPE_ROOTS.items():
            path = find_path(class_qid, set(roots), taxonomy)
            if path is not None:
                matches[kind].append([{'subject_qid':qid,'property_id':'P31','object_qid':class_qid,
                                      'statement_id':claim.get('id')}] + path)
    # An actual product/material/industry category can be represented as such;
    # company/geography classes themselves are not counted as companies/regions.
    has_subclasses = bool(claims_to_qids(entity, 'P279'))
    if has_subclasses:
        for kind in ['product','material','industry','fictional']:
            path = find_path(qid, set(TYPE_ROOTS[kind]), taxonomy)
            if path is not None: matches[kind].append(path)
    fictional = 'fictional' in matches
    kinds = sorted(k for k in matches if k != 'fictional')
    if fictional: final, reason = 'unclassified', 'fictional_or_hypothetical_class_evidence'
    elif len(kinds) == 1: final, reason = kinds[0], 'official_instance_or_subclass_evidence'
    elif len(kinds) > 1: final, reason = 'unclassified', 'ambiguous_overlapping_type_evidence'
    else: final, reason = 'unclassified', 'no_supported_five_type_path_within_taxonomy_budget'
    return {'entity_type':final,'type_candidates':kinds,'type_reason':reason,
            'entity_form':'class_or_category' if has_subclasses else 'instance_or_unspecified',
            'type_evidence':dict(matches),'fictional_or_hypothetical':fictional}

def normalize(base_entities, base_origins, taxonomy, tax_origins):
    if not (ROOT/'entities.jsonl').exists() or not (ROOT/'collection_summary.json').exists():
        raise RuntimeError('Collector must finish before final normalized export; checkpoint remains available.')
    out = ROOT/'normalized'; out.mkdir(exist_ok=True)
    source_nodes = list(load_jsonl(ROOT/'entities.jsonl'))
    assert {r['qid'] for r in source_nodes} == set(base_entities), 'Raw / final collector node mismatch'
    all_origins = {**base_origins, **tax_origins}
    nodes, type_proofs = [], []
    for record in source_nodes:
        qid = record['qid']; typed = classify(qid, base_entities[qid], taxonomy)
        for kind, paths in typed['type_evidence'].items():
            for path in paths:
                for step in path:
                    meta = all_origins.get(step['subject_qid'], {})
                    type_proofs.append({**step,'supported_type':kind,'classified_qid':qid,
                                        'source_response_url':meta.get('source_url'),
                                        'raw_response_sha256':meta.get('sha256'),
                                        'retrieved_at_utc':meta.get('retrieved_at_utc')})
        nodes.append({**record,**typed,'knowledge_available_at':None,
                      'historical_prediction_eligible':False,'risk_label':None,
                      'risk_label_status':'unknown_no_observed_outcome'})
    downloaded_nodes = nodes
    excluded_nodes = [n for n in downloaded_nodes if n['fictional_or_hypothetical']]
    excluded_ids = {n['qid'] for n in excluded_nodes}
    nodes = [n for n in downloaded_nodes if n['qid'] not in excluded_ids]
    node_ids = {n['qid'] for n in nodes}
    downloaded_statement_records = list(load_jsonl(ROOT/'relations.jsonl'))
    excluded_statements = [r for r in downloaded_statement_records
                          if r['subject_qid'] in excluded_ids or r['object_qid'] in excluded_ids]
    statement_records = [r for r in downloaded_statement_records
                         if r['subject_qid'] not in excluded_ids and r['object_qid'] not in excluded_ids]
    facts = {}
    for original in statement_records:
        original.update({'statement_publication_time':None,'knowledge_available_at':None,
                         'historical_prediction_eligible':False,
                         'is_supplier_dependency':False,'is_causal_failure_edge':False})
        key = (original['subject_qid'],original['property_id'],original['object_qid'])
        if key not in facts:
            facts[key] = {'subject_qid':key[0],'property_id':key[1],'predicate':original['predicate'],
                          'object_qid':key[2],'target_record_collected':key[2] in node_ids,
                          'statement_ids':[],'license':'CC0-1.0','is_synthetic':False,
                          'is_supplier_dependency':False,'is_causal_failure_edge':False,
                          'knowledge_available_at':None,'historical_prediction_eligible':False}
        facts[key]['statement_ids'].append(original['statement_id'])
    all_facts = list(facts.values())
    closed_facts = [r for r in all_facts if r['subject_qid'] in node_ids and r['object_qid'] in node_ids]
    closed_statements = [r for r in statement_records if r['object_qid'] in node_ids]
    unknown = Counter(r['object_qid'] for r in all_facts if not r['target_record_collected'])
    labels = [{'qid':n['qid'],'label':None,'label_code':-1,'status':'unknown_no_observed_risk_outcome',
               'prediction_day':None,'target_day':None,'gold_label':False,'is_synthetic':False} for n in nodes]
    write_jsonl(out/'nodes.jsonl',nodes)
    write_jsonl(out/'excluded_entities.jsonl',excluded_nodes)
    write_jsonl(out/'downloaded_entities_audit.jsonl',downloaded_nodes)
    write_jsonl(out/'excluded_relation_statements.jsonl',excluded_statements)
    write_jsonl(out/'relation_statements.jsonl',closed_statements)
    write_jsonl(out/'relations.jsonl',closed_facts)
    write_jsonl(out/'all_nonfictional_relation_statements.jsonl',statement_records)
    write_jsonl(out/'all_nonfictional_relations_including_unresolved.jsonl',all_facts)
    write_jsonl(out/'unresolved_relation_statements.jsonl',[r for r in statement_records if r['object_qid'] not in node_ids])
    write_jsonl(out/'type_evidence.jsonl',type_proofs)
    write_jsonl(out/'unknown_targets.jsonl',[{'qid':qid,'unique_fact_count':count,
                'reason':'uncollected_object_record; no placeholder entity created'} for qid,count in sorted(unknown.items())])
    write_jsonl(out/'risk_labels.jsonl',labels)
    type_counts = {t:0 for t in ['firm','product','material','industry','region','unclassified']}
    type_counts.update(Counter(n['entity_type'] for n in nodes))
    domains = {}
    for domain in ['Auto','Semi','Energy']:
        subset = [n for n in nodes if domain in n['domain_candidates']]
        write_jsonl(out/('domain_'+domain.lower()+'_firm_candidates.jsonl'),
                    [n for n in subset if n['entity_type']=='firm' and not n['fictional_or_hypothetical']])
        counts = {t:0 for t in type_counts}; counts.update(Counter(n['entity_type'] for n in subset))
        domains[domain] = {'node_records':len(subset),'strict_type_counts':counts,
                         'domain_membership':'selector candidate or one-hop relation to candidate; domains may overlap'}
    summary = {'status':'real_public_wikidata_snapshot_normalized','completed_at_utc':stamp(),
        'license':'CC0-1.0','is_synthetic':False,'downloaded_entity_count':len(downloaded_nodes),
        'usable_real_entity_count':len(nodes),'node_records':len(nodes),'strict_type_counts':type_counts,
        'class_or_category_records':sum(n['entity_form']=='class_or_category' for n in nodes),
        'fictional_or_hypothetical_excluded_records':len(excluded_nodes),
        'excluded_incident_relation_statements':len(excluded_statements),
        'downloaded_selected_relation_statements':len(downloaded_statement_records),
        'statement_records':len(closed_statements),'unique_graph_facts':len(closed_facts),
        'nonfictional_relation_statements_including_unresolved':len(statement_records),
        'nonfictional_unique_facts_including_unresolved':len(all_facts),
        'facts_with_both_entities_collected':sum(r['target_record_collected'] for r in all_facts),
        'closed_real_world_graph_facts':len(closed_facts),
        'uncollected_object_qids':len(unknown),'predicate_unique_fact_counts':dict(Counter(r['predicate'] for r in closed_facts)),
        'domain_candidate_counts':domains,'supplier_dependency_edges':0,'observed_risk_labels':0,
        'unknown_risk_labels':len(labels),'historically_available_training_records':0,
        'limitations':['Actual current community facts; not recovered original SC data or exact Table 1 counts.',
            'Five-type mapping uses only saved P31/P279 facts; unsupported or conflicting mappings remain unclassified.',
            'Products/materials/industries may include classes or categories, explicitly marked entity_form.',
            'Domain membership is candidate provenance and may overlap; it is not a guaranteed firm sector classification.',
            'P1056 may name a product, material or service; it is not a product type label or Supplies edge.',
            'Ownership, industry, location and manufacture facts are not supplier dependencies or causal risk edges.',
            'All risk outcomes, resilience capacities and causal failure paths remain unknown.',
            'No publication time is inferred from retrieval, revision, relationship-validity or reference-retrieval dates.',
            'This October 2026 snapshot cannot be used as known-at-2018--2025 prediction input without historical provenance.']}
    summary['files'] = {p.name:{'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in out.glob('*.jsonl')}
    save_json(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=True,indent=2),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--taxonomy-only',action='store_true')
    parser.add_argument('--taxonomy-depth',type=int,default=8)
    parser.add_argument('--taxonomy-limit',type=int,default=2500)
    args=parser.parse_args()
    base, origins, _ = load_official_entities(ROOT/'raw',['seed_entities_','related_entities_'])
    taxonomy, tax_origins, _ = collect_taxonomy(base,args.taxonomy_depth,args.taxonomy_limit)
    if not args.taxonomy_only: normalize(base,origins,taxonomy,tax_origins)

if __name__=='__main__': main()
