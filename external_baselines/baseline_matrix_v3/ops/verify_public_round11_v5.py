"""Portable Round11 cold source/subset/physical/score/paired verification.

No credentials, network, Git, model, private tensor or GPU access. Original
source identities and sanitized bodies have separate, explicit maps. Only a
new exact scratch directory may receive generated CPU artifacts.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import zipfile

sys.dont_write_bytecode = True
SCHEMA = 'trajkv.round11.public_content_refs.v5'
DOMAINS = ('textvqa', 'infovqa', 'docvqa', 'chartqa')
STATES = ('CLOSED_COMPLETE', 'CLOSED_WITH_FAILURES', 'STOPPED', 'NOT_RUN')
FROZEN_SOURCE_PINS = {
    'code/score.py': '72b5182eeb97e930403a970feebb4f1fadac23d04544c9646c6c9fd6b34bfba6',
    'code/metrics.py': '98b260c766e095f2b629d603de5cfd6519a91dccaaee471f02486955bb590d47',
    'code/vendor_textvqa_metric.py': 'b1a8e90abbf35e32e1e6c146c36c4a47ad72b71e73011f97cc3729aa1402b4c5',
    'code/round10_runner_v1.py': '22b5e45152ea9be982ae7ba77950ed09bd065ae574b58b80725beb0983c54e96',
}


def require(value, message):
    if not value: raise ValueError('ROUND11_COLD: ' + message)


def load(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def canonical(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha_stream(stream):
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024**2), b''): h.update(chunk)
    return h.hexdigest()


def sha(path):
    with Path(path).open('rb') as stream: return sha_stream(stream)


def io_path(value):
    p = Path(value).resolve(); prefix = chr(92)*2+'?'+chr(92)
    return Path(prefix+str(p)) if os.name == 'nt' and not str(p).startswith(prefix) else p


def lf_identity(path):
    def chunks():
        with Path(path).open('rb') as stream:
            pending = b''
            for chunk in iter(lambda: stream.read(1024**2), b''):
                chunk = pending+chunk; pending = b''
                if chunk.endswith(b'\r'): chunk, pending = chunk[:-1], b'\r'
                yield chunk.replace(b'\r\n', b'\n')
            if pending: yield pending
    total = 0; digest = hashlib.sha256()
    for chunk in chunks(): digest.update(chunk); total += len(chunk)
    blob = hashlib.sha1(f'blob {total}\0'.encode())
    for chunk in chunks(): blob.update(chunk)
    return {'sha256': digest.hexdigest(), 'blob_oid_sha1': blob.hexdigest(), 'bytes': total}


def safe_relative(name):
    p = PurePosixPath(name)
    require(isinstance(name, str) and not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in name and name == p.as_posix(), 'unsafe/ambiguous archive path')
    return name


def scoped_scalar_sha(input_entries, generated_output):
    """No arbitrary fallback: indexed inputs or this call's exact outputs only."""
    allowed = {Path(path).resolve(): e for path, e in input_entries.items()}; output = Path(generated_output).resolve()
    def resolve(path):
        actual = Path(path).resolve()
        if actual in allowed:
            e = allowed[actual]; require(sha(actual) == e['public_sha256'], 'indexed public input body changed')
            require(isinstance(e.get('source_sha256'), str) and len(e['source_sha256']) == 64, 'missing original input identity')
            return e['source_sha256']
        require(actual.is_relative_to(output) and actual.is_file(), 'unindexed input or unrelated scratch output')
        return sha(actual)
    return resolve


def verify_files(root, archive=None):
    root = io_path(root); index = load(root/'PUBLIC_CONTENT_REFS.json'); entries = index['files']
    require(index.get('schema') == SCHEMA and index.get('publication_eligible') is True and index.get('fixture_only') is False, 'actual eligible Round11 package required')
    require(index.get('completion_scope') == 'root_sealed_research_stage_only' and index.get('SourceRun_admission_claimed') is False and index.get('whole_external_library_completion_claimed') is False and index.get('private_tensor_payloads_archived') is False, 'unsupported admission/completion/private-payload claim')
    allowed = {'.py', '.json', '.jsonl', '.csv', '.md', '.txt', '.toml', '.yaml', '.yml'}
    total = 0; lines = 0
    for name, e in entries.items():
        safe_relative(name); path = root/name
        require(path.resolve().is_relative_to(root.resolve()) and path.is_file(), 'missing/outside indexed member')
        require(path.suffix.lower() in allowed or path.name == '.gitattributes' or path.name.lower().startswith(('license', 'copying', 'notice')), 'private/binary/tensor/image/weight member')
        require(path.stat().st_size == e['public_bytes'] and sha(path) == e['public_sha256'], 'public body identity differs: '+name)
        require(lf_identity(path) == e['git_lf'], 'Git LF/body OID differs: '+name)
        if e['transformation'] == 'EXACT': require(e['source_sha256'] == e['public_sha256'], 'false exact original identity')
        ref = e['content_ref']
        require(ref['member'] == index['namespace']+'/'+name and ref['public_body_sha256'] == e['public_sha256'] and ref['archive_asset'].startswith('https://github.com/zhouz4884-coder/TrajKV/releases/download/'+index['release_tag']+'/'), 'content_ref differs')
        if e.get('line_identities_file'):
            side = e['line_identities_file']; require(side in entries, 'missing exact line-identity file'); count = 0
            with path.open('rb') as data, (root/side).open(encoding='utf-8') as identities:
                for count, line in enumerate(data, 1):
                    identity = json.loads(identities.readline()); row = json.loads(line)
                    require(identity['line'] == count and identity['id'] == row.get('id') and identity['public_sha256'] == hashlib.sha256(line).hexdigest(), 'exact public line identity differs')
                require(not identities.readline(), 'extra line identities')
            require(count == e['line_count'], 'line count differs'); lines += count
        total += e['public_bytes']
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(actual == set(entries)|{'PUBLIC_CONTENT_REFS.json'}, 'unindexed or missing public body')
    force = set(index['git_stage_plan']['force_paths'])
    require(force == {index['namespace']+'/'+name for name, e in entries.items() if e['expected_in_git']} and index['git_stage_plan']['plus_generated_index'] == index['namespace']+'/PUBLIC_CONTENT_REFS.json' and index['git_stage_plan']['verify_actual_tree_and_blob_bodies'] is True and index['git_stage_plan']['configs_data_ignore_rule_must_not_hide_required_files'] is True, 'explicit expected Git tree differs')
    for rel in index['required_reference_files']:
        require(rel in entries and entries[rel]['expected_in_git'] and index['namespace']+'/'+rel in force, 'ignored input/reference omitted from expected Git tree')
    alias_helper = public_module('_round11_exact_alias_validation_v4', root/'ops/round11_public_aliases_v5.py')
    alias_helper.validate_alias_bindings(index)
    if archive:
        with zipfile.ZipFile(archive) as zipped:
            names = zipped.namelist(); prefix = index['namespace']+'/'
            require(len(names) == len(set(names)) and set(names) == {prefix+n for n in entries}|{prefix+'PUBLIC_CONTENT_REFS.json'}, 'archive member coverage/duplicates differ')
            require(zipped.read(prefix+'PUBLIC_CONTENT_REFS.json') == (root/'PUBLIC_CONTENT_REFS.json').read_bytes(), 'archive index differs')
            for name, e in entries.items():
                with zipped.open(prefix+name) as stream: require(sha_stream(stream) == e['public_sha256'], 'archive body differs: '+name)
    return root, index, entries, total, lines


def public_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


@contextmanager
def public_imports(root):
    names = {p.stem for p in (root/'code').glob('*.py')} | {p.stem for p in (root/'ops').glob('*.py')}
    previous = {name: sys.modules[name] for name in names if name in sys.modules}; old_path = sys.path[:]
    for name in names: sys.modules.pop(name, None)
    sys.path[:0] = [str(root/'code'), str(root/'ops')]
    try: yield
    finally:
        for name in names: sys.modules.pop(name, None)
        sys.modules.update(previous); sys.path[:] = old_path


class SourceMap:
    def __init__(self, root, entries, aliases): self.root, self.entries, self.aliases = root, entries, aliases
    def source(self, value):
        value = str(value).replace('\\', '/')
        while '//' in value: value = value.replace('//', '/')
        rel = self.aliases.get(value, value); require(rel in self.entries, 'exact source alias/member missing: '+value); return rel
    def path(self, value): return self.root/self.source(value)
    def original(self, value, digest):
        rel = self.source(value); require(self.entries[rel]['source_sha256'] == digest, 'original source identity differs: '+rel); return rel
    def under(self, folder, name): return self.source(folder+'/'+safe_relative(name))
    def doc(self, value): return load(self.path(value))
    def line_ids(self, rel):
        e = self.entries[self.source(rel)]; require(e.get('line_identities_file') in self.entries, 'original exact JSONL identities missing')
        return [json.loads(line) for line in (self.root/e['line_identities_file']).read_text(encoding='utf-8').splitlines()]


def verify_producer(source, raw, proofname, entry, state):
    under = lambda name: source.under(raw, name); doc = lambda name: source.doc(under(name))
    m, done, cfg, proof = map(doc, ('manifest.json', 'complete.json', 'source_config.json', proofname))
    fresh = entry['kind'] == 'fresh'; item = entry['item'] if fresh else entry['source_item']; cell = item['cell']; n = cell['expected_QA']
    require(m['schema'] == 'baseline_matrix.raw_inference.v1' and m['references_read'] is False and m['prediction_cache_used'] is False and done['quality_scored'] is False and done['coverage_complete'] is True, 'unscored complete source producer required')
    require(m['expected_QA'] == done['expected_QA'] == done['recorded_QA'] == n and len(m['expected_ids']) == n and len(set(m['expected_ids'])) == n, 'original producer denominator changed')
    require(m['config'] == cfg and m['config_sha256'] == cell['config_sha256'], 'original producer config differs')
    source.original(under('manifest.json'), done['manifest_sha256']); source.original(under('records.jsonl'), done['records_jsonl_sha256']); source.original(under('source_config.json'), cell['config_sha256'])
    if fresh:
        from round11_public_confirmation_v5 import d96_runner_contract
        admission = d96_runner_contract(cell['script'])
        require(proofname == admission['proof_name'], 'actual D96 runner/proof namespace differs')
    expected_status = admission['proof_status'] if fresh else 'PASS_ROUND10_SEQUENCE_CHILD_EXECUTED_SOURCE_SHA'
    require(proof['status'] == expected_status and proof['remote_hashes_computed_independently'] is True and proof['SourceRun_admission_established'] is False and proof['cell_name'] == cell['name'] and proof['parent_producer'] == item['job']['name'], 'independent original producer proof differs')
    require(len(proof['files']) == len({e['file'] for e in proof['files']}), 'duplicate producer proof files')
    for e in proof['files']:
        rel = under(e['file']); source.original(rel, e['sha256']); require(source.entries[rel]['source_bytes'] == e['bytes'], 'original proof file length differs')
    if not fresh:
        source.original(under(proofname), entry['source_receipt_sha256'])
        require(m['runner_schema'] == 'trajkv.round10_research.raw.v1' and cell['script'] == 'code/round10_runner_v1.py' and n == {'textvqa': 64, 'infovqa': 64, 'docvqa': 16, 'chartqa': 16}[entry['dataset']], 'reused reference must retain full original Round10 producer count/schema')
        require(cfg['candidate']['mode'] == ('FP16' if entry['mode'] == 'FP16' else 'original') and cfg['variant_source']['name'] == ('r7_budget_k2_cond' if entry['mode'] == 'R7high' else 'r7_joint_k1_gt_cond'), 'reused reference mode/lineage differs')
    else:
        require(m['runner_schema'] == admission['raw_schema'] and cfg['method']['family'] == admission['method_family'] and cfg['candidate']['mode'] == entry['mode'] and entry['mode'] in admission['modes'], 'actual fresh driver/schema/mode differs')
    require(cfg['candidate'].get('technical_only', False) is False and cfg['count'] == n and cfg['start'] == 0, 'technical/partial producer cannot supply task quality')
    job, seq, inventory, launch = map(doc, ('source_parent_job.json', 'source_sequence_config.json', 'source_inventory.json', 'launch_receipt.json'))
    require(job == item['job'] and cell in seq['entries'], 'actual plan/producer job differs')
    source.original(under('source_parent_job.json'), item['job_sha256']); source.original(under('source_sequence_config.json'), item['sequence_sha256']); source.original(under('source_inventory.json'), launch['source_inventory_sha256'])
    require(launch['job_sha256'] == item['job_sha256'] and launch['name'] == proof['parent_producer'] and type(launch['pid']) is int and launch['pid'] > 0 and launch['source_snapshot'] == m['runtime_source_root'], 'actual parent PID/snapshot differs')
    snapshot = launch['source_snapshot']; remote = job['argv'][2].rsplit('/code/', 1)[0]
    executed = [snapshot+'/code/'+a.rsplit('/code/', 1)[1] if a.startswith(remote+'/code/') and a.endswith('.py') else a for a in job['argv']]
    require(executed == launch['argv_executed'] and executed[2] == snapshot+'/code/round10_sequence_v1.py', 'actual parent argv differs')
    source.original(under('source_parent_driver.py'), item['sequence_script_sha256']); require(inventory['code/round10_sequence_v1.py'] == item['sequence_script_sha256'], 'parent frozen source differs')
    event = proof['actual_child_start']; start = event['event']
    require(json.loads(event['exact_line_utf8']) == start and start['name'] == cell['name'] and type(start['pid']) is int and start['pid'] > 0 and start['config_sha256'] == cell['config_sha256'] and start['argv'] == [executed[0], '-B', snapshot+'/'+cell['script'], '--config', cell['config'], '--output', cell['output']], 'actual child START_CELL/PID/config differs')
    if fresh: require(m['runtime_process']['PID'] == start['pid'] and m['runtime_process']['parent_PID'] == launch['pid'] and m['runtime_process']['source_snapshot_root'] == snapshot, 'fresh child/parent PID or snapshot does not bind actual launch and START_CELL')
    require(proof['actual_driver'] == cell['script'] and m['source_hashes'] == cfg['expected_source_hashes'] and m['source_hashes'][cell['script']] == cell['script_sha256'], 'actual driver/runtime graph differs')
    source.original(under('source_driver.py'), cell['script_sha256'])
    for rel, digest in m['source_hashes'].items():
        require(inventory[rel] == digest, 'actual snapshot graph differs'); source.original(under('source_files/'+rel), digest); source.original(rel, digest)
    parent = json.loads(proof['parent_state_snapshot']['body_utf8'])
    require(parent['source_root'] == snapshot and parent['plan_sha256'] == item['sequence_sha256'], 'parent snapshot plan/source differs')
    success = proof['parent_cell_success']; closed = [r for r in parent['completed_cells'] if r['name'] == cell['name']]
    if success:
        require(len(closed) == 1 and closed[0]['exit_code'] == 0 and closed[0]['recorded_QA'] == n and closed[0]['output'] == cell['output'] and closed[0]['complete_sha256'] == source.entries[under('complete.json')]['source_sha256'] and done['status'] == 'PASS_RAW_COMPLETE', 'original parent exit0/raw3 closure differs')
    else:
        failed = parent.get('failed_cell', {}); require(not closed and failed.get('name') == cell['name'] and failed['exit_code'] != 0 and failed['output'] == cell['output'] and done['status'] == 'COMPLETE_WITH_FAILURES', 'preserved actual failure parent closure differs')
    require(success or state == 'CLOSED_WITH_FAILURES', 'quality-complete cell lacks actual parent exit0')
    if not fresh: require(success, 'reused references require original parent success')
    if source.entries[under(proofname)]['transformation'] == 'EXACT':
        require(hashlib.sha256(event['exact_line_utf8'].encode()).hexdigest() == event['exact_line_sha256'] and hashlib.sha256(proof['parent_state_snapshot']['body_utf8'].encode()).hexdigest() == proof['parent_state_snapshot']['sha256'], 'original embedded proof bytes differ')
    for key, digest in cfg['expected_asset_sha256'].items(): source.original(under('assets/base_'+key+'.json'), digest)
    source.original(under('assets/original_profile.json'), cfg['variant_source']['sha256'])
    profile = doc('assets/original_profile.json'); variants = [v for v in profile['variants'] if v['name'] == cfg['variant_source']['name']]
    numeric = lambda v: {k: x for k, x in v.items() if k not in cfg['expected_asset_sha256']}
    require(len(variants) == 1 and numeric(variants[0]) == numeric(cfg['variant']), 'original R7 base numeric lineage changed')
    executed_legacy = set(done.get('executed_legacy_source_sha256', {}).items())
    proven_legacy = {(e['remote_file'], e['sha256']) for e in proof['files'] if e['file'].startswith('legacy_sources/')}
    require(executed_legacy <= proven_legacy, 'executed legacy sources omitted from full source proof')
    source.original(under('input_manifest.json'), item['input_manifest_sha256'])
    if fresh: source.original(under('source_group_manifest.json'), item['source_group_manifest_sha256'])
    rawrows = [json.loads(line) for line in source.path(under('records.jsonl')).read_text(encoding='utf-8').splitlines()]
    require([r['id'] for r in rawrows] == m['expected_ids'] and len(rawrows) == n and dict(Counter(r['status'] for r in rawrows)) == done['counts'], 'full original producer coverage/order/status differs')
    return m, done, cfg, proof, rawrows


def verify(root, archive=None, recompute_scores=False, scratch=None):
    root, index, entries, total_bytes, line_count = verify_files(root, archive)
    source = SourceMap(root, entries, index['source_aliases']); contract = load(root/'evidence/ROUND11_STAGE_CONTRACT.json'); seal = load(root/'evidence/ROUND11_STAGE_SEAL.json')
    for rel, digest in FROZEN_SOURCE_PINS.items(): source.original(rel, digest)
    require(set(index.get('portable_verifier_dependencies', {})) == {'ops/round11_public_confirmation_v5.py', 'ops/round11_public_aliases_v5.py', 'ops/round11_public_metric_contracts_v5.py', 'ops/round11_public_expansion_v5.py'}, 'Exact portable cohort, group-contract and alias cold dependencies required')
    for rel, digest in index['portable_verifier_dependencies'].items(): source.original(rel, digest)
    require(contract['schema'] in ('trajkv.round11.public_stage_contract.v3', 'trajkv.round11.public_stage_contract.v4', 'trajkv.round11.public_stage_contract.v5') and contract['stage_state'] == 'ROOT_SEALED' and contract['SourceRun_admission'] is False and contract['whole_external_library_completion_claimed'] is False and contract['history_and_all_previous_namespaces_releases_must_be_preserved'] is True, 'actual final Root stage contract required')
    require(seal['schema'] == 'trajkv.round11.public_stage_seal.'+contract['schema'].rsplit('.',1)[1] and seal['status'] == 'SEALED_FOR_LOCAL_PUBLIC_PACKAGE' and seal['phase_label'] == contract['phase_label'] and seal['SourceRun_admission'] is False and seal['base_public_commit'] == contract['base_public_commit'] == '43d03af0e8938aeb5f73ffc0c104093a7bd14d73', 'Root/history seal differs')
    source.original(contract['root_seal']['path'], contract['root_seal']['sha256'])
    for key in ('groups', 'files', 'supporting_evidence'):
        binding = index['sealed_scope_bindings'][key]
        require(seal[key+'_sha256'] == binding['source_sha256'] and canonical(contract[key]) == binding['public_sha256'], 'source/public Root semantic scope differs: '+key)
    require(len(contract['groups']) == len({g['group_id'] for g in contract['groups']}), 'duplicate group')
    for f in contract['files']: source.original(f['path'], f['sha256'])
    require(recompute_scores, 'Round11 eligible cold verification requires --recompute-scores')
    if scratch is not None:
        scratch = Path(scratch).resolve(); require(not scratch.exists() and not scratch.is_relative_to(root.resolve()), 'new exact scratch outside public package required'); scratch.mkdir(parents=True)
    quality = []; pairs_out = []; group_results = []; physical_count = 0; fresh_producers = set(); reused_sources = {}; totals = Counter()
    combined = {'status': 'NOT_IN_THIS_CUTOFF', 'new_inference_events': 0, 'combined_quality_computed': False}
    with public_imports(root):
        physical = public_module('_round11_physical_frozen', root/'ops/verify_public_round10_v3.py')
        public_module('vendor_textvqa_metric', root/'code/vendor_textvqa_metric.py')
        metrics = public_module('metrics', root/'code/metrics.py'); scorer = public_module('score', root/'code/score.py')
        gate = source.doc('audit/metrics_gate_v2.json')
        for rel, digest in gate['code_hashes'].items(): source.original(rel, digest)
        input_entries = {(root/rel).resolve(): e for rel, e in entries.items()}; generated = scratch if scratch is not None else root/'__no_generated_outputs__'
        resolver = scoped_scalar_sha(input_entries, generated)
        drivers = {}
        for g in contract['groups']:
            label = g['group_id']; state = g['execution_disposition']; require(state in ('EXECUTED', 'NOT_RUN'), 'pending execution scope cannot export')
            if state == 'NOT_RUN':
                require(not g['cells'] and not g.get('plan') and not g.get('harvest_root') and not g.get('summary_root') and g.get('reason') and g.get('primary_mean') is None, 'unrun branch fabricated result or quality')
                e = g['disposition_evidence']; source.original(e['path'], e['sha256']); group_results.append({'group_id': label, 'execution_disposition': 'NOT_RUN', 'primary_mean': None, 'actual_fresh_answer_events': 0, 'reused_reference_view_events': 0}); continue
            source.original(g['plan']['path'], g['plan']['sha256']); source.original(g['adapter']['path'], g['adapter']['sha256']); plan = source.doc(g['plan']['path'])
            if plan.get('schema') == 'trajkv.round11.expand384_harvest_plan.v1':
                from round11_public_expansion_v5 import verify_group
                result = verify_group(source, g, plan, index['phase'], scorer, metrics, physical, public_module, resolver, fresh_producers, contract['groups'])
                group_results.append(result['group']); quality.extend(result['quality']); pairs_out.extend(result['pairs']); physical_count += result['physical_count']; totals.update(result['counts'])
                continue
            if plan.get('schema') == 'trajkv.round11.confirmation_harvest_plan.v2':
                from round11_public_confirmation_v5 import verify_group
                result = verify_group(source, g, plan, index['phase'], scorer, metrics, physical, public_module, resolver, fresh_producers, contract['groups'])
                group_results.append(result['group']); quality.extend(result['quality']); pairs_out.extend(result['pairs']); physical_count += result['physical_count']; totals.update(result['counts'])
                continue
            require(plan.get('schema', '').startswith('trajkv.round11.harvest_plan.'), 'explicit Round11 actual harvest-plan schema required')
            source.original(plan['source_group_manifest_path'], plan['source_group_manifest_sha256']); source_group = source.doc(plan['source_group_manifest_path'])
            require(source_group.get('schema', '').startswith('trajkv.round11.source_group.'), 'explicit Round11 source-group schema required')
            require(source_group['reference_input_manifest_sha256'] == plan['input_manifest_sha256'], 'source-group original input manifest lineage differs')
            source.original(source_group['reference_input_manifest_path'], plan['input_manifest_sha256'])
            require(g['counts'] == {d: source_group['datasets'][d]['count'] for d in DOMAINS}, 'actual source group counts differ')
            planned = {(e['dataset'], e['mode']): e for e in plan['entries']}; declared = {(r['dataset'], r['mode']): r for r in g['cells']}
            require(len(planned) == len(plan['entries']) and len(declared) == len(g['cells']) and set(planned) == set(declared) and set(g['modes']) == {e['mode'] for e in plan['entries']}, 'actual exact plan matrix differs')
            summary = source.doc(g['summary_root']+'/summary.json'); pairs = source.doc(g['summary_root']+'/paired_comparisons.json'); require(summary['plan_sha256'] == g['plan']['sha256'], 'frozen summary plan differs')
            frozen_points = {(p['dataset'], p['mode']): p for p in summary['points']}; bundles = {}; counts = Counter()
            require(set(frozen_points) == set(planned) and len(frozen_points) == len(summary['points']) and summary['expected_cells_or_views'] == len(planned) and summary['expected_new_free_generation_QA'] == sum(g['counts'][e['dataset']] for e in plan['entries'] if e['kind'] == 'fresh') and summary['expected_reused_reference_views'] == sum(g['counts'][e['dataset']] for e in plan['entries'] if e['kind'] == 'reused') and summary['actual_closed_new_free_generation_QA'] == sum(p['fresh_free_generation_QA'] for p in summary['points']) and summary['actual_closed_reused_reference_views'] == sum(p['reused_reference_views'] for p in summary['points']), 'D96 summary actual matrix/fresh/reused accounting differs')
            scope = next(x for x in index['phase']['groups'] if x['group_id'] == label)
            for key, entry in planned.items():
                row = declared[key]; ds, mode = key; n = g['counts'][ds]; disposition = row['disposition']
                require(row['expected_QA'] == n and row['kind'] == entry['kind'] and disposition in STATES, 'cell kind/count/final disposition differs')
                if disposition in ('STOPPED', 'NOT_RUN'):
                    require(row.get('primary_mean') is None and row.get('reason') and frozen_points[key]['primary_mean'] is None, 'unexecuted quality must remain null and cannot hide an existing primary score'); e = row['disposition_evidence']; source.original(e['path'], e['sha256']); quality.append({'group_id': label, **row, 'primary_recomputed': False, 'scored_QA': 0}); continue
                fresh = entry['kind'] == 'fresh'
                phase_row = next(x for x in scope['cells'] if (x['dataset'], x['mode']) == key)
                raw = phase_row['raw']; view = phase_row['view']; m, done, cfg, proof, rawrows = verify_producer(source, raw, phase_row['proof_name'], entry, disposition)
                if fresh:
                    from round11_public_confirmation_v5 import d96_runner_contract, scoped_driver_sha, verify_runtime_graph
                    driver_path = entry['item']['cell']['script']; driver_name = Path(driver_path).stem
                    if driver_path not in drivers:
                        drivers[driver_path] = public_module(driver_name, root/driver_path)
                    driver = drivers[driver_path]; admission = d96_runner_contract(driver_path)
                    require(driver.SCHEMA == admission['raw_schema'] and list(driver.MODES) == admission['modes'], 'actual archived D96 driver declares another schema/mode set')
                    verify_runtime_graph(driver, resolver, m['source_hashes'])
                if mode != 'FP16': physical.r7_verify_executed_roles(cfg, done.get('executed_legacy_source_sha256', {}))
                require(phase_row['original_producer_QA'] == m['expected_QA'], 'producer denominator was replaced by subset denominator')
                domain = plan['domains'][ds]
                for field in ('inputs', 'refs', 'source_lock'): source.original(domain[field], domain[field+'_sha256'])
                lock = source.doc(domain['source_lock'])
                for field in ('inputs', 'refs'):
                    rel = source.source(domain[field]); name = PurePosixPath(domain[field].replace('\\', '/')).name
                    require(lock['files'][name]['sha256'] == entries[rel]['source_sha256'] and lock['files'][name]['bytes'] == entries[rel]['source_bytes'], 'frozen original source lock differs')
                require(m['inputs_sha256'] == domain['inputs_sha256'], 'actual producer input differs from source group'); source.original(raw+'/source_inputs.jsonl', m['inputs_sha256'])
                selected = [json.loads(line) for line in source.path(view+'/selected_records.jsonl').read_text(encoding='utf-8').splitlines()]; lineage = source.doc(view+'/source_lineage.json'); scored = [json.loads(line) for line in source.path(view+'/scored_records.jsonl').read_text(encoding='utf-8').splitlines()]; point = source.doc(view+'/summary.json')
                require(canonical(point) == canonical(frozen_points[key]), 'per-view and group frozen score differ')
                for rel, digest in point['scoring_source_pins'].items(): source.original(rel, digest)
                require(lineage['inference_run'] is False and lineage['fresh_inference_source'] is fresh and lineage['reused_reference_source'] is (not fresh) and lineage['canonical_SourceRun_admission_claimed'] is False and lineage['source_group_manifest_sha256'] == plan['source_group_manifest_sha256'], 'view freshness/source-group identity differs')
                require(lineage['schema'] == 'trajkv.round11.exact_source_subset.v1' and point['schema'] == 'trajkv.round11.offline_source_subset_score.v1', 'actual source-subset score schema differs')
                require(lineage['source_runtime_sha256'] == m['source_hashes'] and lineage['source_producer'] == proof['parent_producer'] and lineage['source_driver'] == proof['actual_driver'], 'view original producer/runtime graph lineage differs')
                for name, digest in lineage['source_raw_sha256'].items(): source.original(raw+'/'+name, digest)
                source.original(view+'/selected_records.jsonl', lineage['selected_records_sha256']); source.original(view+'/source_lineage.json', point['source_lineage_sha256']); source.original(view+'/scored_records.jsonl', point['scored_records_sha256']); source.original(raw+'/'+phase_row['proof_name'], lineage['source_receipt_sha256'])
                ids = source_group['datasets'][ds]['ids']; require(len(ids) == n and lineage['ids'] == ids and [r['id'] for r in selected] == ids and len(scored) == n and (not fresh or [r['id'] for r in rawrows] == ids), 'exact selected source order/denominator differs')
                original_lines = source.line_ids(raw+'/records.jsonl'); lookup = {r['id']: (i, r) for i, r in enumerate(rawrows, 1)}
                references = {r['id']: r for r in map(json.loads, source.path(domain['refs']).read_text(encoding='utf-8').splitlines())}; inputs = {r['id']: r for r in map(json.loads, source.path(domain['inputs']).read_text(encoding='utf-8').splitlines())}
                require(point['refs_sha256'] == domain['refs_sha256'] and point['source_lock_sha256'] == domain['source_lock_sha256'], 'view frozen reference/source lock differs')
                failures = 0; identities = {}; capacities = []; bits = []
                for r, s, line in zip(selected, scored, lineage['source_lines']):
                    number, producer_row = lookup[r['id']]; original_line = original_lines[number-1]
                    require(r == producer_row and line['id'] == r['id'] == s['id'] and line['line_number'] == s['source_raw_jsonl_line_number'] == number and line['exact_line_sha256'] == s['source_raw_jsonl_exact_line_sha256'] == original_line['source_sha256'], 'selected view is not exact original producer line')
                    for field in ('dataset', 'dataset_revision', 'split', 'source_cluster_id', 'source_id', 'question_id', 'original_image_sha256', 'question_sha256'): require(r.get(field) == inputs[r['id']].get(field), 'raw/source input identity differs: '+field)
                    require(r.get('official_score') is None and references[r['id']]['source_cluster_id'] == inputs[r['id']]['source_cluster_id'], 'raw references leaked or source cluster differs')
                    require(r['status'] in scorer.FINISHED_STATUSES, 'unfinished or unknown raw status')
                    if r['status'] != 'PASS': require(s['score'] is None and s['score_status'] == 'NOT_SCORED_RAW_FAILURE', 'raw failure has primary score'); failures += 1; continue
                    require(r['raw_question'] == inputs[r['id']]['question'] and isinstance(r.get('raw_prediction'), str) and isinstance(r.get('output_token_ids'), list) and r['output_token_ids'] and r['question_unknown_prefix_verified'] is True and r['original_full_prefix_released'] is True and r['original_FP16_visual_alias_retained'] is False and r['first_generation_call']['native_first_logits_injected'] is False, 'genuine full-suffix free-generation/source-release contract differs')
                    amount = physical.physical_complete_bytes(r, m['model_spec']); physical_count += 1; capacities.append(amount); bits.append(r['actual_effective_visual_bits'])
                    require(r['owner_after_generation']['stored_visual_KV_bytes'] == amount and math.isclose(bits[-1], 16*amount/r['full_kv_byte_denominator'], rel_tol=1e-12, abs_tol=1e-12), 'actual complete owner/bits differ')
                    if fresh:
                        driver_path = entry['item']['cell']['script']; driver_name = Path(driver_path).stem
                        if driver_path not in drivers:
                            # Every hash query is restricted to explicit verified archive inputs.
                            mod = public_module(driver_name, root/driver_path); mod.sha = resolver; drivers[driver_path] = mod
                        driver = drivers[driver_path]; require(r['question_suffix'] == driver.frozen.SUFFIX, 'frozen question suffix differs')
                        with scoped_driver_sha(driver, resolver):
                            require(r['round10_budget_vs_original_R7_low'] == driver.round11_budget_guard(mode, r['actual_visual_token_count'], amount, r['r7_bridge'], cfg['variant_source']['name']), 'actual deployed runner complete-B0/source/reader guard differs')
                    elif mode == 'R7low': require(amount == 4590*r['actual_visual_token_count']+89733, 'original reused R7low B0 differs')
                    if mode != 'FP16': physical.r7_verify_record_role_pins(cfg, r)
                    from round11_public_metric_contracts_v5 import verify_metric_fields, RAW_REFERENCE_GROUP
                    metric = verify_metric_fields(metrics, ds, r['raw_prediction'], references[r['id']], s, RAW_REFERENCE_GROUP)
                    require(s['score_status'] == 'SCORED_PASS' and s['stored_visual_KV_bytes'] == amount and s['raw_prediction'] == r['raw_prediction'] and s['output_token_ids'] == r['output_token_ids'], 'scored/raw/physical view differs')
                    identities[r['id']] = {**physical.input_identity(r), 'raw_question_sha256': r['raw_question_sha256']}
                primary = scorer.summarize_group(scored, 2000, 20261123)
                require(canonical(point['groups']['overall']) == canonical(primary) and point['primary_mean'] == primary['primary']['mean'] and point['ci95'] == primary['primary']['ci95'] and point['raw_failure_questions'] == failures, 'full-source-group aggregate/CI/failure denominator differs')
                if failures: require(disposition == 'CLOSED_WITH_FAILURES' and point['primary_mean'] is None and point['resources'] is None, 'failed source group got successful-subset quality')
                else:
                    require(disposition == 'CLOSED_COMPLETE' and done['status'] == 'PASS_RAW_COMPLETE' and point['status'] == 'PASS_FULL_SOURCE_GROUP_SCORE', 'complete task view disposition differs')
                    resources = point['resources']
                    for values, name in ((capacities, 'complete_bytes'), (bits, 'actual_complete_bits')):
                        expected = {'n': n, 'mean': sum(values)/n, 'min': min(values), 'max': max(values)}
                        require(all(math.isclose(resources[name][k], v, rel_tol=1e-12, abs_tol=1e-12) for k, v in expected.items()), 'full physical owner summary differs')
                    require(resources['strict_all_QA_1p3'] == all(160*b <= 13*r['full_kv_byte_denominator'] for b, r in zip(capacities, selected)) and resources['strict_all_QA_1p8'] == all(80*b <= 9*r['full_kv_byte_denominator'] for b, r in zip(capacities, selected)), 'complete all-QA bit threshold differs')
                    bundles[key] = {'context': physical.native_context(m), 'identities': identities, 'scores': {r['id']: r for r in scored}}
                if ds == 'chartqa':
                    for group in ('human', 'aug'):
                        sub = [r for r in scored if ('aug' if r.get('group') == 'augmented' else r.get('group')) == group]
                        if sub: require(canonical(point['groups'][group]) == canonical(scorer.summarize_group(sub, 2000, 20261123)), 'ChartQA subgroup aggregate differs')
                counts['actual_view_answer_events'] += n; counts['actual_fresh_answer_events' if fresh else 'reused_reference_view_events'] += n
                if fresh:
                    producer = tuple(source.entries[source.source(raw+'/'+name)]['source_sha256'] for name in ('manifest.json', 'records.jsonl', 'complete.json'))
                    require(producer not in fresh_producers, 'fresh inference producer counted twice'); fresh_producers.add(producer)
                    counts['fresh_successful_generation_QA'] += n-failures; counts['fresh_failure_status_QA'] += failures
                else:
                    producer = (*tuple(source.entries[source.source(raw+'/'+name)]['source_sha256'] for name in ('manifest.json', 'records.jsonl', 'complete.json')), tuple(lineage['ids']))
                    require(producer not in reused_sources or reused_sources[producer] == n, 'reused source subset denominator differs'); reused_sources[producer] = n
                if not failures: counts['closed_quality_answer_events'] += n
                quality.append({'group_id': label, 'dataset': ds, 'mode': mode, 'disposition': disposition, 'original_producer_QA': m['expected_QA'], 'selected_QA': n, 'scored_QA': n-failures, 'primary_mean': point['primary_mean'], 'primary_recomputed': True, 'kind': entry['kind']})
            expected_pairs = {}
            for ds in DOMAINS:
                for candidate, anchor in g['paired_contrasts']:
                    if (ds, candidate) not in bundles or (ds, anchor) not in bundles: continue
                    left, right = bundles[ds, anchor], bundles[ds, candidate]
                    require(left['context'] == right['context'] and left['identities'] == right['identities'], 'paired runtime/source/positions differ')
                    deltas = [{'id': i, 'source_cluster_id': s['source_cluster_id'], 'delta': right['scores'][i]['score']-s['score']} for i, s in left['scores'].items()]
                    changes = [r['delta'] for r in deltas]; result = {'schema': 'trajkv.round11.paired_source_group.v1', 'status': 'PASS_PAIRED_COMPLETE', **scorer.cluster_bootstrap(deltas, 'delta', 2000, 20261123), 'improved': sum(v > 0 for v in changes), 'degraded': sum(v < 0 for v in changes), 'equal': sum(v == 0 for v in changes), 'gross_gain_pp': 100*sum(max(0, v) for v in changes)/len(changes), 'gross_loss_pp': 100*sum(max(0, -v) for v in changes)/len(changes), 'descriptive_unadjusted_interval': True, 'source_identity_and_runtime_exactly_matched': True}
                    name = ds+'__'+candidate+'_minus_'+anchor; expected_pairs[name] = result; pairs_out.append({'group_id': label, 'comparison': name, 'paired_recomputed': True, 'runtime_source_positions_exact': True})
            require(canonical(pairs) == canonical(expected_pairs), 'paired comparison coverage/CI differs')
            counts['view_cells'] = len(declared); counts['paired_comparisons'] = len(expected_pairs)
            for field, value in counts.items(): require(scope.get(field, 0) == value, 'exported phase actual accounting differs: '+field)
            totals.update(counts); group_results.append({'group_id': label, 'execution_disposition': 'EXECUTED', **dict(counts)})
        from round11_public_expansion_v5 import combined_512_views
        combined = combined_512_views(source, index['phase'], scorer, physical)
    supporting = []
    for e in contract['supporting_evidence']:
        require(e['disposition'] in STATES and e['task_quality_QA'] == 0 and e['teacher_TF_or_technical_is_free_answer_QA'] is False, 'supporting/teacher/QA0 task quality contamination')
        if e['disposition'] in ('STOPPED', 'NOT_RUN'):
            require(e.get('reason') and e.get('primary_mean') is None, 'unexecuted supporting quality/reason differs'); ref = e['disposition_evidence']; source.original(ref['path'], ref['sha256'])
        else:
            require(e.get('files'), 'missing supporting scalar/source files')
            for f in e['files']: source.original(f['path'], f['sha256'])
            for assertion in e.get('scalar_assertions', []):
                require(any(source.source(f['path']) == source.source(assertion['path']) for f in e['files']), 'scalar assertion opened a file outside its explicit supporting inventory')
                value = source.doc(assertion['path'])
                for key in assertion['keys']: value = value[key]
                require(value == assertion['value'], 'supporting scalar assertion differs')
        supporting.append({'evidence_id': e['evidence_id'], 'disposition': e['disposition'], 'task_quality_QA': 0, 'verification_scope': 'archived source and scalar assertions only', 'private_PT_WO_or_raw_logits_replayed': False})
    for field, value in totals.items(): require(index['phase'].get(field, 0) == value, 'whole cutoff actual accounting differs: '+field)
    dedup = {'unique_reused_reference_view_events': sum(reused_sources.values()), 'repeated_reused_reference_view_events': totals['reused_reference_view_events']-sum(reused_sources.values()), 'unique_reused_reference_producer_views': len(reused_sources), 'fresh_producer_cells': len(fresh_producers)}
    require(all(index['phase'].get(key) == value for key, value in dedup.items()) and index['phase']['recorded_QA_tasks_include_failure_status_rows'] is True and index['phase']['fresh_successful_generation_QA_is_reported_separately'] is True, 'unique/repeated reused-view accounting or failure-status task qualification differs')
    result = {'status': 'PASS_PUBLIC_ROUND11_EXPORT_VERIFIED_V5', 'publication_eligible': bool(archive), 'archive_verified': bool(archive), 'quality_recomputations': quality, 'paired_CI_recomputations': pairs_out, 'physical_owner_rows': physical_count, 'groups': group_results, **dict(totals), **dedup, 'supporting_evidence': supporting, 'public_body_bytes': total_bytes, 'public_line_identities': line_count, 'source_public_LF_archive_identities_separate': True, 'SourceRun_admission_claimed': False, 'whole_external_library_completion_claimed': False, 'private_tensor_payloads_verified_or_replayed': False, 'network_or_authentication_used': False, 'new_GPU_QA': 0}
    result['original128_additional384_combined512_separate_report'] = combined
    if scratch is not None:
        with (scratch/'COLD_VERIFICATION.json').open('x', encoding='utf-8', newline='\n') as stream: json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--archive', type=Path); p.add_argument('--recompute-scores', action='store_true'); p.add_argument('--scratch', type=Path); a = p.parse_args()
    print(json.dumps(verify(a.root, a.archive, a.recompute_scores, a.scratch), ensure_ascii=False)); return 0


if __name__ == '__main__': raise SystemExit(main())
