"""Portable text-only R12 producer, scoring and physical-accounting checks.

No transport, credentials, GPU, tensor loading, source mutation or publication.
Local and sanitized public sources share the same semantic validation.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import importlib
import json
import math
import statistics
from pathlib import Path, PurePosixPath
import sys

SCHEMA = 'trajkv.round12.public_scope.v1'
NAMESPACE = 'external_baselines/baseline_matrix_v3'
CORE_FILES = ('manifest', 'records', 'complete')
SAFE_SUFFIXES = {'.py', '.json', '.jsonl', '.csv', '.md', '.txt', '.toml', '.yaml', '.yml'}
PRIVATE_NAMES = {'chat.md', 'cha.md', 'remote.py', 'remote_control.py',
                 'check_github_access.py', 'contractkv_ssh_credential.py'}
ROLES = {'code', 'config', 'inputs', 'refs', 'source_lock', 'raw', 'scored',
         'readback', 'launch', 'producer_inventory', 'calibration', 'technical',
         'attempt', 'report', 'quality_table', 'protocol', 'license', 'support',
         'portable_verifier'}
FROZEN_METRIC_SOURCES = {
    'code/score.py': '72b5182eeb97e930403a970feebb4f1fadac23d04544c9646c6c9fd6b34bfba6',
    'code/metrics.py': '98b260c766e095f2b629d603de5cfd6519a91dccaaee471f02486955bb590d47',
    'code/vendor_textvqa_metric.py': 'b1a8e90abbf35e32e1e6c146c36c4a47ad72b71e73011f97cc3729aa1402b4c5',
}


def require(condition, message):
    if not condition:
        raise ValueError('ROUND12_PUBLIC: ' + message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 ** 2), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def safe_relative(value):
    require(isinstance(value, str) and value and '\\' not in value, 'plain forward-slash public path required')
    p = PurePosixPath(value)
    require(not p.is_absolute() and '..' not in p.parts and ':' not in value
            and not any(x in ('.git', '__pycache__', 'attachments', '.codex') for x in p.parts),
            'unsafe or private public path: ' + value)
    require(p.name.lower() not in PRIVATE_NAMES
            and not p.name.lower().startswith(('private_chat', '已粘贴')),
            'private discussion/transport selected: ' + value)
    require(p.suffix.lower() in SAFE_SUFFIXES or p.name == '.gitattributes'
            or p.name.upper().startswith(('LICENSE', 'COPYING', 'NOTICE')),
            'binary/image/weight/tensor member excluded: ' + value)
    return p.as_posix()


def payload(scope):
    return {k: v for k, v in scope.items() if k != 'root_seal'}


def check_header(scope, *, sealed=False, allow_fixture=False):
    require(scope.get('schema') == SCHEMA and scope.get('namespace') == NAMESPACE,
            'explicit R12 scope schema/namespace required')
    require(isinstance(scope.get('phase_label'), str) and scope['phase_label']
            and all(c.isascii() and (c.isalnum() or c in '_-') for c in scope['phase_label']),
            'safe explicit phase label required')
    fixture = scope.get('fixture_only') is True
    require(allow_fixture or not fixture, 'fixture cannot become production evidence')
    require(scope.get('research_cutoff_complete') is True, 'actual final cutoff required')
    require(scope.get('whole_external_library_completion_claimed') is False
            and scope.get('SourceRun_admission_claimed') is False,
            'bounded stage cannot imply whole-library/SourceRun admission')
    require(scope.get('metric_contract') == 'matrix_v3_metrics_v1', 'frozen metric contract required')
    require(isinstance(scope.get('files'), list) and scope['files'], 'explicit file inventory required')
    require(isinstance(scope.get('cells'), list) and scope['cells'], 'actual producer cells required')
    require(len({x['cell_id'] for x in scope['cells']}) == len(scope['cells']), 'duplicate cell ID')
    for group in scope.get('groups', []):
        require(group.get('disposition') in ('EXECUTED', 'NOT_RUN', 'STOPPED'), 'pending group cannot publish')
        require(set(group.get('cell_ids', [])) <= {x['cell_id'] for x in scope['cells']}, 'group names unknown cell')
        if group['disposition'] == 'NOT_RUN':
            require(not group.get('cell_ids') and group.get('primary_mean') is None
                    and group.get('actual_terminal_answers', 0) == 0 and group.get('reason'),
                    'unrun group requires null quality and explicit reason')
    if sealed:
        require(scope.get('stage_state') == 'ROOT_SEALED' and isinstance(scope.get('root_seal'), dict),
                'unsealed scope cannot export')
        require(scope['root_seal'].get('schema') == 'trajkv.round12.root_seal.v1', 'root seal schema differs')
    return fixture


class LocalSource:
    def __init__(self, scope, allowed_roots):
        self.files = {}
        for item in scope['files']:
            rel = safe_relative(item['public_path'])
            require(rel not in self.files and item.get('role') in ROLES, 'duplicate path/unknown role')
            require(item.get('delivery_channel') in (None, 'BROWSER_AND_ARCHIVE', 'ARCHIVE_ONLY'),
                    'unknown explicit delivery channel')
            path = Path(item['source_path']).resolve()
            require(any(path.is_relative_to(Path(r).resolve()) for r in allowed_roots) and path.is_file(),
                    'missing/outside explicitly permitted source: ' + rel)
            require(path.name.lower() not in PRIVATE_NAMES
                    and not path.name.lower().startswith(('private_chat', '已粘贴')),
                    'private source cannot be renamed into public scope')
            require(path.suffix.lower() in SAFE_SUFFIXES
                    or (path.suffix.lower() == '.log' and item['role'] == 'attempt')
                    or path.name.upper().startswith(('LICENSE', 'COPYING', 'NOTICE')),
                    'binary source cannot be renamed into a text public member')
            require(sha(path) == item['sha256'], 'selected original body changed: ' + rel)
            self.files[rel] = {**item, 'path': path, 'source_sha256': item['sha256']}

    def path(self, rel):
        rel = safe_relative(rel)
        require(rel in self.files, 'unindexed source dependency: ' + rel)
        return self.files[rel]['path']

    def doc(self, rel):
        return json.loads(self.path(rel).read_text(encoding='utf-8-sig'))

    def original(self, rel, digest):
        require(self.files[safe_relative(rel)]['source_sha256'] == digest,
                'original source hash binding differs: ' + rel)

    def lines(self, rel):
        result = []
        with self.path(rel).open('rb') as f:
            for i, line in enumerate(f, 1):
                require(line.strip(), 'blank source JSONL row')
                result.append((json.loads(line), i, hashlib.sha256(line).hexdigest()))
        return result


class PublicSource:
    def __init__(self, root, index):
        self.root, self.files = Path(root), index['files']

    def path(self, rel):
        rel = safe_relative(rel)
        require(rel in self.files, 'unindexed public dependency: ' + rel)
        return self.root / rel

    def doc(self, rel):
        return json.loads(self.path(rel).read_text(encoding='utf-8-sig'))

    def original(self, rel, digest):
        require(self.files[safe_relative(rel)]['source_sha256'] == digest,
                'original source identity differs: ' + rel)

    def lines(self, rel):
        entry = self.files[safe_relative(rel)]
        require(entry.get('line_identities_file') in self.files, 'original JSONL line identities missing')
        result = []
        with self.path(rel).open('rb') as f, self.path(entry['line_identities_file']).open(encoding='utf-8') as ids:
            for i, line in enumerate(f, 1):
                ident = json.loads(ids.readline())
                row = json.loads(line)
                require(ident['line'] == i and ident.get('id') == row.get('id')
                        and ident['public_sha256'] == hashlib.sha256(line).hexdigest(), 'JSONL public row identity differs')
                result.append((row, i, ident['source_sha256']))
            require(not ids.readline(), 'extra JSONL identities')
        return result


@contextmanager
def metric_modules(source):
    names = ('metrics', 'vendor_textvqa_metric', 'score', 'round12_report_v1')
    paths = [source.path('code/' + name + '.py') for name in names]
    require(len({p.parent for p in paths}) == 1, 'portable metric files must share code directory')
    previous = {n: sys.modules[n] for n in names if n in sys.modules}
    old = sys.path[:]
    try:
        for n in names:
            sys.modules.pop(n, None)
        sys.path.insert(0, str(paths[0].parent))
        yield (importlib.import_module('metrics'), importlib.import_module('score'),
               importlib.import_module('round12_report_v1'))
    finally:
        sys.path[:] = old
        for n in names:
            sys.modules.pop(n, None)
        sys.modules.update(previous)


def near(a, b, name, tolerance=1e-12):
    if a is None or b is None:
        require(a is None and b is None, 'null/full-coverage value differs: ' + name)
    elif isinstance(a, (list, tuple)):
        require(isinstance(b, (list, tuple)) and len(a) == len(b), 'vector differs: ' + name)
        for x, y in zip(a, b):
            near(x, y, name, tolerance)
    else:
        require(math.isfinite(float(a)) and math.isfinite(float(b))
                and abs(float(a) - float(b)) <= tolerance, 'numeric value differs: ' + name)


def physical_row(row, *, fixture=False, reused=False):
    own = row.get('owner_before_suffix')
    require(isinstance(own, dict) and isinstance(own.get('tensors'), dict), 'actual physical owner table missing')
    categories = Counter()
    seen = {}
    sizes = {'torch.float16': 2, 'torch.bfloat16': 2, 'torch.float32': 4,
             'torch.float64': 8, 'torch.int64': 8, 'torch.int32': 4,
             'torch.int16': 2, 'torch.uint8': 1, 'torch.int8': 1, 'torch.bool': 1}
    for name, item in own['tensors'].items():
        logical = math.prod(item['shape']) * sizes[item['dtype']]
        require(item['logical_bytes'] == logical and item['storage_bytes'] >= logical, 'tensor logical capacity differs')
        charge, other = item['charged_bytes'], item['shared_with']
        if other is not None:
            require(other in seen and charge == 0 and seen[other]['storage_bytes'] == item['storage_bytes']
                    and seen[other]['device'] == item['device'], 'invalid unique-storage alias')
        else:
            require(charge == item['storage_bytes'], 'uncharged unique storage')
        seen[name] = item
        leaf = name.rsplit('.', 1)[-1]
        if name == 'original_visual_positions' or name.startswith('common.'):
            categories['positions_bytes'] += charge
        elif '.visual.' in name:
            if leaf == 'positions':
                categories['positions_bytes'] += charge
            elif 'encoder_constants' in name or leaf in ('k_lut', 'v_lut'):
                categories['encoder_constants_bytes'] += charge
            elif leaf in ('k_code', 'v_code', 'k_packed', 'v_packed', 'key_mse_indices',
                          'key_qjl_signs', 'value_codes', 'keys', 'values', 'k_tail', 'v_tail'):
                categories['stored_visual_payload_bytes'] += charge
            else:
                categories['coder_metadata_bytes'] += charge
        elif name.endswith(('text_keys', 'text_values')):
            categories['text_kv_bytes'] += charge
        elif leaf == 'text_positions':
            categories['text_positions_bytes'] += charge
        else:
            categories['other_persistent_metadata_bytes'] += charge
    require(sum(categories.values()) == own['total_bytes'], 'physical total charge differs')
    if reused and own.get('accounting_scope') == 'native_full_allocation_equals_disjoint_visual_plus_text_regions':
        require(row.get('original_full_prefix_released') == 'native_FP16_is_the_declared_owner'
                and len(seen) == 56 and all(k.endswith(('.K','.V')) for k in seen),
                'native FP16 regions require actual56 K/V allocations')
        lengths = set()
        for tensor in seen.values():
            shape=tensor['shape']
            require(tensor['dtype']=='torch.float16' and len(shape)==4 and shape[:2]==[1,4]
                    and shape[3]==128 and tensor['shared_with'] is None,
                    'native FP16 allocation geometry differs')
            lengths.add(shape[2])
        require(len(lengths)==1 and own['total_bytes']==57344*next(iter(lengths)),
                'native FP16 complete sequence capacity differs')
        visual=57344*row['actual_visual_token_count']
        require(visual<=own['total_bytes'], 'native visual region exceeds actual allocation')
        categories=Counter(stored_visual_payload_bytes=visual,
                           text_kv_bytes=own['total_bytes']-visual)
    for name, value in categories.items():
        require(own.get(name, 0) == value, 'physical field classification differs: ' + name)
    complete = sum(categories[n] for n in ('positions_bytes', 'encoder_constants_bytes',
                    'stored_visual_payload_bytes', 'coder_metadata_bytes'))
    require(complete == own['stored_visual_KV_bytes'] == own['complete_visual_owner_bytes']
            == row['complete_visual_owner_bytes'] == row['stored_visual_KV_bytes'], 'complete visual owner differs')
    n = row['actual_visual_token_count']
    require(type(n) is int and n > 0 and row['full_kv_byte_denominator'] == 57344 * n,
            'M1 visual byte denominator differs')
    near(row['actual_effective_visual_bits'], 16 * complete / (57344 * n), 'effective bits')
    if not fixture:
        native = reused and row.get('original_full_prefix_released') == 'native_FP16_is_the_declared_owner'
        require(row.get('dtype') == 'float16' and row.get('model_id') == 'M1'
                and (native or row.get('original_full_prefix_released') is True)
                and (reused or row.get('original_FP16_visual_alias_retained') is False),
                'actual M1/source-release proof missing')
        require(row.get('peak_allocated_bytes', 0) >= 15888747520,
                'full-model physical peak proof missing')
    bridge = row.get('r7_bridge', {})
    detail = bridge.get('round12_static')
    if detail:
        require(detail.get('encoding_applied') is True and detail.get('native_source_retained') is False
                and detail['actual_owner_bytes'] == detail['expected_complete_owner_bytes'] == complete,
                'actual static encoding/owner proof differs')
        original = list(map(tuple, detail['original43_selected_stages']))
        actual = list(map(tuple, detail['selected_stages']))
        require(len(original) == len(set(original)) == 43, 'original concrete43 differs')
        if detail['mode'] != 'R':
            require(actual == original, 'fixed concrete43 order differs')
        else:
            require(len(actual) >= 43 and actual[:43] == original, 'R does not preserve original43 prefix')
    return complete


def producer_binding(cell, source, manifest, complete, *, fixture=False):
    p = cell.get('producer')
    if isinstance(p, dict) and p.get('kind') in ('legacy_source_proof', 'legacy_derived_view'):
        from round12_public_legacy_v1 import binding
        source.path('ops/round12_public_legacy_v1.py')
        binding(cell, source, manifest, complete)
        return
    require(isinstance(p, dict) and p.get('kind') in ('standalone', 'sequence_child'), 'actual producer binding required')
    launch, job, inventory, cfg = [source.doc(p[n]) for n in ('launch', 'job', 'snapshot_inventory', 'source_config')]
    source.original(p['source_config'], manifest['config_sha256'])
    require(cfg == manifest['config'], 'actual producer config differs')
    source.original(p['job'], launch['job_sha256'])
    source.original(p['snapshot_inventory'], launch['source_inventory_sha256'])
    require(launch['name'] == job['name'] and launch.get('source_snapshot'), 'launch/job identity differs')
    graph = {x['logical']: x['file'] for x in cell.get('source_graph', [])}
    require(len(graph) == len(cell.get('source_graph', [])), 'duplicate source graph key')
    for logical, digest in manifest.get('source_hashes', {}).items():
        require(inventory.get(logical) == digest and logical in graph, 'snapshot/runtime graph differs: ' + logical)
        source.original(graph[logical], digest)
    script_argument = next((x for x in job.get('argv', []) if isinstance(x, str) and x.endswith('.py')), None)
    require(script_argument is not None, 'actual launch Python script argument missing')
    script_keys = [key for key in inventory if script_argument.replace('\\', '/').endswith('/' + key)
                   or script_argument.replace('\\', '/') == key]
    require(len(script_keys) == 1 and script_keys[0] in graph, 'launched driver not bound to archived snapshot source')
    source.original(graph[script_keys[0]], inventory[script_keys[0]])
    legacy = dict(complete.get('executed_legacy_source_sha256', {}))
    for row, _, _ in source.lines(cell['paths']['records']):
        for bridge in (row.get('r7_bridge', {}), row.get('low_stage', {}).get('r7_bridge', {})):
            for logical, value in bridge.get('legacy_sources', {}).items():
                digest = value.get('sha256') if isinstance(value, dict) else value
                require(logical not in legacy or legacy[logical] == digest, 'legacy source changed between rows')
                legacy[logical] = digest
    for logical, digest in legacy.items():
        require(logical in graph, 'executed legacy source not archived: ' + logical)
        source.original(graph[logical], digest)
    if not fixture:
        assets = {x['logical']: x['file'] for x in cell.get('asset_graph', [])}
        cfg = manifest['config']
        demands = {}
        variant = cfg.get('variant', {})
        for key, digest in cfg.get('expected_asset_sha256', {}).items():
            if key in variant:
                demands[variant[key]] = digest
        origin = cfg.get('variant_source', {})
        if origin.get('path') and origin.get('sha256'):
            demands[origin['path']] = origin['sha256']
        if cfg.get('model_source_lock') and manifest.get('model_identity', {}).get('source_lock_sha256'):
            demands[cfg['model_source_lock']] = manifest['model_identity']['source_lock_sha256']
        candidate = cfg.get('candidate', {})
        for stem in ('input_manifest', 'selected_algorithm', 'selection_receipt'):
            if cfg.get(stem + '_path') and cfg.get(stem + '_sha256'):
                demands[cfg[stem + '_path']] = cfg[stem + '_sha256']
        for stem in ('key_book', 'ratio'):
            if candidate.get(stem + '_path'):
                demands[candidate[stem + '_path']] = candidate[stem + '_sha256']
        for row, _, _ in source.lines(cell['paths']['records']):
            for bridge in (row.get('r7_bridge', {}), row.get('low_stage', {}).get('r7_bridge', {})):
                for value in bridge.get('fixed_assets', {}).values():
                    if isinstance(value, dict) and 'path' in value and 'sha256' in value:
                        demands[value['path']] = value['sha256']
        for logical, digest in demands.items():
            require(logical in assets, 'actual book/factor/policy asset not archived: ' + logical)
            source.original(assets[logical], digest)
        if manifest.get('runner_schema') == 'trajkv.round12.fixed500_static.raw.v1':
            selected = source.doc(assets[cfg['selected_algorithm_path']])
            require(selected.get('candidate') == cfg.get('candidate')
                    and selected.get('name') == cfg['selected_method_name'], 'frozen selected algorithm differs')
            for logical, digest in selected['algorithm_source_hashes'].items():
                require(logical in graph, 'selected algorithm source absent')
                source.original(graph[logical], digest)
    if not fixture:
        require(manifest.get('source_hashes'), 'actual runtime source graph missing')
        process = manifest.get('runtime_process', {})
        if process:
            require(process.get('source_snapshot_root') == launch['source_snapshot'], 'executed snapshot path differs')
    if p['kind'] == 'standalone':
        require(any(str(x).replace('\\', '/').rsplit('/', 1)[-1] == Path(p['source_config']).name
                    for x in job['argv']), 'standalone job does not reference actual config')
    else:
        seq, done = source.doc(p['sequence_config']), source.doc(p['parent_complete'])
        require(any(str(x).replace('\\', '/').rsplit('/', 1)[-1] == Path(p['sequence_config']).name
                    for x in job['argv']), 'parent job does not reference actual sequence config')
        entries = [x for x in seq['entries'] if x['name'] == p['cell_name']]
        require(len(entries) == 1, 'sequence cell identity missing/duplicate')
        entry = entries[0]
        require(entry['script'] in graph and inventory.get(entry['script']) == entry['script_sha256'],
                'child script differs from actual snapshot')
        source.original(graph[entry['script']], entry['script_sha256'])
        require(entry['expected_QA'] == manifest['expected_QA']
                and entry['config_sha256'] == manifest['config_sha256'], 'sequence config/count differs')
        source.original(p['sequence_config'], done['plan_sha256'])
        actual = [x for x in done.get('completed_cells', []) if x['name'] == p['cell_name']]
        if not actual and done.get('failed_cell', {}).get('name') == p['cell_name']:
            actual = [done['failed_cell']]
            require(cell['disposition'] == 'CLOSED_WITH_FAILURES' and actual[0]['exit_code'] != 0,
                    'failed parent child cannot become complete success')
        else:
            require(len(actual) == 1 and actual[0]['exit_code'] == 0, 'parent did not record completed child')
        if actual[0].get('complete_sha256'):
            source.original(cell['paths']['complete'], actual[0]['complete_sha256'])


def validate_cell(cell, source, metric, scorer, *, fixture=False, require_producer=True):
    require(cell.get('kind') in ('new_round12', 'pilot', 'reused_reference'), 'cell origin must be explicit')
    excluded = cell.get('disposition') == 'CLOSED_PILOT_EXCLUDED'
    require(cell.get('disposition') in ('CLOSED_COMPLETE', 'CLOSED_WITH_FAILURES', 'CLOSED_PILOT_EXCLUDED'),
            'pending or unsupported cell disposition')
    require(not excluded or (cell['kind'] == 'pilot' and cell.get('quality_exclusion_reason')
                             and cell.get('include_in_quality') is False), 'pilot exception must be explicit and excluded')
    paths = cell['paths']
    m, done = source.doc(paths['manifest']), source.doc(paths['complete'])
    raw_lines = source.lines(paths['records'])
    rows = [x[0] for x in raw_lines]
    expected = m['expected_ids']
    derived_reuse = (cell['kind'] == 'reused_reference'
                     and m.get('schema') == 'baseline_matrix.derived_prediction_view.v1'
                     and cell.get('producer', {}).get('kind') == 'legacy_derived_view')
    require((m.get('schema') == 'baseline_matrix.raw_inference.v1' or derived_reuse)
            and m.get('references_read') is False
            and (m.get('prediction_cache_used') is False or derived_reuse)
            and done.get('quality_scored') is False,
            'raw inference must be reference-free and unscored')
    require(fixture or not m.get('fixture_only', False), 'CPU fixture producer cannot enter actual scope')
    require(not (set(m.get('config', {})) & {'refs', 'references', 'answers', 'labels',
            'reference_path', 'reference_answers', 'scores'}), 'inference config contains references/labels')
    require(len(expected) == len(set(expected)) == len(rows) == m['expected_QA']
            == done['expected_QA'] == done['recorded_QA']
            and [r['id'] for r in rows] == expected, 'actual expected raw coverage differs')
    require(dict(Counter(r['status'] for r in rows)) == done['counts'], 'raw failure/status accounting differs')
    finished = {'PASS', 'UNSUPPORTED_MODEL', 'HARDWARE_UNSUPPORTED', 'OOM_AT_DECLARED_INPUT',
                'METHOD_PORT_UNVERIFIED', 'INPUT_MISMATCH', 'OWNER_AUDIT_FAILED', 'FAILED_RUN', 'STALE_PROTOCOL'}
    require(all(r['status'] in finished for r in rows), 'unfinished or unknown task status')
    if not excluded:
        all_pass = all(r['status'] == 'PASS' for r in rows)
        require(cell['disposition'] == ('CLOSED_COMPLETE' if all_pass else 'CLOSED_WITH_FAILURES')
                and done['status'] == ('PASS_RAW_COMPLETE' if all_pass else 'COMPLETE_WITH_FAILURES'),
                'cell disposition or complete status hides failed tasks')
    require(excluded or done.get('coverage_complete') is True, 'producer did not close coverage')
    source.original(paths['manifest'], done['manifest_sha256'])
    source.original(paths['records'], done['records_jsonl_sha256'])
    if not derived_reuse:
        receipt = source.doc(paths['remote_receipt'])
        require(receipt.get('remote_hashes_computed_independently') is True, 'independent remote readback missing')
        if not fixture:
            require(receipt.get('status') in ('PASS_REMOTE_RAW_SHA256_EXACT',
                        'PASS_CONFIRMATION_EXECUTED_SOURCE_AND_RAW_SHA'),
                    'actual independent producer readback status required')
        inventory = {r['file']: r for r in receipt['files']}
        for field in CORE_FILES:
            name = {'manifest': 'manifest.json', 'records': 'records.jsonl', 'complete': 'complete.json'}[field]
            require(name in inventory, 'readback triplet member missing')
            source.original(paths[field], inventory[name]['sha256'])
            entry = source.files[paths[field]]
            require(entry.get('source_bytes', entry.get('bytes', source.path(paths[field]).stat().st_size))
                    == inventory[name]['bytes'], 'readback member byte count differs')
    if cell['kind'] == 'new_round12':
        require(m.get('runner_schema') in ('trajkv.round12_research.raw.v1', 'trajkv.round12_action.raw.v1',
                                          'trajkv.round12_action.raw.v2', 'trajkv.round12_oracle.raw.v1',
                                          'trajkv.round12.fixed500_static.raw.v1', 'trajkv.round12_budget_action.raw.v3'),
                'actual R12 runner schema differs')
        if m.get('runner_schema') == 'trajkv.round12_budget_action.raw.v3':
            require(m.get('round12_budget_action_version') == 3
                    and done.get('round12_budget_action_version') == 3
                    and done.get('all_PASS_sources_verified_strict_total_KV_budget') is True,
                    'budget action must retain actual v3 closure and strict KV budget proof')
        if m.get('runner_schema') == 'trajkv.round12.fixed500_static.raw.v1':
            cfg = m['config']
            require(cfg.get('input_role') == 'round12_fixed500_comparison'
                    and cfg.get('input_count') == 500
                    and cfg.get('selected_method_name') == 'TrajKV_R12_R_G8matched'
                    and cfg.get('selected_algorithm_sha256') and cfg.get('selection_receipt_sha256'),
                    'fixed500 candidate must retain frozen algorithm and exact comparison role')
        if m.get('runner_schema') == 'trajkv.round12_action.raw.v2':
            require(m.get('round12_action_closure_version', 0) >= 3
                    and done.get('round12_action_closure_version', 0) >= 3,
                    'formal action requires actual runtime v3 coverage closure')
    if require_producer:
        producer_binding(cell, source, m, done, fixture=fixture)
    if not fixture and cell['kind'] != 'reused_reference':
        require(m.get('model_identity', {}).get('revision') == '259a3b64a14623c0ec91a045cb43f7c5af5fa6af'
                and m.get('model_residency', {}).get('weight_resident_bytes') == 15888747520,
                'full frozen M1 model identity/residency required')
    success = sum(r['status'] == 'PASS' and bool(r.get('output_token_ids')) for r in rows)
    untrusted = sum(r['status'] != 'PASS' and bool(r.get('output_token_ids') or r.get('untrusted_generated_output_token_ids')) for r in rows)
    if excluded:
        return {'cell_id': cell['cell_id'], 'kind': cell['kind'], 'dataset': cell['dataset'],
                'selected_rows': {}, 'producer_rows': len(rows), 'successful_terminal_answers': success,
                'untrusted_generated_answer_rows': untrusted, 'failure_status_rows': len(rows) - success,
                'primary_mean': None, 'pilot_excluded': True}
    refs_lines, inputs_lines = source.lines(paths['refs']), source.lines(paths['inputs'])
    refs, inputs = ({r['id']: r for r, _, _ in x} for x in (refs_lines, inputs_lines))
    lock = source.doc(paths['source_lock'])
    for name in ('refs', 'inputs'):
        filename = PurePosixPath(paths[name]).name
        require(filename in lock['files'], 'frozen data source-lock file absent')
        source.original(paths[name], lock['files'][filename]['sha256'])
    if cell['kind'] != 'reused_reference':
        source.original(paths['inputs'], m['inputs_sha256'])
        cfg = m['config']
        selected = cfg.get('ids') or [r[0]['id'] for r in inputs_lines][cfg['start']:cfg['start'] + cfg['count']]
        require(expected == selected, 'input selection over locked cohort differs')
    quality_ids = cell.get('quality_ids', expected)
    require(len(quality_ids) == len(set(quality_ids)) and set(quality_ids) <= set(expected)
            and set(quality_ids) <= set(refs) and set(quality_ids) <= set(inputs), 'selected view reference coverage differs')
    scored_lines = source.lines(paths['scored_records'])
    scored = {r['id']: r for r, _, _ in scored_lines}
    require(len(scored) == len(scored_lines) and set(quality_ids) <= set(scored), 'scored view coverage differs')
    summary = source.doc(paths['scored_summary'])
    if 'derived_file_hashes' in summary:
        require(fixture or summary.get('integrity', {}).get('fixture_only') is False,
                'fixture scoring cannot become actual task quality')
        source.original(paths['scored_records'], summary['derived_file_hashes']['scored_records.jsonl'])
        require(summary['metric_contract_version'] == metric.METRIC_CONTRACT_VERSION,
                'official scoring version differs')
        for key in CORE_FILES:
            filename = {'manifest': 'manifest.json', 'records': 'records.jsonl', 'complete': 'complete.json'}[key]
            source.original(paths[key], summary['integrity']['raw_files'][filename])
    elif cell['kind'] == 'reused_reference':
        source.original(paths['scored_records'], summary['scored_records_sha256'])
    else:
        require(False, 'new R12 requires frozen official-scored summary')
    admitted = {}
    for raw, number, line_sha in raw_lines:
        if raw['id'] not in quality_ids:
            continue
        ident, score = raw['id'], scored[raw['id']]
        ref, inp = refs[ident], inputs[ident]
        for key in ('id', 'dataset', 'source_cluster_id', 'original_image_sha256', 'question_sha256'):
            require(raw.get(key) == inp.get(key), 'raw/frozen source mismatch: ' + key)
        require(raw['dataset'] == cell['dataset'] and ref['source_cluster_id'] == raw['source_cluster_id'],
                'source/domain/reference mismatch')
        bound_number = score.get('raw_jsonl_line_number', score.get('raw_line_number'))
        bound_sha = score.get('raw_jsonl_exact_line_sha256', score.get('raw_line_sha256'))
        require(bound_number == number and bound_sha == line_sha
                and score['raw_prediction'] == raw.get('raw_prediction')
                and score.get('output_token_ids') == raw.get('output_token_ids'), 'scored row not bound to exact original raw line')
        if raw['status'] == 'PASS':
            require(raw.get('output_token_ids') and isinstance(raw.get('raw_prediction'), str)
                    and raw.get('official_score') is None, 'PASS is not genuine unscored generation')
            actual = metric.score_question(raw['dataset'], raw['raw_prediction'], ref)
            require(score['score_status'] == 'SCORED_PASS' and actual['metric_id'] == score['metric_id'], 'scored metric differs')
            near(actual['score'], score['score'], 'frozen primary score')
            physical_row(raw, fixture=fixture, reused=cell['kind'] == 'reused_reference')
        else:
            require(score['score_status'] != 'SCORED_PASS' and score.get('score') is None,
                    'failed task removed or given a fake primary score')
        # Owner tensors and source traces have been verified above; retaining
        # them again in every aggregate view is unnecessary and makes portable
        # full500 verification consume several gigabytes of duplicate dicts.
        merged = {**raw, **score}
        compact_fields = ('id','dataset','status','source_cluster_id','score','score_status','metric_id',
                          'stored_visual_KV_bytes','actual_effective_visual_bits','encode_ms','wall_ms',
                          'ttft_ms','peak_allocated_bytes')
        admitted[ident] = {k:merged[k] for k in compact_fields if k in merged}
    complete_quality = all(r['score_status'] == 'SCORED_PASS' for r in admitted.values())
    mean = sum(float(r['score']) for r in admitted.values()) / len(admitted) if complete_quality else None
    if cell['kind'] != 'reused_reference' and quality_ids == expected:
        near(mean, summary['primary_mean'], 'full producer primary')
        expected_ci = scorer.cluster_bootstrap(list(admitted.values()), 'score',
                summary['bootstrap']['repetitions'], summary['bootstrap']['seed'])['ci95'] if complete_quality else None
        near(expected_ci, summary['primary_ci95'], 'official primary interval')
    return {'cell_id': cell['cell_id'], 'kind': cell['kind'], 'dataset': cell['dataset'],
            'selected_rows': admitted, 'producer_rows': len(rows),
            'successful_terminal_answers': success, 'untrusted_generated_answer_rows': untrusted,
            'failure_status_rows': len(rows) - success, 'primary_mean': mean, 'pilot_excluded': False}


def verify_semantics(scope, source, *, allow_fixture=False, require_producer=True):
    fixture = check_header(scope, allow_fixture=allow_fixture)
    # Explicit metric support is small, pure CPU and must itself be indexed.
    for rel in ('code/metrics.py', 'code/vendor_textvqa_metric.py', 'code/score.py', 'code/round12_report_v1.py'):
        source.path(rel)
    for rel, digest in FROZEN_METRIC_SOURCES.items():
        source.original(rel, digest)
    totals, seen_producers, admitted = Counter(), set(), {}
    with metric_modules(source) as (metric, scorer, reporter):
        for cell in scope['cells']:
            result = validate_cell(cell, source, metric, scorer, fixture=fixture,
                                   require_producer=require_producer)
            triple = tuple(source.files[cell['paths'][k]]['source_sha256'] for k in CORE_FILES)
            if cell['kind'] != 'reused_reference':
                require(triple not in seen_producers, 'fresh/pilot producer counted more than once')
                seen_producers.add(triple)
                prefix = 'pilot_' if cell['kind'] == 'pilot' else 'new_'
                for name in ('producer_rows', 'successful_terminal_answers', 'untrusted_generated_answer_rows', 'failure_status_rows'):
                    totals[prefix + name] += result[name]
            else:
                totals['reused_selected_view_rows'] += len(result['selected_rows'])
            admitted[cell['cell_id']] = result
        views = {}
        quality = []
        for view in scope.get('views', []):
            name = view['view_id']
            require(name not in views, 'duplicate quality view')
            values = {}
            for member in view['members']:
                cell = admitted[member['cell_id']]
                require(not cell['pilot_excluded'], 'excluded pilot cannot re-enter quality view')
                chosen = member.get('ids', list(cell['selected_rows']))
                for ident in chosen:
                    require(ident in cell['selected_rows'] and ident not in values, 'duplicate/missing quality view ID')
                    values[ident] = cell['selected_rows'][ident]
            require(values and all(r['dataset'] == view['dataset'] for r in values.values()), 'mixed/empty quality domain')
            complete = all(r['score_status'] == 'SCORED_PASS' for r in values.values())
            mean = sum(float(r['score']) for r in values.values()) / len(values) if complete else None
            q = {'view_id': name, 'label': view['label'], 'dataset': view['dataset'], 'questions': len(values),
                 'source_clusters': len({r['source_cluster_id'] for r in values.values()}), 'primary_mean': mean}
            for field in ('stored_visual_KV_bytes', 'actual_effective_visual_bits', 'encode_ms', 'wall_ms',
                          'ttft_ms', 'peak_allocated_bytes'):
                vs = [float(r[field]) for r in values.values() if isinstance(r.get(field), (int, float))]
                q[field] = ({'n': len(vs), 'complete_coverage': len(vs) == len(values),
                    'mean': statistics.mean(vs) if len(vs) == len(values) else None,
                    'median': statistics.median(vs) if len(vs) == len(values) else None,
                    'max': max(vs) if len(vs) == len(values) else None} if vs else None)
            views[name] = values
            quality.append(q)
        pairs = []
        for pair in scope.get('comparisons', []):
            result = reporter.paired(views[pair['candidate_view']], views[pair['reference_view']],
                                    pair.get('bootstrap', 2000), pair.get('seed', 20261124))
            for key in ('primary_delta_pp', 'ci95_pp'):
                if key in pair:
                    near(result[key], pair[key], 'declared paired ' + key)
            pairs.append({**pair, 'recomputed': result})
    if scope.get('report_summary'):
        report = source.doc(scope['report_summary'])
        by_key = {(r['label'], r['dataset']): r for r in quality}
        require(len(by_key) == len(quality), 'report quality label/domain is ambiguous')
        for row in report['quality']:
            require((row['label'], row['dataset']) in by_key, 'report has an undeclared quality view')
            actual = by_key[(row['label'], row['dataset'])]
            require(row['questions'] == actual['questions'], 'report denominator differs')
            near(row['primary_mean'], actual['primary_mean'], 'report quality')
            for field in ('stored_visual_KV_bytes', 'actual_effective_visual_bits', 'encode_ms', 'wall_ms',
                          'ttft_ms', 'peak_allocated_bytes'):
                if field in row:
                    if row[field] is None:
                        require(actual[field] is None, 'report resource omission differs')
                    else:
                        require(row[field]['n'] == actual[field]['n']
                                and row[field]['complete_coverage'] == actual[field]['complete_coverage'],
                                'report resource denominator differs')
                        for key in ('mean', 'median', 'max'):
                            near(row[field][key], actual[field][key], 'report resource ' + field)
        require(len(report['quality']) == len(quality), 'report omits quality view')
        view_headers = {v['view_id']: v for v in scope.get('views', [])}
        declared_pairs = {}
        for pair in pairs:
            candidate, reference = view_headers[pair['candidate_view']], view_headers[pair['reference_view']]
            require(candidate['dataset'] == reference['dataset'], 'paired views cross domains')
            key = (candidate['label'], reference['label'], candidate['dataset'])
            require(key not in declared_pairs, 'duplicate report pair')
            declared_pairs[key] = pair['recomputed']
        require(len(report.get('paired', [])) == len(declared_pairs), 'report paired comparison count differs')
        for row in report.get('paired', []):
            key = (row['candidate'], row['reference'], row['dataset'])
            require(key in declared_pairs, 'report has undeclared pair')
            actual = declared_pairs[key]
            require(row['n'] == actual['n'] and row['status'] == actual['status'], 'paired report denominator/status differs')
            near(row['primary_delta_pp'], actual['primary_delta_pp'], 'paired report delta')
            near(row['ci95_pp'], actual['ci95_pp'], 'paired report interval')
    qa0 = []
    for item in scope.get('qa0', []):
        obj = source.doc(item['file'])
        require(item['kind'] in ('packed_cuda', 'calibration', 'model_TF'), 'unknown QA0 kind')
        if not fixture:
            p=item.get('producer',{})
            require({'launch','job','snapshot_inventory','entrypoint'}<=set(p), 'actual QA0 producer binding omitted')
            launch,job,inventory=(source.doc(p[k]) for k in ('launch','job','snapshot_inventory'))
            source.original(p['job'],launch['job_sha256'])
            source.original(p['snapshot_inventory'],launch['source_inventory_sha256'])
            driver=next((x for x in job['argv'] if isinstance(x,str) and x.endswith('.py')),None)
            keys=[k for k in inventory if driver and driver.replace('\\','/').endswith('/'+k)]
            require(len(keys)==1, 'actual QA0 driver not in snapshot inventory')
            source.original(p['entrypoint'],inventory[keys[0]])
            available={source.files[r]['source_sha256'] for r in p.get('support_files',[])+[p['entrypoint']]}
            for logical,digest in obj.get('source_sha256',{}).items():
                key=logical if '/' in logical else 'code/'+logical
                require(inventory.get(key)==digest and digest in available, 'QA0 source body/snapshot differs')
            for digest in {**obj.get('legacy_source_sha256',{}),**obj.get('executed_legacy_source_sha256',{})}.values():
                require(digest in available, 'QA0 executed legacy source omitted')
            for digest in {**obj.get('locks',{}),**obj.get('locked_config_and_asset_sha256',{})}.values():
                require(digest in available, 'QA0 locked input/config/asset body omitted')
        if item['kind'] == 'packed_cuda':
            require(obj.get('actual_CUDA') is True and obj.get('task_answer_events') == 0,
                    'packed QA0 cannot imply task answers')
            totals['QA0_cuda_attention_cases'] += len(obj.get('attention_checks', obj.get('attention_cases', [])))
        elif item['kind'] == 'calibration':
            require(obj.get('new_free_QA') == 0 and obj.get('native_prefix_model_forwards', 0) >= 0,
                    'calibration task/TF accounting differs')
            totals['QA0_native_prefix_model_forwards'] += obj['native_prefix_model_forwards']
        else:
            require(item.get('new_free_QA') == 0, 'TF must have explicit zero free answers')
            totals['QA0_model_TF_forwards'] += int(item.get('actual_model_forwards', 0))
        qa0.append({'kind': item['kind'], 'file': item['file'], 'new_free_QA': 0})
    attempts = []
    attempt_ids = set()
    for item in scope.get('attempts', []):
        require(item.get('attempt_id') not in attempt_ids, 'duplicate technical/setup attempt')
        attempt_ids.add(item['attempt_id'])
        require(item.get('disposition') in ('PRE_MODEL_FAILED', 'NOT_LAUNCHED', 'TECHNICAL_FAILED', 'CLOSED_ABORTED')
                and item.get('reason') and item.get('files'), 'unclosed or unexplained attempt')
        for rel in item['files']:
            source.path(rel)
        require(item.get('successful_terminal_answers') == 0,
                'terminal-answer producer must be a cell, not hidden among setup attempts')
        if item['disposition'] in ('PRE_MODEL_FAILED', 'NOT_LAUNCHED'):
            require(item.get('actual_model_forwards') == 0 and item.get('recorded_task_rows') == 0,
                    'pre-model/unlaunched attempt cannot imply model/task work')
        totals['closed_non_quality_attempts'] += 1
        attempts.append({k: item[k] for k in ('attempt_id', 'disposition', 'reason', 'successful_terminal_answers')})
    for key, expected in scope.get('expected_accounting', {}).items():
        require(totals[key] == expected, 'Root actual event accounting differs: ' + key)
    return {'status': 'PASS_R12_FIXTURE_SEMANTICS' if fixture else 'PASS_R12_ACTUAL_LOCAL_SEMANTICS',
            'fixture_only': fixture, 'publication_eligible': (not fixture and require_producer
                and scope.get('stage_state') == 'ROOT_SEALED' and bool(scope.get('root_seal'))),
            'counts': dict(totals), 'quality': quality, 'paired': pairs,
            'QA0': qa0, 'attempts': attempts,
            'physical_rows_verified': sum(sum(r['status'] == 'PASS' for r in x['selected_rows'].values())
                                                    for x in admitted.values()),
            'actual_producer_binding_required': require_producer,
            'private_tensors_or_GPU_replayed': False, 'whole_external_library_completion_claimed': False}
