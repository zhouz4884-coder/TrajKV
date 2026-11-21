"""Exact-contract quality-prediction reuse; no model/metric/GPU imports.

Source runs remain immutable. A derived view copies original JSONL lines byte
for byte and records their source, rather than claiming a new GPU experiment.
Timing remains that of the original instrumented requests, never replay speed.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import time

ROOT = Path(__file__).resolve().parents[1]
RAW_FILES = ('manifest.json', 'records.jsonl', 'complete.json')
CORE = ('code/evaluate.py', 'code/ownership.py', 'code/cache_bridge.py', 'code/attention_bridge.py',
        'code/dataset_registry.py', 'code/model_registry.py')
INPUT_FIELDS = ('id', 'dataset', 'dataset_revision', 'split', 'source_id', 'source_cluster_id', 'question_id',
                'original_image_sha256', 'width', 'height', 'question', 'question_sha256', 'group')
KNOWN_CONFIG = {'schema', 'model_id', 'model_revision', 'model_path', 'model_source_lock', 'inputs', 'start',
                'count', 'ids', 'max_new_tokens', 'question_suffix', 'preprocessing', 'method', 'scope',
                'verify_question_unknown_prefix', 'stage_role'}
LEGACY_ADAPTER_PAIRS = (
    ('methods.kivi_style','KIVIStyleAdapter'),
    ('methods.calibquant_adapter','CalibQuantAdapter'),
    ('methods.calibquant_fp32_bias_adapter','CalibQuantFP32BiasAdapter'),
    ('methods.turboquant_adapter','TurboQuantAdapter'),
    ('methods.eviction_adapter','PyramidKVVisualAdapter'),
)
APPENDED_ADAPTER_PAIRS = (
    ('methods.kivi_example_adapter','KIVIExampleAdapter'),
    ('methods.turboquant_buffered_adapter','TurboQuantBufferedAdapter'),
    ('methods.turboquant_onlyterp_mse_adapter','OnlyTerpMSEAdapter'),
)
APPENDED_ADAPTER_PAIRS_V3 = APPENDED_ADAPTER_PAIRS + (
    ('methods.calibquant_fp32_text_adapter','CalibQuantFP32AffineTextAdapter'),
)
REVIEWED_POLICY_JSON_SHA256 = {
    'evaluate_identity_equivalence_v1.json':'2316ccdae4082157b5fac7f17fd3f27c2d36d150cb4e4f5435429f0eadaed684',
    'evaluate_identity_equivalence_v2.json':'c02f69a0c1819db611f9df57814030866cdd544baea1760583eeaeb216c8d112',
    'evaluate_identity_equivalence_v3.json':'0ea9512133985f41c6916b6a86b846f77f80457c2fff2af0a0dff9b81c5bc402',
}


class CacheIntegrityError(ValueError): pass
class CacheConflictError(CacheIntegrityError): pass


def require(condition, message):
    if not condition: raise CacheIntegrityError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024**2), b''): h.update(part)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def _ast_plain(node):
    if isinstance(node,ast.AST):
        return {'node':type(node).__name__,'fields':{key:_ast_plain(value) for key,value in ast.iter_fields(node)
                if not (key=='type_params' and value==[])}}
    if isinstance(node,list):return [_ast_plain(value) for value in node]
    if isinstance(node,bytes):return {'literal_bytes_hex':node.hex()}
    return node


def _evaluate_ast_identity_tree(tree):
    matches=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name=='model_identity']
    require(len(matches)==1,'Expected exactly one model_identity function')
    # Empty type_params was added to the Python AST after 3.10. Removing only
    # this empty syntax field makes the lock reproducible on CPU 3.10/3.13.
    identity_sha=canonical_sha(_ast_plain(matches[0]))
    index=tree.body.index(matches[0])
    tree.body[index]=ast.Expr(value=ast.Constant(value='REVIEWED_MODEL_IDENTITY_IO_ONLY_EQUIVALENCE_V1'))
    numerical_sha=canonical_sha(_ast_plain(tree))
    return identity_sha,numerical_sha


def evaluate_ast_identity(body):
    """V1 normalization is retained unchanged for archived source checks."""
    return _evaluate_ast_identity_tree(ast.parse(body.decode('utf-8')))


def evaluate_ast_registry_identity(body):
    """Strip only three fixed registry additions, preserving all legacy AST."""
    tree=ast.parse(body.decode('utf-8'))
    matches=[node for node in tree.body if isinstance(node,ast.Assign) and len(node.targets)==1
             and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='ALLOWLIST']
    require(len(matches)==1 and isinstance(matches[0].value,ast.Set),'One literal ALLOWLIST set is required')
    allowed=matches[0].value
    pairs=[]
    for item in allowed.elts:
        require(isinstance(item,ast.Tuple) and len(item.elts)==2 and all(isinstance(v,ast.Constant) and isinstance(v.value,str) for v in item.elts),
                'ALLOWLIST must contain literal module/class string pairs only')
        pairs.append(tuple(v.value for v in item.elts))
    require(len(pairs)==len(set(pairs)),'Duplicate literal registry entries are forbidden')
    require(set(LEGACY_ADAPTER_PAIRS).issubset(pairs),'Legacy adapter registry entries may not be removed')
    require(set(pairs).issubset(set(LEGACY_ADAPTER_PAIRS)|set(APPENDED_ADAPTER_PAIRS)),
            'Only the three specifically approved registry additions are compatible')
    allowed.elts=[item for item,pair in zip(allowed.elts,pairs) if pair not in APPENDED_ADAPTER_PAIRS]
    identity,numerical=_evaluate_ast_identity_tree(tree)
    return identity,numerical,pairs


def evaluate_ast_registry_identity_v3(body):
    """V3 strips only the four individually approved literal registry pairs."""
    tree=ast.parse(body.decode('utf-8'))
    matches=[node for node in tree.body if isinstance(node,ast.Assign) and len(node.targets)==1
             and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='ALLOWLIST']
    require(len(matches)==1 and isinstance(matches[0].value,ast.Set),'One literal ALLOWLIST set is required')
    allowed=matches[0].value
    pairs=[]
    for item in allowed.elts:
        require(isinstance(item,ast.Tuple) and len(item.elts)==2 and all(isinstance(v,ast.Constant) and isinstance(v.value,str) for v in item.elts),
                'ALLOWLIST must contain literal module/class string pairs only')
        pairs.append(tuple(v.value for v in item.elts))
    require(len(pairs)==len(set(pairs)),'Duplicate literal registry entries are forbidden')
    require(set(LEGACY_ADAPTER_PAIRS).issubset(pairs),'Legacy adapter registry entries may not be removed')
    require(set(pairs).issubset(set(LEGACY_ADAPTER_PAIRS)|set(APPENDED_ADAPTER_PAIRS_V3)),
            'Only the four specifically approved registry additions are compatible')
    allowed.elts=[item for item,pair in zip(allowed.elts,pairs) if pair not in APPENDED_ADAPTER_PAIRS_V3]
    identity,numerical=_evaluate_ast_identity_tree(tree)
    return identity,numerical,pairs


def evaluate_semantics(path, original_sha):
    body=Path(path).read_bytes()
    require(hashlib.sha256(body).hexdigest()==original_sha,'Original evaluate source proof mismatch')
    # Immutable inference snapshots copy methods/configs but deliberately not
    # every project configuration. Resolve a missing policy from the launcher's
    # data root, with its complete canonical JSON content locked below. This
    # does not weaken admission or require mutating a running queue's launcher.
    data_root=Path(os.environ.get('BASELINE_DATA_ROOT',str(ROOT)))
    candidates=[root/'configs/cache'/name for name in ('evaluate_identity_equivalence_v3.json','evaluate_identity_equivalence_v2.json','evaluate_identity_equivalence_v1.json')
                for root in (ROOT,data_root)]
    policy_path=next((p for p in candidates if p.is_file()),None)
    require(policy_path is not None,'Explicit identity I/O-equivalence policy is required')
    policy=read(policy_path)
    require(canonical_sha(policy)==REVIEWED_POLICY_JSON_SHA256[policy_path.name],
            'STALE_PROTOCOL: reviewed policy content changed; LF/JSON formatting alone is allowed')
    allowed=policy['allowed_sources'].get(original_sha)
    require(allowed is not None,'STALE_PROTOCOL: unreviewed evaluate.py source version')
    if policy['schema']=='baseline_matrix.reviewed_identity_registry_equivalence.v3':
        require(policy['legacy_adapter_pairs']==[list(p) for p in LEGACY_ADAPTER_PAIRS] and
                policy['approved_appended_pairs']==[list(p) for p in APPENDED_ADAPTER_PAIRS_V3],
                'Registry compatibility policy differs from the fixed v3 code review')
        identity_sha,numerical_sha,pairs=evaluate_ast_registry_identity_v3(body)
        require([list(p) for p in pairs]==allowed['literal_adapter_pairs'],'Whole-source registry declaration mismatch')
    elif policy['schema']=='baseline_matrix.reviewed_identity_registry_equivalence.v2':
        require(policy['legacy_adapter_pairs']==[list(p) for p in LEGACY_ADAPTER_PAIRS] and
                policy['approved_appended_pairs']==[list(p) for p in APPENDED_ADAPTER_PAIRS],
                'Registry compatibility policy differs from the fixed code review')
        identity_sha,numerical_sha,pairs=evaluate_ast_registry_identity(body)
        require([list(p) for p in pairs]==allowed['literal_adapter_pairs'],'Whole-source registry declaration mismatch')
    else:identity_sha,numerical_sha=evaluate_ast_identity(body)
    require(identity_sha==allowed['model_identity_ast_sha256'] and numerical_sha==policy['numerical_program_ast_sha256'],
            'STALE_PROTOCOL: source changed outside the reviewed I/O function and fixed registry additions')
    # Keep the original semantic key for existing branches. The wider source
    # whitelist is an admission proof, not a change to their numerical program.
    return {'equivalence':'model_identity_IO_only_reviewed_v1','numerical_program_ast_sha256':numerical_sha}


def resolve_run(path):
    path = Path(path)
    pointer = path / 'artifact_pointer.json'
    if pointer.exists(): path = Path(read(pointer)['local_raw_dir'])
    require(path.is_dir(), 'Raw run missing: ' + str(path))
    return path.resolve()


def normalize_behavior(config, spec):
    unknown = set(config) - KNOWN_CONFIG
    require(not unknown, 'Unknown config behavior cannot be safely cached: ' + repr(sorted(unknown)))
    if 'stage_role' in config:
        require(isinstance(config['stage_role'],str),'stage_role is declared description metadata and must be a string')
    method = deepcopy(config['method'])
    family = method['family'].lower()
    if method.get('adapter_class') == 'TurboQuantAdapter': method.setdefault('per_layer_seed', {'base':42,'step':6})
    if method.get('adapter_class') == 'PyramidKVVisualAdapter':
        method.setdefault('encode_scope', 'prefix')
        method.setdefault('constructor_args', {}).setdefault('num_layers', spec['layers'])
    method.setdefault('information_condition', 'query_unknown')
    return {'model_id':config['model_id'], 'model_revision':config.get('model_revision',spec['revision']),
            'method':method,'preprocessing':config.get('preprocessing',{}),
            'max_new_tokens':int(config.get('max_new_tokens',64)),
            'question_suffix':config.get('question_suffix','\nAnswer the question using a single word or phrase.'),
            'verify_question_unknown_prefix':config.get('verify_question_unknown_prefix',True),
            'weight_dtype':'float16','weight_quantization':False,'batch_size':1,'do_sample':False,
            'tf32':False,'seed':0,'compression_boundary_semantics':'native_image_prefix_then_full_question_suffix_recompute',
            'image_decode_policy':'PIL.open original bytes; first frame; convert RGB; no explicit EXIF transpose',
            'method_family':family}


def input_identity(row): return {key:row.get(key) for key in INPUT_FIELDS}
def input_key(row): return canonical_sha(input_identity(row))


def execution_fingerprint(row):
    names = ('model_commit','dtype','method','preprocessing','actual_visual_token_count','chat_template_hash',
             'full_prompt_hash','input_ids_sha256','rendered_question_sha256','question_suffix_sha256',
             'compression_boundary','native_position_ids_shape','native_position_ids_sha256',
             'stored_visual_KV_bytes','complete_visual_owner_bytes','full_kv_byte_denominator')
    return canonical_sha({key:row.get(key) for key in names})


def _proof(directory, receipt_name, files, status):
    receipt = read(directory / receipt_name)
    require(receipt.get('status') == status and receipt.get('remote_hashes_computed_independently') is True,
            'Independent remote proof required: ' + str(directory / receipt_name))
    anchored = {r['file']:r for r in receipt['files']}
    require(set(anchored) == set(files), 'Proof file coverage mismatch')
    actual = {}
    for name in files:
        actual[name] = sha(directory / name)
        require(anchored[name]['sha256'] == actual[name] and anchored[name]['bytes'] == (directory/name).stat().st_size,
                'Proof mismatch: ' + name)
    return actual


def _dependencies(manifest, inventory, source_evaluate):
    cfg, spec = manifest['config'], manifest['model_spec']
    protocol = 'code/m1_protocol.py' if spec['model_id'] == 'M1' else 'code/qwen_protocol.py'
    files = set(CORE) | {protocol}
    pair = cfg['method'].get('adapter_class')
    declared=(cfg['method'].get('adapter_module'),pair)
    if pair is not None:
        require(declared in set(LEGACY_ADAPTER_PAIRS)|set(APPENDED_ADAPTER_PAIRS_V3),
                'Unsupported module/class dependency pair; class names alone are not enough')
    if pair == 'KIVIStyleAdapter':
        files |= {'code/methods/kivi_style.py','code/methods/kivi_reader.py','third_party/kivi/quant/new_pack.py'}
    elif pair == 'KIVIExampleAdapter':
        files |= {'code/methods/kivi_example_adapter.py','code/methods/kivi_example_reader.py',
                  'third_party/kivi/quant/new_pack.py','third_party/kivi/example.py',
                  'configs/methods/kivi_example_source_v1.json'}
    elif pair in ('CalibQuantAdapter','CalibQuantFP32BiasAdapter'):
        files |= {'code/methods/calibquant_adapter.py','third_party/calibquant/calibquant.py'}
        if pair == 'CalibQuantFP32BiasAdapter': files.add('code/methods/calibquant_fp32_bias_adapter.py')
    elif pair == 'CalibQuantFP32AffineTextAdapter':
        require(inventory.get('code/methods/calibquant_fp32_text_adapter.py')==
                'd2b1c26caee553622ba159ffb2df8748b6d1d02731103cd109bfc121a0d43075',
                'DEPENDENCY_CONTRACT_PENDING: unreviewed Calib text FP32 adapter production-source revision')
        files |= {'code/methods/calibquant_fp32_text_adapter.py',
                  'code/methods/calibquant_fp32_affine_adapter.py',
                  'code/methods/calibquant_adapter.py','third_party/calibquant/calibquant.py'}
    elif pair in ('TurboQuantAdapter','TurboQuantBufferedAdapter'):
        files |= {'code/methods/turboquant_adapter.py','configs/methods/turboquant_eviction_lock.json'}
        source_files = {p for p in inventory if p.startswith('third_party/turboquant_0xsero/') and p.endswith(('.py','.json'))}
        require(bool(source_files), 'TurboQuant upstream proof absent')
        files |= source_files
        if pair=='TurboQuantBufferedAdapter':files.add('code/methods/turboquant_buffered_adapter.py')
    elif pair == 'OnlyTerpMSEAdapter':
        # P's confirmed v2 production path is self-contained. Unknown adapter
        # revisions must be reviewed again before assuming this dependency map.
        require(inventory.get('code/methods/turboquant_onlyterp_mse_adapter.py')==
                'ef52ac83eaabae8cbc41c46bc976bc6d6fcb696f2897bcb2bda534e2431ee86e',
                'DEPENDENCY_CONTRACT_PENDING: unreviewed OnlyTerpMSEAdapter production-source revision')
        files |= {'code/methods/turboquant_onlyterp_mse_adapter.py',
                  'configs/methods/onlyterp_mse_source_lock_v2.json',
                  'third_party/turboquant_onlyterp/src/cache.py'}
    elif pair == 'PyramidKVVisualAdapter':
        files |= {'code/methods/eviction_adapter.py','code/methods/turboquant_adapter.py',
                  'third_party/kvpress/kvpress/presses/pyramidkv_press.py',
                  'third_party/kvpress/kvpress/presses/snapkv_press.py'}
    else:
        require(cfg['method']['family'].lower() in ('fp16','native','fp16_hybrid','full_native'), 'Unsupported cache dependency map')
    for path in files: require(path in inventory, 'Snapshot dependency missing: ' + path)
    for path,digest in manifest['source_hashes'].items():
        require(inventory.get(path) == digest, 'Manifest/source-inventory mismatch: ' + path)
    require(inventory[protocol] == manifest['protocol_source_sha256'], 'Protocol source identity mismatch')
    for path in files:
        if path.startswith('code/methods/'):
            require(manifest['method_and_reader_source_hashes'].get(PurePosixPath(path).name) == inventory[path],
                    'Adapter/reader source identity mismatch: ' + path)
    if cfg['method'].get('adapter_module'):
        modulepath = 'code/' + cfg['method']['adapter_module'].replace('.','/') + '.py'
        require(manifest['method_source_sha256'] == inventory[modulepath], 'Selected adapter fingerprint mismatch')
    result={key:inventory[key] for key in sorted(files)}
    result['code/evaluate.py']=evaluate_semantics(source_evaluate,inventory['code/evaluate.py'])
    return result


class SourceRun:
    """Stream-validated raw run, retaining offsets and small identities only."""
    def __init__(self, directory):
        self.path = resolve_run(directory)
        self.hashes = _proof(self.path, 'retrieval_receipt.json', RAW_FILES, 'PASS_REMOTE_RAW_SHA256_EXACT')
        self.manifest, self.complete = read(self.path/'manifest.json'), read(self.path/'complete.json')
        m,c = self.manifest,self.complete
        require(m['schema'] == 'baseline_matrix.raw_inference.v1', 'Index original raw runs; do not index a derived view')
        require(c['manifest_sha256'] == self.hashes['manifest.json'] and c['records_jsonl_sha256'] == self.hashes['records.jsonl'], 'Raw closure SHA mismatch')
        require(c['coverage_complete'] is True and c['quality_scored'] is False and m['references_read'] is False,
                'Expected closed, reference-free unscored inference')
        self.source_hashes = _proof(self.path, 'source_retrieval_receipt_v2.json', ('source_inventory.json','launch_receipt.json','source_evaluate.py'), 'PASS_REMOTE_SOURCE_SHA256_EXACT')
        inventory,launch = read(self.path/'source_inventory.json'),read(self.path/'launch_receipt.json')
        require(launch['source_snapshot'] == m['runtime_source_root'], 'Wrong snapshot receipt')
        require(launch['runtime_python'] == m['runtime_executable'] == launch['argv_executed'][0], 'Wrong inference runtime receipt')
        require(launch['source_inventory_sha256'] == self.source_hashes['source_inventory.json'], 'Original launch/source inventory binding mismatch')
        identity = m['model_identity']
        require(identity['status'] == 'PASS_EXACT_HF_REQUIRED_TREE' and identity['revision'] == m['model_spec']['revision'], 'Unverified model identity')
        self.contract = {'schema':'baseline_matrix.effective_prediction_contract.v1',
            'behavior':normalize_behavior(m['config'],m['model_spec']), 'model_spec':m['model_spec'],
            'model_files':sorted(({k:r[k] for k in ('file','bytes','sha256')} for r in identity['files']),key=lambda r:r['file']),
            'runtime':{k:m[k] for k in ('torch_version','transformers_version','cuda_version','python_version','hardware','runtime_executable')},
            'processor_id':m['processor_id'],'processor_version':m['processor_version'],
            'processor_file_hashes':m['processor_file_hashes'], 'generation_native':m['generation_native'],
            'generation_overrides':m['generation_overrides'], 'relevant_source_hashes':_dependencies(m,inventory,self.path/'source_evaluate.py')}
        self.contract_sha = canonical_sha(self.contract)
        expected = m['expected_ids']
        require(isinstance(expected,list) and len(expected)==len(set(expected)), 'Invalid expected IDs')
        require(len(expected)==m['expected_QA']==c['expected_QA']==c['recorded_QA'], 'Coverage count mismatch')
        self.entries, counts, seen = [], Counter(), set()
        with (self.path/'records.jsonl').open('rb') as stream:
            number = 0
            while True:
                offset=stream.tell();line=stream.readline()
                if not line: break
                number+=1;require(line.endswith(b'\n') and line.strip(), 'Raw JSONL line must end with newline')
                row=json.loads(line)
                require(number<=len(expected) and row['id']==expected[number-1] and row['id'] not in seen, 'Raw expected-order/duplicate mismatch')
                seen.add(row['id']);counts[row['status']]+=1
                require(row.get('official_score') is None, 'Raw unexpectedly contains score')
                require(row.get('model_id')==m['config']['model_id'] and row.get('method')==m['config']['method']['id'], 'Row/config method/model mismatch')
                if row['status']=='PASS':
                    require(isinstance(row.get('raw_prediction'),str) and bool(row.get('output_token_ids')), 'PASS lacks generated output')
                    require(row['model_commit']==identity['revision'] and row['dtype']=='float16', 'Row/model identity mismatch')
                    require(row['method_contract']==m['config']['method'], 'Per-row method mismatch')
                    require(row['raw_question']==row['question'], 'Raw question mismatch')
                    require(row['rendered_question']==row['question']+self.contract['behavior']['question_suffix'], 'Rendered question mismatch')
                self.entries.append({'id':row['id'],'status':row['status'],'input':input_identity(row),'input_key':input_key(row),
                    'line_number':number,'offset':offset,'length':len(line),'line_sha256':hashlib.sha256(line).hexdigest(),
                    'prediction_sha256':canonical_sha({'ids':row.get('output_token_ids'),'prediction':row.get('raw_prediction')}),
                    'execution_sha256':execution_fingerprint(row),'contract_sha256':self.contract_sha})
        require(len(self.entries)==len(expected) and dict(counts)==c['counts'], 'Raw final coverage/status mismatch')
        required_status='PASS_RAW_COMPLETE' if counts.get('PASS',0)==len(expected) else 'COMPLETE_WITH_FAILURES'
        require(c['status']==required_status, 'Raw completion status mismatch')

    def line(self, entry):
        with (self.path/'records.jsonl').open('rb') as stream:
            stream.seek(entry['offset']);line=stream.read(entry['length'])
        require(hashlib.sha256(line).hexdigest()==entry['line_sha256'], 'Frozen source row changed')
        return line

    def source_descriptor(self):
        return {'raw_dir':str(self.path),'raw_hashes':self.hashes,'source_proof_hashes':self.source_hashes,
                'retrieval_receipt_sha256':sha(self.path/'retrieval_receipt.json'),
                'source_retrieval_receipt_sha256':sha(self.path/'source_retrieval_receipt_v2.json'),
                'effective_contract_sha256':self.contract_sha,'original_inference_run':True,
                'source_recorded_QA':self.complete['recorded_QA'],'source_status_counts':self.complete['counts']}


def index_runs(directories):
    runs,index,failures,duplicates = [],{},[],[]
    seen_dirs=set()
    for directory in directories:
        resolved=resolve_run(directory)
        if resolved in seen_dirs: continue
        seen_dirs.add(resolved);run=SourceRun(resolved);runs.append(run)
        for entry in run.entries:
            if entry['status']!='PASS':
                failures.append({'run':str(run.path),'id':entry['id'],'status':entry['status']});continue
            key=(run.contract_sha,entry['input_key'])
            if key in index:
                prior_run,prior=index[key]
                if (entry['prediction_sha256'],entry['execution_sha256']) != (prior['prediction_sha256'],prior['execution_sha256']):
                    raise CacheConflictError('Same-contract duplicate predictions/execution differ: '+entry['id']+' '+str(prior_run.path)+' '+str(run.path))
                duplicates.append({'id':entry['id'],'kept_first_run':str(prior_run.path),'same_answer_run':str(run.path)})
            else:index[key]=(run,entry)
    return runs,index,failures,duplicates


def selected_inputs(config, input_path):
    input_path=Path(input_path)
    require(input_path.name.endswith('_inputs.jsonl'), 'Only frozen reference-free *_inputs.jsonl may be read by cache planning')
    lock=read(input_path.parent/'source_lock.json')['files'][input_path.name]
    require(sha(input_path)==lock['sha256'] and input_path.stat().st_size==lock['bytes'], 'Frozen input/source-lock mismatch')
    require(PurePosixPath(config['inputs'].replace('\\','/')).name==input_path.name, 'Wrong phase input file')
    allrows=[json.loads(line) for line in input_path.read_text(encoding='utf-8').splitlines()]
    lookup={row['id']:row for row in allrows};require(len(lookup)==len(allrows), 'Duplicate frozen input IDs')
    if config.get('ids') is not None:rows=[lookup[i] for i in config['ids']]
    else:
        start=int(config.get('start',0));rows=allrows[start:start+int(config.get('count',len(allrows)-start))]
    require(bool(rows) and len({r['id'] for r in rows})==len(rows), 'Empty/duplicate target cohort')
    return rows


def plan_reuse(config, input_path, reference_run, cached_dirs, *, source_root=ROOT):
    reference = reference_run if isinstance(reference_run,SourceRun) else SourceRun(reference_run)
    behavior = normalize_behavior(config,reference.manifest['model_spec'])
    require(behavior==reference.contract['behavior'], 'STALE_PROTOCOL: planned model/method/pixels/prompt/generation differs')
    for key in ('model_path','model_source_lock'):
        require(config.get(key)==reference.manifest['config'].get(key), 'Planned checkpoint relocation requires a fresh verified reference run: '+key)
    source_root=Path(source_root)
    for path,digest in reference.contract['relevant_source_hashes'].items():
        if path=='code/evaluate.py':
            require(evaluate_semantics(source_root/path,sha(source_root/path))==digest,'STALE_PROTOCOL: numerical execution changed')
        else:require((source_root/path).is_file() and sha(source_root/path)==digest, 'STALE_PROTOCOL: current source differs from reference: '+path)
    source_lock_name = PurePosixPath(config['model_source_lock']).name
    require(sha(source_root/'configs/models'/source_lock_name)==reference.manifest['model_identity']['source_lock_sha256'],
            'STALE_PROTOCOL: planned model source lock changed')
    rows=selected_inputs(config,input_path)
    runs,index,failures,duplicates=index_runs(cached_dirs)
    hits,missing=[],[]
    for row in rows:
        pair=index.get((reference.contract_sha,input_key(row)))
        if pair:
            run,entry=pair;hits.append({'id':row['id'],'source_run':str(run.path),'source_line':entry['line_number'],
                                      'source_line_sha256':entry['line_sha256'],'input_key_sha256':entry['input_key']})
        else:missing.append(row['id'])
    missing_config=deepcopy(config)
    missing_config.update(ids=missing,start=0,count=len(missing))
    plan={'schema':'baseline_matrix.prediction_reuse_plan.v1','status':'REUSE_PLAN_VALID',
          'inference_run':False,'references_read':False,'target_config':deepcopy(config),
          'inputs_sha256':sha(input_path),'expected_ids':[r['id'] for r in rows],
          'expected_QA':len(rows),'cached_PASS_QA':len(hits),'missing_QA':len(missing),
          'hits':hits,'missing_ids':missing,'failed_records_not_cache_hits':failures,'verified_duplicate_aliases':duplicates,
          'effective_contract':reference.contract,'effective_contract_sha256':reference.contract_sha,
          'contract_reference_run':reference.source_descriptor(),
          'new_worker_environment_note':'Expected context is the frozen reference contract; every fresh raw run must independently match it before composition',
          'timing_reuse':'quality predictions and original resource records only; never canonical replay speed',
          'created_unix':time.time()}
    return plan,missing_config


def compose_view(config, input_path, reference_run, cached_dirs, fresh_dirs, output, *, source_root=ROOT):
    output=Path(output)
    require(not output.exists(), 'Derived output must be new; frozen views cannot be overwritten')
    reference=reference_run if isinstance(reference_run,SourceRun) else SourceRun(reference_run)
    plan,_=plan_reuse(config,input_path,reference,cached_dirs,source_root=source_root)
    cached_runs,cached,_,_=index_runs(cached_dirs)
    fresh_runs,fresh,_,_=index_runs(fresh_dirs)
    for run in fresh_runs:require(run.contract_sha==reference.contract_sha, 'STALE_PROTOCOL: fresh run differs from declared effective contract')
    # Failure rows are not hits. A fresh failure is retained in final coverage.
    fresh_failure={}
    for run in fresh_runs:
        for entry in run.entries:
            if entry['status']!='PASS':
                key=(run.contract_sha,entry['input_key'])
                require(key not in fresh_failure, 'Duplicate fresh failure: '+entry['id'])
                fresh_failure[key]=(run,entry)
    chosen=[];counts=Counter();role_counts=Counter();all_sources={}
    for row in selected_inputs(config,input_path):
        key=(reference.contract_sha,input_key(row))
        old,new=cached.get(key),fresh.get(key)
        if old and (new or key in fresh_failure):
            require(new is not None and old[1]['prediction_sha256']==new[1]['prediction_sha256'] and old[1]['execution_sha256']==new[1]['execution_sha256'],
                    'Cached/fresh duplicate conflict: '+row['id'])
        pair=old or new or fresh_failure.get(key)
        require(pair is not None, 'Missing target QA cannot be silently dropped: '+row['id'])
        run,entry=pair;role='reused' if old else 'fresh'
        chosen.append((run,entry,role));counts[entry['status']]+=1;role_counts[role]+=1
        all_sources[str(run.path)]=run
    output.mkdir(parents=True)
    manifest=deepcopy(reference.manifest)
    manifest.update(schema='baseline_matrix.derived_prediction_view.v1',config=deepcopy(config),config_sha256=canonical_sha(config),
                    inputs_sha256=sha(input_path),expected_ids=plan['expected_ids'],expected_QA=plan['expected_QA'],
                    inference_run=False,prediction_cache_used=True,model_load_ms=None,started_unix=None,
                    references_read=False,derived_created_unix=time.time(),effective_contract=reference.contract,
                    effective_contract_sha256=reference.contract_sha,
                    source_runs=[all_sources[k].source_descriptor() for k in sorted(all_sources)],
                    QA_origin_counts=dict(role_counts),
                    fresh_source_run_recorded_QA=sum(run.complete['recorded_QA'] for run in fresh_runs),
                    fresh_source_runs_supplied=[run.source_descriptor() for run in fresh_runs],
                    timing_scope='Derived quality view; each unchanged row retains its original instrumented timing; no new GPU run or controlled speed claim')
    with (output/'records.jsonl').open('xb') as raw, (output/'provenance.jsonl').open('x',encoding='utf-8') as provenance:
        for run,entry,role in chosen:
            raw.write(run.line(entry))
            item={'id':entry['id'],'role':role,'source_run':str(run.path),
                  'source_line_number':entry['line_number'],'source_exact_line_sha256':entry['line_sha256'],
                  'source_raw_hashes':run.hashes,'input_key_sha256':entry['input_key'],
                  'effective_contract_sha256':run.contract_sha,'source_execution_sha256':entry['execution_sha256']}
            provenance.write(json.dumps(item,ensure_ascii=False)+'\n')
    with (output/'manifest.json').open('x',encoding='utf-8') as stream:json.dump(manifest,stream,ensure_ascii=False,indent=2)
    complete={'status':'PASS_RAW_COMPLETE' if counts.get('PASS',0)==len(chosen) else 'COMPLETE_WITH_FAILURES',
              'expected_QA':len(chosen),'recorded_QA':len(chosen),'counts':dict(counts),'coverage_complete':True,
              'quality_scored':False,'inference_run':False,'derived_prediction_view':True,
              'QA_origin_counts':dict(role_counts),'manifest_sha256':sha(output/'manifest.json'),
              'records_jsonl_sha256':sha(output/'records.jsonl'),'provenance_jsonl_sha256':sha(output/'provenance.jsonl'),
              'completed_unix':time.time()}
    with (output/'complete.json').open('x',encoding='utf-8') as stream:json.dump(complete,stream,indent=2)
    validate_view_source_provenance(output)
    return complete


def validate_view_source_provenance(directory):
    """Validate local derived closure + every original source SHA and row.

    Returns raw_hashes, origin_counts, source_runs, inference_run=False.
    The offline scorer then independently verifies its locked target inputs/refs.
    No fictitious remote-retrieval receipt is created for the derived view.
    """
    directory=Path(directory).resolve()
    m,c=read(directory/'manifest.json'),read(directory/'complete.json')
    require(m['schema']=='baseline_matrix.derived_prediction_view.v1' and m['inference_run'] is False and m['prediction_cache_used'] is True,
            'Not an explicitly derived prediction view')
    hashes={name:sha(directory/name) for name in RAW_FILES}
    require(c['manifest_sha256']==hashes['manifest.json'] and c['records_jsonl_sha256']==hashes['records.jsonl'], 'Derived closure mismatch')
    require(c['provenance_jsonl_sha256']==sha(directory/'provenance.jsonl'), 'Derived provenance SHA mismatch')
    require(c['inference_run'] is False and c['derived_prediction_view'] is True and c['coverage_complete'] is True,
            'Derived view mislabeled as GPU run')
    require(canonical_sha(m['effective_contract'])==m['effective_contract_sha256'], 'Derived effective contract SHA mismatch')
    runs={}
    for descriptor in m['source_runs']:
        run=SourceRun(descriptor['raw_dir'])
        require(run.source_descriptor()==descriptor, 'Underlying source descriptor changed')
        require(run.contract_sha==m['effective_contract_sha256'], 'Underlying source context differs')
        runs[str(run.path)]=run
    rows=0;roles=Counter();counts=Counter()
    with (directory/'records.jsonl').open('rb') as raw, (directory/'provenance.jsonl').open('rb') as provenance:
        for expected_id in m['expected_ids']:
            line,proofline=raw.readline(),provenance.readline()
            require(line and proofline, 'Derived missing expected row/provenance')
            row,p=json.loads(line),json.loads(proofline)
            require(row['id']==p['id']==expected_id, 'Derived row order/identity differs')
            require(p['role'] in ('fresh','reused'), 'Unknown prediction-origin role')
            run=runs.get(p['source_run']);require(run is not None,'Unlisted source run')
            number=p['source_line_number'];require(1<=number<=len(run.entries),'Invalid source line number')
            entry=run.entries[number-1]
            require(entry['id']==expected_id and run.line(entry)==line, 'Derived record bytes differ from original source row')
            require(p['source_exact_line_sha256']==entry['line_sha256'] and p['source_raw_hashes']==run.hashes,
                    'Derived source SHA proof differs')
            require(p['input_key_sha256']==entry['input_key'] and p['source_execution_sha256']==entry['execution_sha256'] and
                    p['effective_contract_sha256']==run.contract_sha, 'Derived cache identity proof differs')
            rows+=1;roles[p['role']]+=1;counts[row['status']]+=1
        require(not raw.readline() and not provenance.readline(), 'Derived unexpected extra row')
    require(rows==m['expected_QA']==c['expected_QA']==c['recorded_QA'] and dict(counts)==c['counts'], 'Derived coverage/status mismatch')
    require(dict(roles)==m['QA_origin_counts']==c['QA_origin_counts'], 'Derived fresh/reused counts differ')
    require(c['status']==('PASS_RAW_COMPLETE' if counts.get('PASS',0)==rows else 'COMPLETE_WITH_FAILURES'), 'Derived status mismatch')
    return {'status':'PASS_DERIVED_SOURCE_PROVENANCE','raw_hashes':hashes,'origin_counts':dict(roles),
            'source_runs':len(runs),'inference_run':False,'provenance_sha256':sha(directory/'provenance.jsonl')}
