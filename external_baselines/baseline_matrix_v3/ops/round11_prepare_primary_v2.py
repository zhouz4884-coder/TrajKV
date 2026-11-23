"""Freeze the authorized D96 primary jobs and complete 672-view harvest plan.

Local JSON/source preparation only. No SSH, staging, model, CUDA, launch,
queue mutation or Git operation occurs. Original R7low templates retain
all model, native prompt, codec, original input and generation parameters.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'ops'), str(ROOT / 'code')]
import round11_jobs_v1 as jobs
import round11_runner_v2 as runner
import round11_technical_v2 as technical
import round11_harvest_v2 as harvest

SOURCE_GROUP_SHA = '663076f21fc7df7d00253d06604e9f46da06ffe9d62532cd66d82794c05fa9b4'
INPUT_MANIFEST_SHA = 'b2ba5f54516abf96344e02356c7cee079b4ff48e2a270ea8fc61ea1c78baaecf'
FINAL_SOURCE_PINS = {
    'code/round11_joint_value_v2.py': 'b819a540f9a07dba153513b45a248f7b56cbe3b46f0115b6e2c7671260f5ea82',
    'code/round11_value_bank_v2.py': 'f65273b2f69c39e6ca9271a62f337b3e57b808d0f9e59325c6ef055456985e16',
    'code/round11_runner_v2.py': 'ec87b1270bfae11d4a2d98fcafc544fcf42a3c132c72bc30d41fe4176ee73ce1',
    'code/round11_technical_v2.py': 'f5664886ca629f5f4b54744917459653770c96f7d60eb9e92182cf74945cef83',
    'code/round11_joint_value_v1.py': '6d76a770163d17574f5f15fa888886edc5cd1e12c6e153d610a856a78c95378b',
    'code/round11_attention_v1.py': '3290a1b372a8639f702e9ac3a9e183b06147c0b73ac402a24ef21bb35556119f',
    'code/round10_sequence_v1.py': '2d0dce9da80156fb0224a04d48a9a3e3869ede5e9885885a6a25073ab511fe00',
}
ALLOCATION = {0: ('JointV', 'Spatial'), 1: ('JointSpatial', 'Shuffled')}
DATASET_ORDER = ('chartqa', 'docvqa', 'textvqa', 'infovqa')
REFERENCE_PLAN = jobs.E / 'driver_cpu_v6/reference_only_plan.json'
BOOTSTRAP = jobs.E / 'bootstrap_v1'


def require(value, message):
    if not value:
        raise ValueError('ROUND11_PRIMARY_PREPARATION: ' + message)


def finite_source_graph():
    for role, digest in FINAL_SOURCE_PINS.items():
        require(jobs.sha(ROOT / role) == digest, 'final source pin differs: ' + role)
    require(jobs.sha(jobs.E / 'model_technical_retrieved_v1/source_files/code/round11_attention_v1.py')
            == FINAL_SOURCE_PINS['code/round11_attention_v1.py'], 'reader must match independently retrieved executed v1 source')
    graph = runner.runtime_source_manifest()
    technical_graph = technical.runtime_source_manifest()
    require(len(graph) == 19 and len(technical_graph) == 23, 'complete approved source graph differs')
    for candidate in (graph, technical_graph):
        require(all(re.fullmatch('[0-9a-f]{64}', digest) and digest != '0' * 64
                    and jobs.sha(ROOT / role) == digest for role, digest in candidate.items()),
                'runtime source SHA missing, placeholder or different from actual bytes')
    return graph, technical_graph


def unchanged_template_parameters(original, config):
    # Every change is in this explicit execution/declaration allowlist. In
    # particular variant (including all three asset paths) is exactly equal.
    changed = {key for key in set(original) | set(config) if original.get(key) != config.get(key)}
    allowed = {'schema', 'count', 'method', 'scope', 'candidate', 'expected_source_hashes',
               'input_manifest_sha256', 'source_group_manifest_path', 'source_group_manifest_sha256'}
    require(changed <= allowed, 'unexpected old-template change: ' + ','.join(sorted(changed - allowed)))
    for role in ('model_id', 'model_revision', 'model_path', 'model_source_lock', 'max_new_tokens',
                 'question_suffix', 'preprocessing', 'seed', 'variant', 'variant_source',
                 'expected_asset_sha256', 'position_owner_layout', 'inputs', 'inputs_sha256',
                 'input_manifest_path', 'input_role', 'input_count', 'start', 'evaluation_contract',
                 'legacy_a2', 'legacy_experiments', 'legacy_scripts', 'kivi_source'):
        require(config[role] == original[role], 'original model/input/codec/protocol field changed: ' + role)
    source = jobs.read(ROOT / 'configs/r7_research/frozen_assets_v2/confirm_doc128_s0.json')
    matched = [value for value in source['variants'] if value['name'] == 'r7_joint_k1_gt_cond']
    require(len(matched) == 1, 'one original frozen R7low variant required')
    asset_roles = set(original['expected_asset_sha256'])
    without_assets = lambda variant: {key: value for key, value in variant.items() if key not in asset_roles}
    require(without_assets(config['variant']) == without_assets(matched[0]), 'original variant outside three asset paths changed')
    require(config['candidate'] == {'mode': config['candidate']['mode']}
            and config['candidate']['mode'] in runner.MODES, 'fitted/mixed/technical candidate forbidden')
    require(not (set(config) & runner.frozen.FORBIDDEN), 'references entered inference config')
    # Use only the byte-frozen structural validator. The public R11 validator
    # performs actual remote input/group I/O and must run on the target host.
    runner.install_driver_overrides()
    shadow = deepcopy(config); shadow['method']['family'] = 'trajkv_round10_research'
    runner._validate_base(shadow)
    return sorted(changed)


def prepare():
    graph, technical_graph = finite_source_graph()
    group_path = ROOT / 'configs/round11_v1/SOURCE_GROUP_MANIFEST.json'
    require(jobs.sha(group_path) == SOURCE_GROUP_SHA, 'SourceGroupD96 changed')
    group = jobs.read(group_path)
    require(group['reference_input_manifest_sha256'] == INPUT_MANIFEST_SHA
            and sum(detail['count'] for detail in group['datasets'].values()) == 96,
            'SourceGroupD96 original manifest/count differs')
    reference_plan = jobs.read(REFERENCE_PLAN)
    require(reference_plan['schema'] == 'trajkv.round11.harvest_plan.v1'
            and len(reference_plan['entries']) == 12
            and all(entry['kind'] == 'reused' for entry in reference_plan['entries'])
            and reference_plan['source_group_manifest_sha256'] == SOURCE_GROUP_SHA,
            'exact twelve original-producer reference entries required')
    require(sum(runner.COUNTS[entry['dataset']] for entry in reference_plan['entries']) == 288,
            'reused D96 references must be288 views and0 fresh QA')
    require(all(Path(domain['source_lock']).name == 'SOURCE_LOCK.json'
                for domain in reference_plan['domains'].values()), 'preserve original source-lock casing')
    scripts = 'code/round11_runner_v2.py'
    source_group_remote = jobs.REMOTE_ROOT + '/configs/round11_v1/SOURCE_GROUP_MANIFEST.json'
    all_cells, queue_plans, new_entries, config_receipts = [], [], [], []
    for gpu, modes in ALLOCATION.items():
        cells = []
        for dataset in DATASET_ORDER:
            detail = group['datasets'][dataset]
            original_size = detail['input_count']; count = detail['count']
            template_path = ROOT / f'configs/round10_v1/R7low_{dataset}_{original_size}_v1.json'
            original = jobs.read(template_path)
            require(original['inputs_sha256'] == detail['input_sha256']
                    and original['inputs'] == detail['input_path']
                    and original['input_count'] == original_size and original['start'] == detail['start'] == 0,
                    'original input lineage/source-group slice differs')
            for mode in modes:
                name = f'{mode}_{dataset}_{count}_v2'
                config = deepcopy(original)
                config.update(schema=runner.SCHEMA, count=count,
                    method={'id': 'Round11_' + mode + '_D96_v2', 'family': 'trajkv_round11_research',
                            'encode_scope': 'visual', 'information_condition': 'query_unknown'},
                    scope='Round11 SourceGroupD96 development; fixed first32 Text/Info and16 Doc/Chart; no source-quality claim from QA0',
                    candidate={'mode': mode}, expected_source_hashes=graph,
                    input_manifest_sha256=INPUT_MANIFEST_SHA,
                    source_group_manifest_path=source_group_remote, source_group_manifest_sha256=SOURCE_GROUP_SHA)
                changes = unchanged_template_parameters(original, config)
                relative = f'configs/round11_v1/{name}.json'
                config_sha = jobs.freeze(ROOT / relative, config)
                cell = {'name': name, 'dataset': dataset, 'mode': mode,
                    'script': scripts, 'script_sha256': graph[scripts],
                    'config': jobs.REMOTE_ROOT + '/' + relative, 'config_sha256': config_sha,
                    'output': jobs.REMOTE_ROOT + '/results/round11_v1/dev/' + name, 'expected_QA': count}
                cells.append(cell); all_cells.append(cell)
                config_receipts.append({'name': name, 'local_path': str(ROOT / relative), 'sha256': config_sha,
                    'template_path': str(template_path), 'template_sha256': jobs.sha(template_path),
                    'explicit_changed_fields': changes, 'all_model_prompt_generation_variant_fields_unchanged': True,
                    'actual_remote_inputs_opened_locally': False})
        require(len(cells) == 8 and sum(cell['expected_QA'] for cell in cells) == 192, 'one queue must be8 cells192QA')
        require([(cell['dataset'], cell['mode']) for cell in cells]
                == [(dataset, mode) for dataset in DATASET_ORDER for mode in modes], 'sentinel/interleaved queue order differs')
        sequence_relative = f'configs/round10_v1/R11_primary_gpu{gpu}_v2.json'
        sequence = {'schema': 'trajkv.round10.sequence.v1', 'scope': 'Round11 SourceGroupD96 primary; frozen sequence producer',
                    'expected_QA': 192, 'entries': cells}
        sequence_sha = jobs.freeze(ROOT / sequence_relative, sequence)
        parent_output = jobs.REMOTE_ROOT + f'/results/round11_v1/queues/primary_gpu{gpu}_v2'
        job = jobs.job(f'bm_R11_primary_gpu{gpu}_v2', 'code/round10_sequence_v1.py',
            ['--config', jobs.REMOTE_ROOT + '/' + sequence_relative, '--output', parent_output], gpu)
        queue_plans.append({'GPU': gpu, 'modes': list(modes), 'expected_new_QA': 192,
            'sequence_path': str(ROOT / sequence_relative), 'sequence_sha256': sequence_sha,
            'sequence_remote': jobs.REMOTE_ROOT + '/' + sequence_relative, 'parent_output': parent_output,
            'job': job, 'cells': cells})
        for cell in cells:
            new_entries.append({'kind': 'fresh', 'dataset': cell['dataset'], 'mode': cell['mode'],
                'item': {'root': jobs.REMOTE_ROOT, 'job': job['job'], 'job_sha256': job['sha256'],
                    'cell': cell, 'sequence_sha256': sequence_sha,
                    'sequence_script_sha256': FINAL_SOURCE_PINS['code/round10_sequence_v1.py'],
                    'parent_script': 'code/round10_sequence_v1.py', 'sequence_schema': 'trajkv.round10.sequence.v1',
                    'input_manifest_sha256': INPUT_MANIFEST_SHA, 'source_group_manifest_sha256': SOURCE_GROUP_SHA}})
    require(len(all_cells) == len({cell['name'] for cell in all_cells}) == 16
            and sum(cell['expected_QA'] for cell in all_cells) == 384, 'primary must be16 unique cells384QA')
    require({(cell['dataset'], cell['mode']) for cell in all_cells}
            == {(dataset, mode) for dataset in DATASET_ORDER for mode in runner.MODES}, 'primary source/mode coverage differs')
    harvest_plan = deepcopy(reference_plan)
    harvest_plan.update(entries=[*reference_plan['entries'], *new_entries],
        primary_new_free_generation_QA=384, reused_reference_views=288, total_scored_views=672,
        reference_only_plan_sha256=jobs.sha(REFERENCE_PLAN), inference_references_read=False,
        QA0_failure_v1_is_not_primary_quality=True)
    harvest_path = BOOTSTRAP / 'HARVEST_PLAN_v2.json'
    harvest_sha = jobs.freeze(harvest_path, harvest_plan)
    # All domain paths in this plan are explicit local E mirrors. This check
    # never tries to open a /media remote input path on the local machine.
    validated, _ = harvest.load_plan(harvest_path)
    require(len(validated['entries']) == 28, 'complete harvest must bind16 fresh plus12 reused views')
    preparation = {'schema': 'trajkv.round11.primary_preparation.v2', 'status': 'PASS_LOCAL_IMMUTABLE_PREPARATION',
        'source_script_sha256': jobs.sha(__file__), 'primary_runtime_source_hashes': graph,
        'technical_runtime_source_hashes': technical_graph, 'seven_actual_execution_source_pins': FINAL_SOURCE_PINS,
        'source_group_manifest_sha256': SOURCE_GROUP_SHA, 'input_manifest_sha256': INPUT_MANIFEST_SHA,
        'expected_new_free_generation_QA': 384, 'expected_reused_reference_views': 288, 'total_views': 672,
        'declared_primary_cells': 16, 'new_GPU_jobs_started': 0, 'SSH_used': False, 'remote_inputs_opened_locally': False,
        'sentinel_order': list(DATASET_ORDER), 'queues': queue_plans, 'config_receipts': config_receipts,
        'harvest_plan_path': str(harvest_path), 'harvest_plan_sha256': harvest_sha,
        'frozen_reference_only_plan_sha256': jobs.sha(REFERENCE_PLAN),
        'QA0_failure_v1_counted_separately_from_primary_quality': True,
        'actual_remote_runtime_validation_still_required_at_execution': True}
    receipt_path = BOOTSTRAP / 'PRIMARY_JOB_PLAN_v2.json'
    receipt_sha = jobs.freeze(receipt_path, preparation)
    return {'status': preparation['status'], 'cells': 16, 'new_QA': 384, 'reused_views': 288,
        'primary_plan': str(receipt_path), 'primary_plan_sha256': receipt_sha,
        'harvest_plan': str(harvest_path), 'harvest_plan_sha256': harvest_sha,
        'jobs': [{'GPU': queue['GPU'], 'sequence_sha256': queue['sequence_sha256'],
                  'job_name': queue['job']['name'], 'job_sha256': queue['job']['sha256']} for queue in queue_plans]}


if __name__ == '__main__':
    print(json.dumps(prepare(), ensure_ascii=False))
