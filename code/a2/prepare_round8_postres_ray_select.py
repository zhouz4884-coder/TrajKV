"""Local cap8 select controls for postres direction and final-state rays."""
import argparse
import copy
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path

from round8_data_prepare import ROOT, read, sha, new
from round8_quality_owner import local_path

FAMILIES = {
    'postres': dict(name='postres_select2_v1', runner='round8_postres_runner', marker='complete_postres.json',
        sha256='47b7dba060fcc06a9e49840e390e93c85194b16421013d1beb5e6af715c47bcd', modes=[False, True]),
    'final_ray': dict(name='final_ray_select5_v1', runner='round8_final_ray_runner', marker='complete_final_ray.json',
        sha256='c6a4d57da9cf3215c07c3d7c7c88ea23c23bb5f3d8b542b69bf7fb0ceedc48b8', modes=['off', 'mse', 'norm', 'projection']),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--families', nargs='+', choices=list(FAMILIES), default=list(FAMILIES))
    args = parser.parse_args()
    runtime_path = ROOT / 'configs/a2/round8_v1/job_joint_cd_select5_v1_doc.json'
    runtime = read(runtime_path)
    source_root, worker = runtime['round8_source_root'], runtime['argv'][0]
    global_root = runtime['argv'][runtime['argv'].index('--output') + 1].split('/results/', 1)[0]
    shared = global_root + '/cache/round8_v1/shared'
    require(source_root.startswith(global_root + '/stages/round8_')
        and worker.startswith(global_root.rsplit('/', 1)[0] + '/_runtime/'), 'Use isolated code and data-volume P72')
    views_path = ROOT / 'audit/a2/round8_exposed_evaluation_views_v1.json'
    views = read(views_path)['files']
    for family in args.families:
        spec = FAMILIES[family]
        backend = importlib.import_module(spec['runner'])
        sources = backend.source_fingerprints()
        require(sources[spec['runner'] + '.py'] == spec['sha256'], 'Expected approved immutable runner version')
        payloads, outputs, inputs, codec_inputs = [], [], {}, {}
        receipt_path = ROOT / f'audit/a2/round8_{spec["name"]}_prepared.json'
        for dataset, gpu, questions in [('doc', 0, 38), ('text', 1, 22)]:
            parent_path = ROOT / f'configs/a2/round8_v1/joint_cd_select5_v1_{dataset}.json'
            parent = read(parent_path)
            indexed = {variant['name']: variant for variant in parent['variants']}
            base = copy.deepcopy(indexed['r8_l2_joint0'])
            for key in list(base):
                if key.startswith('row_cosine_'):
                    base.pop(key)
            base.update(key_assignment='l2', key_amplitude='norm', joint_gain_refit=False,
                value_joint_refit=False, postres_direction=False, postres_row_chunk=4096, cap=8)
            require(base['direction_book_path'] == base['metadata_reference_book_path']
                and base['value_encoding'] == 'binary' and base['v_group_size'] == 'all'
                and len(base['selected_stages']) == 224 and base['res_stages_limit'] == 43,
                'Preserve old Gaussian16/binary/all/full224/actual43 contract')
            variants = [copy.deepcopy(indexed['full8'])] if family == 'final_ray' else []
            for mode in spec['modes']:
                variant = copy.deepcopy(base)
                if family == 'postres':
                    variant.update(name='r8_postres_direction_' + ('on' if mode else 'off'), postres_direction=mode)
                else:
                    variant.update(name='r8_final_ray_' + mode, final_ray_mode=mode)
                variants.append(variant)
            config = copy.deepcopy(parent)
            config.pop('joint_cd_factorial_design', None)
            config.update(variants=variants, backend_family=family, candidate_subset=True, factorial_complete=False,
                scope='Exposed select QA with explicit matched cap8; selected physical-field controls, not fresh confirmation')
            config['arguments']['max_patches'] = 8
            for variant in variants:
                variant['cap'] = 8
            expected_arguments = dict(parent['arguments'], max_patches=8)
            require(config['arguments'] == expected_arguments and config['count'] == questions
                and config.get('start', 0) == 0 and all(v['cap'] == 8 for v in variants), 'Explicit aligned cap8 required')
            samples = local_path(config['arguments']['samples'])
            require(config['arguments']['samples'].startswith(global_root + '/configs/')
                and sha(samples) == views[dataset]['samples_sha256'], 'Preserve exact prior select sample bytes')
            require(sha(ROOT / views[dataset]['references']) == views[dataset]['references_sha256'], 'Official old reference bytes changed')
            declarations = backend.validate_config(config)
            require(len(declarations) == (2 if family == 'postres' else 5), 'Incomplete declared controls')
            config['selected_control_design'] = dict(family=family, declared_modes=spec['modes'],
                initialization='original L2 assignment and norm gain/frozen factor; final physical coefficients described by the declared backend',
                no_ordinary_joint_gain=True, no_V_joint=True, no_CD=True, no_new_retained_tensor_fields=True,
                explicit_max_patches=8, all_variant_cap=8, same_original_select_questions=True,
                required_final_marker=spec['marker'], parent_config_sha256=sha(parent_path),
                source_scope='Historically exposed selection,16sources/domain; no independent new-source claim')
            cp = ROOT / f'configs/a2/round8_v1/{spec["name"]}_{dataset}.json'
            jp = ROOT / f'configs/a2/round8_v1/job_{spec["name"]}_{dataset}.json'
            job = copy.deepcopy(runtime)
            job.update(name='round8_' + spec['name'] + '_' + dataset, round8_source_root=source_root, cwd=source_root)
            job['env'].update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE='1')
            for key, value in job['env'].items():
                if key in ('TMPDIR', 'TMP', 'TEMP', 'HF_HOME', 'TORCH_HOME', 'PYTHONPYCACHEPREFIX') or 'CACHE' in key:
                    require(value.startswith(shared + '/'), 'Use shared data-disk cache and temp roots')
            output = global_root + '/results/round8_v1/' + spec['name'] + '_' + dataset
            job['argv'] = [worker, source_root + '/code/a2/' + spec['runner'] + '.py', '--config',
                global_root + '/' + cp.relative_to(ROOT).as_posix(), '--output', output]
            job['prerequisites'] = ['Root verifies the family actual-reader GPU checker PASS before launch',
                'Final ' + spec['marker'] + ' plus all canonical/source hashes must pass before reference scoring',
                'Explicit max_patches8/all cap8; isolated code, globalR config/sample/output and8TB P72/shared cache']
            job['dependency_enforcement'] = 'Manual root scheduling; this preparation and the launcher do not enforce GPU-check prerequisites'
            for variant in variants:
                for key, value in variant.items():
                    if key.endswith('_path') and isinstance(value, str):
                        codec_inputs[value] = sha(local_path(value))
            payloads.extend([(cp, config), (jp, job)])
            inputs[parent_path.relative_to(ROOT).as_posix()] = sha(parent_path)
            outputs.append(dict(dataset=dataset, gpu=gpu, config=cp.relative_to(ROOT).as_posix(), job=jp.relative_to(ROOT).as_posix(),
                output=output, samples=samples.relative_to(ROOT).as_posix(), samples_sha256=sha(samples),
                references_local_only=views[dataset]['references'], references_sha256=views[dataset]['references_sha256'],
                sources=16, questions=questions, variants=len(variants), expected_records=questions * len(variants),
                required_final_marker=spec['marker'], explicit_max_patches=8, all_cap=8))
        require(not any(path.exists() for path, _ in payloads) and not receipt_path.exists(), 'Existing files preserved; use a new version')
        require(backend.source_fingerprints() == sources, 'Runner changed during preparation')
        for path, value in payloads:
            new(path, value)
        for output in outputs:
            for key in ('config', 'job'):
                output[key + '_sha256'] = sha(ROOT / output[key])
        new(receipt_path, dict(schema='trajkv.round8.cap8-select-physical-controls-preparation.v1', status='PREPARED_NOT_EXECUTED',
            family=family, created_at_utc=datetime.now(timezone.utc).isoformat(), code_sha256=sha(__file__), outputs=outputs,
            source_sha256=sources, consumed_codec_inputs_sha256=codec_inputs, parent_input_sha256=inputs,
            views_sha256=sha(views_path), source_root=source_root, global_config_sample_asset_result_root=global_root,
            worker_python=worker, shared_cache=shared, explicit_max_patches=8, all_cap=8,
            original224_priority_limit43_preserved=True, expected_extra_retained_tensor_bytes=0,
            backend_GPU_gate_at_preparation='Parent reported postres8 PASS; final_ray GPU check pending root',
            refitting_or_new_book_learning=False, old_inputs_and_configs_modified=False,
            future_confirmation_answers_opened=False, GPU_used=False, SSH_used=False, uploaded=False))
        print(json.dumps(dict(status='PREPARED_NOT_EXECUTED', family=family, outputs=outputs, GPU_used=False, SSH_used=False)))


if __name__ == '__main__':
    main()
