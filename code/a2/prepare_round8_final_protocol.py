"""Post-freeze first-question confirmation and bounded measurement configs.

No fitting, selection, SSH or model execution. All reserved images are used;
the question identity rule predates answer materialization. New files only.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POOLS = ('text_validation128', 'doc_train64', 'infovqa_validation64')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def remote_path(root, path):
    return root.rstrip('/') + '/' + Path(path).relative_to(ROOT).as_posix()


def make_job(template, name, entry, config, output, gpu, root):
    job = deepcopy(template)
    stage = job['round8_source_root']
    job['name'], job['cwd'] = name, stage
    job.pop('expected_launch_receipt', None)
    job['env']['CUDA_VISIBLE_DEVICES'] = str(gpu)
    job['argv'] = [job['argv'][0], stage + '/code/a2/' + entry,
        '--config', remote_path(root, config), '--output', root + '/results/round8_v1/' + output]
    job['prerequisites'] = ['Root grants an idle GPU lease; use the dedicated immutable Round8 launcher',
        'Exact final method freeze and all consumed artifacts bound before real execution',
        'All data/model/output/cache/temp remain on the8TB project data disk']
    return job


def make_measurements(freeze, freeze_path, freeze_sha, all_variants, template):
    root = freeze['remote_project_root']
    folder = ROOT / 'configs/a2/round8_v1'
    sources = read(folder / 'calibration_select32.json')
    groups = {}
    for row in sources:
        groups.setdefault(row['dataset'], []).append(row)
    assert len(groups) == 2
    names = sorted(groups)
    quiet_rows = [deepcopy(groups[names[0]][0]), deepcopy(groups[names[1]][0]), deepcopy(groups[names[0]][1])]
    for row in quiet_rows:
        row['questions'], row['turns'] = row['questions'][:1], 1
    quiet_samples = folder / 'final_quiet3_sources_v1.json'
    write_new(quiet_samples, quiet_rows)
    result = []
    for kind, key in (('continuous', 'continuous_variant_names'), ('timing', 'timing_variant_names')):
        name = 'final_continuous8_v1' if kind == 'continuous' else 'final_quiet6_v1'
        args = deepcopy(freeze['runtime_arguments'])
        args.update(max_patches=8, turns=1, max_tokens=8, compress_before_question=True,
                    samples=remote_path(root, folder / 'calibration_select32.json' if kind == 'continuous' else quiet_samples))
        config = dict(schema='trajkv.round8.final-bounded-measurement.v1',
            backend_module='round8_postcal_runner', backend_family='postcal', arguments=args,
            variants=[deepcopy(all_variants[n]) for n in freeze[key]],
            method_freeze_sha256=freeze_sha,
            frozen_variant_contract=dict(path=remote_path(root, freeze_path), sha256=freeze_sha))
        if kind == 'continuous':
            assert len(config['variants']) <= 4
            config.update(count=8, balance_datasets=True, questions_per_source=1, target_tokens=8,
                save_reference_logits=False,
                diagnostic_references=root + '/configs/data/round8_calibration_select32_references_v1.json')
        else:
            assert len(config['variants']) == 6
            config.update(images=3, warmups=1, repeats=3, tokens_per_turn=8)
        path = folder / (name + '.json')
        write_new(path, config)
        job_path = folder / ('job_' + name + '.json')
        job = make_job(template, 'round8_' + name, 'round8_final_measurement.py', path, name, 0, root)
        job['argv'][2:2] = ['--kind', kind]
        if kind == 'timing':
            job['prerequisites'].append('Both GPUs otherwise idle before every quiet trial; do not overlap training or QA')
        write_new(job_path, job)
        result.append(dict(kind=kind, config=str(path.relative_to(ROOT)), config_sha256=sha(path),
            job=str(job_path.relative_to(ROOT)), job_sha256=sha(job_path)))
    return result


def execute(freeze_path, expected_sha):
    assert sha(freeze_path) == expected_sha, 'Exact method freeze SHA required'
    freeze = read(freeze_path)
    assert freeze['status'].startswith('FROZEN') and 'round8' in freeze['schema']
    first_plan = ROOT / 'configs/a2/round8_v1/confirmation_first_question_plan_v1.json'
    assert sha(first_plan) == freeze['first_question_plan_sha256']
    plan = read(first_plan)
    all_variants = {v['name']: v for v in freeze['variants']}
    assert len(all_variants) == len(freeze['variants'])
    root = freeze['remote_project_root']
    args = deepcopy(freeze['runtime_arguments'])
    args.update(max_patches=8, turns=1, max_tokens=64, compress_before_question=True)
    template = read(ROOT / 'configs/a2/round8_v1/job_natural_dev64_mean1_v1_doc.json')
    prepared, selected = [], []
    for pool in POOLS:
        sample_path = ROOT / ('configs/a2/round8_v1/' + pool + '.json')
        reference_path = ROOT / ('configs/data/round8_' + pool + '_references_v1.json')
        materialized = read(ROOT / ('audit/a2/round8_' + pool + '_materialized_v1.json'))
        assert materialized['method_freeze_sha256'] == expected_sha
        assert materialized['samples_sha256'] == sha(sample_path)
        assert materialized['references_sha256'] == sha(reference_path)
        rows, refs = read(sample_path), read(reference_path)
        by_id = {r['id']: r for r in refs}
        expected = {r['image_sha256']: r for r in plan['selected'] if r['pool'] == pool}
        assert len(rows) == len(expected)
        samples, references = [], []
        for original in rows:
            row, ref = deepcopy(original), deepcopy(by_id[original['id']])
            image_sha = row['hashes'][0]
            assert image_sha in expected and len(row['hashes']) == 1
            if pool == 'text_validation128':
                assert str(row['source_question_id']) == expected[image_sha]['question_id']
                assert len(ref['answers']) == 10
            else:
                assert str(row['source_question_ids'][0]) == expected[image_sha]['question_id']
                assert str(ref['references'][0]['question_id']) == expected[image_sha]['question_id']
                row['source_question_ids'] = row['source_question_ids'][:1]
                ref['references'] = ref['references'][:1]
            row['questions'], row['turns'] = row['questions'][:1], 1
            row['first_turn_scored'] = True
            row['question_selection_rule'] = 'first original membership question, fixed before GT materialization'
            assert row['content'][-1]['text'] == row['questions'][0]
            assert not any(key in row for key in ('answer', 'answers', 'references', 'ground_truth'))
            samples.append(row); references.append(ref)
            selected.append(dict(pool=pool, image_sha256=image_sha,
                question_id=expected[image_sha]['question_id']))
        name = 'confirm_firstq4_v1_' + pool
        first_samples = ROOT / ('configs/a2/round8_v1/' + name + '_samples.json')
        first_refs = ROOT / ('configs/data/round8_' + name + '_references.json')
        config_path = ROOT / ('configs/a2/round8_v1/' + name + '.json')
        job_path = ROOT / ('configs/a2/round8_v1/job_' + name + '.json')
        write_new(first_samples, samples); write_new(first_refs, references)
        variant_names = freeze['confirmation_variant_names']
        assert len(variant_names) == 4
        config = dict(schema='trajkv.round8.frozen-confirmation.v1', backend_module='round8_postcal_runner',
            backend_family='postcal', arguments={**args, 'samples': remote_path(root, first_samples)},
            start=0, count=len(samples), variants=[deepcopy(all_variants[n]) for n in variant_names],
            method_freeze_sha256=expected_sha, source_pool=pool,
            evidence_role='fresh source first-question independent QA after exact pre-GT method freeze')
        write_new(config_path, config)
        job = make_job(template, 'round8_' + name, 'round8_postcal_runner.py', config_path, name,
                       0 if pool == 'text_validation128' else 1, root)
        write_new(job_path, job)
        prepared.append(dict(pool=pool, samples=str(first_samples.relative_to(ROOT)),
            samples_sha256=sha(first_samples), references=str(first_refs.relative_to(ROOT)),
            references_sha256=sha(first_refs), config=str(config_path.relative_to(ROOT)),
            config_sha256=sha(config_path), job=str(job_path.relative_to(ROOT)),
            job_sha256=sha(job_path), sources=len(samples), questions=len(samples)))
    assert len(selected) == plan['sources'] == plan['questions'] == 256
    selected_keys = {(r['pool'], r['image_sha256'], r['question_id']) for r in selected}
    assert selected_keys == {(r['pool'], r['image_sha256'], r['question_id']) for r in plan['selected']}
    measurements = make_measurements(freeze, freeze_path, expected_sha, all_variants, template)
    receipt = dict(status='PREPARED_LOCAL_NO_INFERENCE', schema='trajkv.round8.final-protocol-preparation.v1',
        method_freeze_sha256=expected_sha, first_question_plan_sha256=sha(first_plan),
        preparer_sha256=sha(__file__), pools=prepared, measurements=measurements, sources=256, questions=256,
        variants_per_question=4, outcome_based_question_selection=False, GPU_executed=False,
        uploaded=False, original_materialized_files_unchanged=True)
    out = ROOT / 'audit/a2/round8_confirmation_firstq4_prepared_v1.json'
    write_new(out, receipt)
    print(json.dumps(dict(status=receipt['status'], receipt=str(out.relative_to(ROOT)), sources=256, questions=256)), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    args = parser.parse_args()
    execute(args.freeze, args.expected_sha256)


if __name__ == '__main__':
    main()
