"""Owner-only compatibility on preserved completed GPU records; no GT scoring."""
import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

from analyze_round7_quality import ROOT, read, sha, audit_owner


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    buckets, receipts, cache = defaultdict(list), [], {}
    example = None
    for run in a.runs:
        if (run / 'quality/records.jsonl').is_file():
            run = run / 'quality'
        assert read(run / 'complete.json')['status'] == 'COMPLETE'
        manifest = read(run / 'manifest.json')
        variants = {row['name']: row for row in manifest['config']['variants']}
        checked = 0
        with (run / 'records.jsonl').open(encoding='utf-8-sig') as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record['method'] not in ('hybrid_packed', 'base2', 'native4', 'full'):
                    continue
                assert record['variant_config'] == variants[record['variant']]
                result = audit_owner(record, variants[record['variant']], manifest.get('external_codec_inputs', {}), cache)
                buckets[record['variant']].append(result)
                checked += 1
                if record['method'] == 'hybrid_packed' and example is None:
                    example = (record, variants[record['variant']], manifest.get('external_codec_inputs', {}))
        receipts.append(dict(run=str(run), manifest_sha256=sha(run / 'manifest.json'),
            records_sha256=sha(run / 'records.jsonl'), complete_sha256=sha(run / 'complete.json'), checked_records=checked))
    assert example is not None
    record, variant, external = example
    baseline = audit_owner(record, variant, external, cache)
    negative = []
    for label, modify in [
        ('missing Hadamard', lambda x: x['owned_initial'].update(hadamard_constant=0)),
        ('double Hadamard', lambda x: x['owned_initial'].update(hadamard_constant=131072)),
        ('retained source FP16', lambda x: x['owned_initial'].update(full_visual_retained=True)),
        ('materialized quality execution', lambda x: x.update(visual_fp16_materialization=True)),
        ('configuration clipping mismatch', lambda x: x.update(clip_v=.7)),
        ('wrong reader fingerprint', lambda x: x.update(kernel_source_sha256='0' * 64)),
    ]:
        malformed = copy.deepcopy(record)
        modify(malformed)
        try:
            audit_owner(malformed, variant, external, cache)
        except ValueError as error:
            negative.append(dict(control=label, status='PASS', rejected=str(error)))
        else:
            raise RuntimeError('Malformed hybrid owner was accepted: ' + label)
    diagnostic = copy.deepcopy(record)
    diagnostic['owned_initial']['total_unique'] += 987654321
    assert audit_owner(diagnostic, variant, external, cache)['complete_visual_owner_bytes'] == baseline['complete_visual_owner_bytes']
    # Independent named-category addition control, without fabricating a new
    # execution or any task score. These are byte-accounting test inputs only.
    auxiliary = copy.deepcopy(record)
    auxiliary['method'] = 'auxiliary_cpu_control'
    auxiliary['variant'] = 'auxiliary_cpu_control'
    auxiliary['variant_config'] = dict(variant, name='auxiliary_cpu_control', method='auxiliary_cpu_control')
    amounts = {'hadamard_constant': 1, 'axis_packed_reference': 2, 'calib_packed_reference': 3,
        'hybrid_packed_reference': 4, 'kvarn_packed_reference': 5, 'rotation_packed_reference': 6,
        'anchor_visual_auxiliary': 7, 'spectrum_experiment_reference': 8}
    auxiliary['owned_initial'].update(amounts)
    tested = audit_owner(auxiliary, auxiliary['variant_config'], external, cache)
    assert tested['auxiliary_visual_owner_bytes'] == sum(amounts.values())
    table = []
    for name, rows in sorted(buckets.items()):
        table.append(dict(variant=name, method=rows[0]['method'], checked_records=len(rows),
            complete_bits=16 * sum(row['complete_visual_owner_bytes'] for row in rows) / sum(row['visual_fp16_denominator'] for row in rows),
            minimum_bits=min(row['actual_visual_bits'] for row in rows), maximum_bits=max(row['actual_visual_bits'] for row in rows),
            auxiliary_bytes_observed=sorted({row['auxiliary_visual_owner_bytes'] for row in rows}),
            role=rows[0]['role'], reconstructed_from_shapes=all(row['actual_packed_owner_reconstructed'] for row in rows)
                if rows[0]['method'] != 'full' else None))
    result = dict(schema='trajkv.round7.preserved-reference-owner-compatibility.v1', status='PASS',
        runs=receipts, table=table, negative_controls=negative, total_unique_not_added=True,
        all_explicit_auxiliary_categories_added_once_CPU_control=True,
        GT_reference_files_opened=False, task_scores_computed=False, GPU_used=False,
        original_records_modified=False, scope='CPU byte/configuration/provenance compatibility on already-completed GPU evidence; no new model measurement',
        analyzer_sha256=sha(Path(__file__).with_name('analyze_round7_quality.py')), source_sha256=sha(__file__))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    print(json.dumps(dict(status='PASS', checked_records=sum(row['checked_records'] for row in receipts),
        negative_controls=len(negative), GT_references_opened=False, table=table)))


if __name__ == '__main__':
    main()
