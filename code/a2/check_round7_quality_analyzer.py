"""CPU acceptance/negative controls; never synthetic task-performance evidence."""
import argparse
import copy
import json
from pathlib import Path

from analyze_round7_quality import ROOT, read, sha, audit_owner, independent_plan


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, default=ROOT / 'results/round7_v1/key_smoke2')
    p.add_argument('--metric', type=Path, default=ROOT / 'results/round7_v1/metric8/metric_cap16.json')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    manifest = read(a.run / 'manifest.json')
    record = next(json.loads(line) for line in (a.run / 'records.jsonl').read_text(encoding='utf-8').splitlines()
                  if json.loads(line)['variant'] == 'r7_k2_norm')
    variant = record['variant_config']
    baseline = audit_owner(record, variant, manifest['external_codec_inputs'], {})
    tests = []
    def reject(label, mutate):
        value = copy.deepcopy(record)
        mutate(value)
        try:
            audit_owner(value, variant, manifest['external_codec_inputs'], {})
        except ValueError as error:
            tests.append(dict(control=label, expected='reject', status='PASS', reason=str(error)))
        else:
            raise RuntimeError('Failed to reject malformed cost/protocol evidence: ' + label)
    reject('index bytes mislabeled as nibble without physical representation change', lambda x: x.update(key_index_bytes_per_token=16))
    reject('encoder constant duplicated', lambda x: x.update(encoder_calibration_parameters_bytes=x['encoder_calibration_parameters_bytes'] * 2))
    reject('one visual position byte omitted', lambda x: x['owned_initial'].update(visual_position=x['owned_initial']['visual_position'] - 1))
    reject('residual mask changed despite same reference configuration', lambda x: x['depth_residual_selected_stages'].reverse())
    reject('semantic visual padding hidden as storage', lambda x: x.update(padded_visual_tokens=1))
    diagnostic = copy.deepcopy(record)
    diagnostic['owned_initial']['total_unique'] += 1234567
    same_owner = audit_owner(diagnostic, variant, manifest['external_codec_inputs'], {})
    assert same_owner['complete_visual_owner_bytes'] == baseline['complete_visual_owner_bytes']
    tests.append(dict(control='CPU/global total_unique is not added to visual cache', expected='same complete owner', status='PASS'))
    metric = read(a.metric)
    layers, heads, dimension = len(metric['u']), len(metric['u'][0]), len(metric['u'][0][0])
    metric_bytes = layers * heads * (dimension * 2 + 4)
    assert (layers, heads, dimension, metric_bytes) == (28, 4, 128, 29120)
    assert metric.get('tensor_bytes') == metric_bytes
    tests.append(dict(control='rank-one FP16 u plus FP32 gamma payload', expected=29120, actual=metric_bytes, status='PASS'))
    nv, full = record['visual_tokens'], record['owned_initial']['visual_fp16_denominator']
    base_position = record['owned_initial']['visual_base'] + 8 * nv
    k2 = independent_plan(variant, layers, heads, nv, base_position, 1344, 0, 256, full)
    k1_variant = copy.deepcopy(variant)
    k1_variant['residual_layout_reference'] = 'k2'
    k1_base = base_position - layers * heads * nv * 16 - 2048 + 128
    k1 = independent_plan(k1_variant, layers, heads, nv, k1_base, 1344, 0, 16, full)
    assert k1['stages'] == k2['stages']
    assert k2['complete'] - k1['complete'] == layers * heads * nv * 16 + 2048 - 128
    tests.append(dict(control='K1 fixed K2 mask preserves stages while charging smaller real key payload',
        expected='same mask and exact physical key/book byte saving', status='PASS', byte_saving=k2['complete'] - k1['complete']))
    result = dict(schema='trajkv.round7.quality-analyzer-cpu-acceptance.v1', status='PASS', controls=tests,
        GPU_used=False, inference_executed=False, synthetic_task_scores_created=False,
        scope='malformed-evidence rejection and scalar byte algebra controls; not GPU reader validation or empirical accuracy',
        real_record_source=str(a.run), real_records_sha256=sha(a.run / 'records.jsonl'),
        metric_source_sha256=sha(a.metric), analyzer_sha256=sha(Path(__file__).with_name('analyze_round7_quality.py')))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(dict(status='PASS', controls=len(tests), GPU_used=False, output=str(a.output))))


if __name__ == '__main__':
    main()
