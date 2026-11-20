"""CPU-only integrity/failure fixtures derived from a frozen raw schema.

These fixtures are not new inference and never contribute benchmark scores.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from score import score_run, validate_raw, IntegrityError


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_fixture(target, manifest, rows):
    target.mkdir(parents=True, exist_ok=False)
    cfg = dict(manifest['config']); cfg['ids'] = [r['id'] for r in rows]
    manifest = {**manifest, 'config': cfg, 'expected_ids': cfg['ids'], 'expected_QA': len(rows),
                'scope': 'CPU_LOCAL_FIXTURE_ONLY_NOT_INFERENCE'}
    (target / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    (target / 'records.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    counts = {}
    for row in rows: counts[row['status']] = counts.get(row['status'], 0) + 1
    complete = dict(status='PASS_RAW_COMPLETE' if counts.get('PASS') == len(rows) else 'COMPLETE_WITH_FAILURES',
                    expected_QA=len(rows), recorded_QA=len(rows), counts=counts,
                    coverage_complete=True, quality_scored=False,
                    manifest_sha256=sha(target / 'manifest.json'), records_jsonl_sha256=sha(target / 'records.jsonl'))
    (target / 'complete.json').write_text(json.dumps(complete), encoding='utf-8')
    receipt = dict(status='CPU_LOCAL_FIXTURE_ONLY', files=[dict(file=name, bytes=(target / name).stat().st_size,
                    sha256=sha(target / name)) for name in ('manifest.json', 'records.jsonl', 'complete.json')])
    (target / 'retrieval_receipt.json').write_text(json.dumps(receipt), encoding='utf-8')


def main():
    source = Path('${EVIDENCE_ROOT}/results/p1/M1_F_textvqa_p1_128_v1')
    parent = Path('${EVIDENCE_ROOT}/results/scorer_gate') / str(time.time_ns())
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in (source / 'records.jsonl').read_text(encoding='utf-8').splitlines()][:2]
    original = {name: sha(source / name) for name in ('manifest.json', 'records.jsonl', 'complete.json')}
    refs = ROOT / 'configs/datasets/v3/textvqa_p1_refs.jsonl'
    passed = []
    valid = parent / 'valid'
    make_fixture(valid, manifest, rows)
    summary = score_run(valid, refs, parent / 'valid_scored', allow_fixture=True, repetitions=1000)
    assert summary['status'] == 'PASS_SCORED_COMPLETE' and summary['primary_mean'] is not None
    passed.append('complete-valid-subset-scores-with-full-denominator')
    failed = parent / 'one_failure'
    bad = dict(rows[1], status='FAILED_RUN', raw_prediction=None, output_token_ids=None, error='CPU_FIXTURE')
    make_fixture(failed, manifest, [rows[0], bad])
    summary = score_run(failed, refs, parent / 'failure_scored', allow_fixture=True, repetitions=1000)
    assert summary['primary_mean'] is None and summary['primary_ci95'] is None
    assert summary['groups']['overall']['conditional_success_only']['questions'] == 1
    assert summary['groups']['overall']['failure_questions'] == 1
    passed.append('failure-retained-primary-null-conditional-separately-labeled')
    with (valid / 'records.jsonl').open('ab') as stream: stream.write(b' ')
    try: validate_raw(valid, refs, allow_fixture=True)
    except IntegrityError: passed.append('tampered-raw-rejected')
    else: raise AssertionError('Tampered raw accepted')
    duplicate = parent / 'duplicate_ids'
    make_fixture(duplicate, manifest, [rows[0], rows[0]])
    try: validate_raw(duplicate, refs, allow_fixture=True)
    except IntegrityError: passed.append('duplicate-expected-or-recorded-id-rejected')
    else: raise AssertionError('Duplicate ID accepted')
    try: validate_raw(failed, refs, allow_fixture=False)
    except IntegrityError: passed.append('CPU-fixture-cannot-pass-real-remote-receipt-contract')
    else: raise AssertionError('CPU fixture accepted as remote inference')
    assert all(sha(source / name) == digest for name, digest in original.items())
    passed.append('original-GPU-raw-preserved-byte-for-byte')
    result = {'status': 'PASS_CPU_SCORER_INTEGRITY_AND_FAILURE_GATE', 'cases': passed,
              'fixture_root': str(parent), 'GPU_inference_performed': False,
              'fixture_scores_are_benchmark_results': False, 'source_raw_sha256': original,
              'scorer_sha256': sha(ROOT / 'code/score.py'), 'gate_sha256': sha(__file__)}
    out = ROOT / 'audit/offline_scorer_integrity_gate_v1.json'
    with out.open('x', encoding='utf-8') as stream: json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__': main()
