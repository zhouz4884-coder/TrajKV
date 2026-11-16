"""Freeze bounded development choices, then prepare answer-free two-GPU jobs.

New cohort membership/annotations are staged separately after the method lock.
Never launches jobs, changes old evidence, or uploads reference annotations.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from remote import REMOTE_ROOT as R, PYTHON as P, connect, upload

FOLDER = ROOT / 'configs/a2/round6_v1'
MARKER = ROOT / 'audit/a2/round6_confirmation_v1.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)


def freeze():
    depth = read(FOLDER / 'depth_doc64.json')['variants']
    rd = read(FOLDER / 'depth_rd_dev64.json')['variants']
    previous = {v['name']: v for v in read(ROOT / 'audit/a2/round5_confirmation_v1.json')['variants']}
    controls = read(FOLDER / 'controls_doc64.json')['variants']
    names = ['r6_depth_v_pair', 'r6_depth_incumbent', 'r6_depth_RD_actual']
    found = {v['name']: v for v in depth + rd}
    variants = [copy.deepcopy(found[name]) for name in names]
    variants += [copy.deepcopy(previous[name]) for name in ('v_impact_max', 'full8', 'native4_8', 'base2_8')]
    variants += [copy.deepcopy(v) for v in controls if v['name'] == 'r6_axis_matched_spread']
    if len(variants) != 8:
        raise ValueError('Expected exactly eight predeclared candidate/control variants')
    summaries = {}
    for name in ('controls_doc64_analysis', 'depth_doc64_analysis', 'gamma_doc64_analysis',
                 'directional_doc64_analysis', 'depth_rd_doc64_analysis', 'depth_text64_seen_analysis_v1'):
        path = ROOT / 'results/round6_v1' / name / 'summary.json'
        data = read(path)
        summaries[name] = dict(sha256=sha(path), table=data['table'])
    table = {r['variant']: r for name, data in summaries.items() if 'text' not in name for r in data['table']}
    # Bounded comparison: original mixed, pair depth, and calibrated RD depth.
    candidates = ['r6_depth_incumbent', 'r6_depth_v_pair', 'r6_depth_RD_actual']
    primary = max(candidates, key=lambda name: (table[name]['anls'], -table[name]['budget_bits_including_encoder_constants']))
    inference = set(read(ROOT / 'audit/a2/round5_confirmation_v1.json')['inference_source_sha256'])
    inference.update(['code/a2/' + name for name in ['round6_runner.py', 'round6_depth_bank.py', 'round6_depth_attention.py', 'round6_depth_codec.py', 'round6_timing.py']])
    sources = {name: sha(ROOT / name) for name in sorted(inference)}
    external = {}
    for variant in variants:
        for key, value in variant.items():
            if key.endswith('_path') and isinstance(value, str) and value.startswith(R + '/'):
                external[value] = sha(ROOT / value[len(R) + 1:])
    value = dict(schema='trajkv.round6.fresh-confirmation.v1',
        created_at_utc=datetime.now(timezone.utc).isoformat(), primary=primary,
        variants=variants, candidate_names=candidates,
        selection_rule='highest complete DocVQA dev64 ANLS among incumbent/pair/actual-RD; all retained for matched confirmation',
        development=summaries, inference_source_sha256=sources,
        external_codec_inputs_sha256=external,
        doc_scope='all 54 remaining source/content-disjoint source documents, fixed hash question order, up to six questions',
        text_scope='256 source/content-disjoint images, salt fixed before annotations and outcomes',
        primary_comparisons=[['r6_depth_v_pair', 'r6_depth_incumbent'],
                             ['r6_depth_v_pair', 'r6_depth_RD_actual'],
                             ['r6_depth_v_pair', 'v_impact_max'],
                             ['r6_depth_v_pair', 'native4_8']],
        inference_protocol='real free-running image-boundary compressed InternVL3-8B FP16 backbone; text KV FP16',
        policy='no further method selection on fresh scores; reports all frozen variants and source-cluster uncertainty',
        cost='actual packed visual owner, positions, lookup, ranking and specified encoder constants; complete budget <=1.8 only for eligible variants',
        transfer_rule='frozen Doc calibration/ranking, no Text fitting',
        metrics='DocVQA ANLS; official TextVQA normalized leave-one-out score; paired source-image cluster bootstrap/sign flip',
        answers_uploaded=False)
    write_new(MARKER, value)
    print(json.dumps(dict(status='FROZEN', path=str(MARKER), sha256=sha(MARKER), primary=primary, variants=len(variants))))


def prepare():
    frozen = read(MARKER)
    for name, digest in frozen['inference_source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Inference source changed after freeze: ' + name)
    base = read(FOLDER / 'depth_doc64.json')
    outputs = []
    for cohort, count, questions in [('doc54', 54, 6), ('text256', 256, 1)]:
        sample = 'doc_confirm54.json' if cohort == 'doc54' else 'text_transfer256.json'
        runner = FOLDER / sample
        membership = ROOT / 'audit/a2' / ('round6_doc54_membership.json' if cohort == 'doc54' else 'round6_text_membership_v1.json')
        audit = read(membership)
        if audit['method_freeze_sha256'] != sha(MARKER) or len(read(runner)) != count:
            raise ValueError('Staged cohort differs from frozen method/membership')
        for shard in (0, 1):
            cfg = copy.deepcopy(base)
            cfg.update(variants=frozen['variants'], start=shard * (count // 2), count=count // 2,
                       frozen_contract_sha256=sha(MARKER))
            cfg['arguments'].update(samples=R + '/' + runner.relative_to(ROOT).as_posix(), turns=questions)
            cp = FOLDER / f'confirm_{cohort}_s{shard}.json'
            write_new(cp, cfg)
            output = R + f'/results/round6_v1/confirm_{cohort}_s{shard}'
            job = dict(name=f'round6_confirm_{cohort}_s{shard}_v1', env={'CUDA_VISIBLE_DEVICES': str(shard)},
                       argv=[P, R + '/code/a2/round6_runner.py', '--config', R + '/' + cp.relative_to(ROOT).as_posix(), '--output', output])
            jp = FOLDER / f'job_confirm_{cohort}_s{shard}.json'
            write_new(jp, job)
            outputs.append(str(jp.relative_to(ROOT)))
    print(json.dumps(dict(status='PREPARED_NOT_LAUNCHED', jobs=outputs)))


def transport():
    frozen = read(MARKER)
    for name, digest in frozen['inference_source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Source drift before transport: ' + name)
    manifests = [read(ROOT / 'audit/a2' / name) for name in ('round6_doc54_membership.json', 'round6_text_membership_v1.json')]
    c = connect()
    sftp = c.open_sftp()
    evidence = []
    try:
        for manifest in manifests:
            for entry in manifest['images_manifest']:
                local = Path(entry['local'])
                if sha(local) != entry['sha256']:
                    raise ValueError('Local staged image changed')
                upload(c, local, entry['remote'])
                digest = hashlib.sha256()
                with sftp.open(entry['remote'], 'rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(block)
                if digest.hexdigest() != entry['sha256']:
                    raise ValueError('Remote image differs')
                evidence.append(dict(remote=entry['remote'], sha256=digest.hexdigest(), bytes=entry['bytes']))
            source = ROOT / manifest['samples']
            upload(c, source, R + '/' + manifest['samples'])
        for path in (ROOT / 'code/a2').glob('*.py'):
            upload(c, path, R + '/' + path.relative_to(ROOT).as_posix())
        for path in FOLDER.glob('confirm_*.json'):
            upload(c, path, R + '/' + path.relative_to(ROOT).as_posix())
    finally:
        sftp.close()
        c.close()
    path = ROOT / 'audit/a2/round6_confirmation_transport_v1.json'
    write_new(path, dict(status='PASS', images=len(evidence), remote_images=evidence,
                         method_freeze_sha256=sha(MARKER), answers_uploaded=False,
                         scope='raw staged image bytes rehashed from remote; answer-free runners only; no reference transport'))
    print(json.dumps(dict(status='PASS', images=len(evidence), answers_uploaded=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'prepare', 'transport'])
    args = parser.parse_args()
    dict(freeze=freeze, prepare=prepare, transport=transport)[args.action]()
