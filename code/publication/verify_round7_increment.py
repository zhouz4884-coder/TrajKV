"""Verify derived Round7 archive/content refs and independently recount scores/owners.

No Git/network/model execution. Optional tree JSON comes from the publisher's
actual staged/committed Git tree, rather than a working-directory inventory.
"""
import argparse
import csv
from collections import defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/a2'), str(ROOT / 'code/analysis')]
from analyze_external import score_docvqa
from textvqa_metric import TextVQAAccuracyEvaluator
from prepare_round5_delivery import secret_scan
from round7_content_index import verify as verify_content, safe_name, file_digest

OWNER = ('visual_base', 'visual_residual', 'selector_metadata', 'visual_position', 'hadamard_constant',
         'axis_packed_reference', 'calib_packed_reference', 'hybrid_packed_reference', 'kvarn_packed_reference',
         'rotation_packed_reference', 'anchor_visual_auxiliary', 'spectrum_experiment_reference')


def require(value, message):
    if not value:
        raise ValueError(message)


def close(left, right, context):
    require(math.isfinite(float(left)) and math.isfinite(float(right))
            and math.isclose(float(left), float(right), rel_tol=1e-11, abs_tol=1e-10), context)


def rel(value):
    text = str(value).replace('\\', '/')
    return text.split('/TrajKV/', 1)[1] if '/TrajKV/' in text else text


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def recount_owner(record, manifest, get, entries):
    owner, nv = record['owned_initial'], int(record['visual_tokens'])
    encoder = int(record.get('encoder_calibration_parameters_bytes', 0))
    total = sum(int(owner.get(key, 0)) for key in OWNER) + encoder
    if record['method'].startswith('kvarn') and not owner.get('hadamard_constant'):
        total += 65536
    denominator = int(owner['visual_fp16_denominator'])
    require(nv > 0 and denominator > 0 and owner['visual_position'] == 8 * nv, 'Invalid real visual denominator/positions')
    for where in (owner, record):
        for key in ('complete_visual_owner_bytes', 'depth_residual_complete_bytes'):
            if key in where:
                require(int(where[key]) == total, 'Declared complete owner differs from raw component recount')
    for key in ('actual_visual_bits_including_encoder', 'depth_residual_actual_budget_bits'):
        if key in record:
            close(record[key], 16 * total / denominator, 'Actual complete bit rate differs')
    method = record['method']
    lh = denominator // (nv * 512)
    require(denominator == lh * nv * 512, 'Non-D128 visual denominator')
    require(record.get('padded_visual_tokens', 0) == 0 and record.get('padded_visual_positions', 0) == 0, 'Semantic visual padding')
    if method in ('base2', 'native4'):
        require(owner['visual_base'] == lh * nv * (72 if method == 'base2' else 136), 'KIVI-style code/metadata owner mismatch')
    elif method == 'hybrid_packed':
        codec = record['variant_config'].get('key_codec', 'kvarn2')
        require(nv % 128 == 0 and owner.get('full_visual_retained') is False
                and record.get('visual_fp16_materialization') is False, 'Hybrid is not actual packed complete-g128 execution')
        require(owner.get('hadamard_constant', 0) == (65536 if codec == 'kvarn2' else 0), 'Hybrid Hadamard charge missing/doubled')
        require(owner['visual_base'] == lh * (nv * (54 if codec == 'kvarn2' else 52) + 512), 'Hybrid code/metadata owner mismatch')
    elif method in ('round7_key', 'round7_tail_key', 'round6_depth'):
        variant = record['variant_config']
        def artifact(key):
            name = rel(variant[key]); expected = manifest['external_codec_inputs'][variant[key]]
            require(name in entries and entries[name]['source_sha256'] == expected,
                    'Missing original codec fingerprint/content mapping: ' + name)
            return json.loads(get(name))
        factor, book = artifact('metadata_factor_path'), artifact('direction_book_path')
        words = len(book['centers']); index_bytes = 16 if words == 16 else 32
        require(words in (16, 256), 'Unsupported K word count')
        window = nv if variant.get('v_group_size', 'all') == 'all' else int(variant['v_group_size'])
        physical = ((nv + 31) // 32) * 32 if method == 'round7_tail_key' else nv
        base = lh * (nv * (index_bytes + 2) + physical * 16) + lh * 256 + lh * 512 * (nv // window) + words * 8
        require(owner['visual_base'] == base, 'New Key/tail actual base owner mismatch')
        metric = 0
        if variant.get('key_metric_path'):
            value = artifact('key_metric_path')
            require(sum(len(v) for layer in value['u'] for v in layer) == lh * 128, 'Rank-one metric geometry mismatch')
            metric = lh * 260
            require(record['key_metric_bytes'] == metric, 'FP16-u/FP32-gamma charge mismatch')
        factors = 4 * sum(sum(len(row) for row in factor[key]) for key in ('key_head_factors', 'value_head_factors', 'value_center_offsets') if factor.get(key) is not None)
        require(encoder == factors + metric + record['depth_residual_encoder_ranking_bytes'] + record['depth_residual_encoder_policy_bytes'], 'Encoder constants omitted/doubled')
        require(owner['visual_residual'] == len(record['depth_residual_selected_stages']) * nv * 18
                and owner['selector_metadata'] == lh * 16, 'Residual/lookup storage mismatch')
        if method == 'round7_tail_key':
            require(record['value_tail_padding_actual_bytes'] == (physical - nv) * lh * 16
                    and record['value_centroid_fit_real_tokens_only'] and record['visual_positions_exact_original'], 'Physical tail storage/semantic scope mismatch')
    return total, denominator


def quality_check(spec, get, entries):
    summary = json.loads(get(spec['summary']))
    require(summary['status'] == 'COMPLETE' and summary['audit']['status'] == 'PASS', 'Incomplete quality summary')
    text = get(spec['references']).decode('utf-8-sig')
    references = (json.loads(text) if text.lstrip().startswith('[') else [json.loads(line) for line in text.splitlines() if line.strip()])
    refs = {row['id']: row for row in references}
    scores = list(csv.DictReader(io.StringIO(get(spec['scores']).decode('utf-8-sig'))))
    metric, scored, groups = TextVQAAccuracyEvaluator(), {}, defaultdict(list)
    for row in scores:
        key = (row['session_id'], row['variant'], int(row['turn']))
        require(key not in scored and row['variant'] in spec['selected_variants'], 'Duplicate/unselected scored answer')
        target = (refs[key[0]]['answers'] if spec['dataset'] == 'text' else
                  next(value.get('answer', value.get('answers')) for value in refs[key[0]]['references'] if int(value['turn']) == key[2]))
        require(json.loads(row['reference']) == target, 'Scored references differ from original selected labels')
        actual = (metric.eval_pred_list([dict(pred_answer=row['prediction'], gt_answers=target)]) if spec['dataset'] == 'text'
                  else score_docvqa(row['prediction'], target)['score'])
        close(row['score'], actual, 'Official score recomputation differs')
        groups[row['variant']].append(float(actual)); scored[key] = row
    stores, denominators, raw_keys, original_records = defaultdict(int), defaultdict(int), set(), 0
    for filename in spec['raw_records']:
        folder = str(Path(filename).parent).replace('\\', '/')
        manifest = json.loads(get(folder + '/manifest.json'))
        require('cached-prefix' not in str(manifest.get('quality_scope', '')).lower(), 'Cached-prefix calibration mixed into quality')
        require(json.loads(get(folder + '/complete.json'))['status'] == 'COMPLETE', 'Incomplete producer')
        for line in get(filename).splitlines():
            if not line.strip():
                continue
            record = json.loads(line); original_records += 1
            require('free_running' in record.get('protocol', '') and record.get('task_accuracy_eligible') is not False,
                    'Non-free diagnostic mixed into quality run')
            amount, denominator = recount_owner(record, manifest, get, entries)
            if record['variant'] not in spec['selected_variants']:
                continue
            stores[record['variant']] += amount; denominators[record['variant']] += denominator
            for answer in record['turns']:
                key = (record['session_id'], record['variant'], int(answer['turn']))
                require(key in scored and key not in raw_keys and answer['answer'] == scored[key]['prediction'], 'Raw/scored answer coverage or identity mismatch')
                raw_keys.add(key)
    require(raw_keys == set(scored), 'Raw/scored question union differs')
    for row in summary['table']:
        name = row['variant']
        close(statistics.mean(groups[name]), row.get('score', row.get('anls') if spec['dataset'] == 'doc' else row.get('accuracy')), 'Quality mean mismatch')
        close(16 * stores[name] / denominators[name], row['budget_bits_including_encoder_constants'], 'Aggregate complete owner mismatch')
    return dict(dataset=spec['dataset'], analysis_role=spec['analysis_role'], variants=len(groups),
        scored_answers=len(scored), original_producer_records_verified=original_records, score_and_owner_recomputed=True)


def verify(assets, tree_map=None):
    delivery = load(assets / 'DELIVERY.json')
    archive_path = assets / delivery['archive']; index_path = assets / 'ROUND7_SOURCE_INDEX.json'
    content_path = assets / 'ROUND7_CONTENT_REFS.json'; index = load(index_path)
    require(index['schema'] == 'trajkv.round7.public.increment.v1' and index['status'] == 'PASS', 'Wrong/incomplete R7 source index')
    require(file_digest(archive_path) == delivery['archive_sha256'] and file_digest(index_path) == delivery['source_index_sha256']
            and file_digest(content_path) == delivery['content_index_sha256'], 'Prepared delivery identity changed')
    content = verify_content(SimpleNamespace(verify_index=content_path, archive=archive_path,
        payload=assets / 'payload', git_tree_map=tree_map))
    source_verified, derived_verified = 0, 0
    with zipfile.ZipFile(archive_path) as archive:
        require(len(archive.namelist()) == len(set(archive.namelist())), 'Duplicate ZIP members')
        require(set(archive.namelist()) == set(index['files']) | {'ROUND7_SOURCE_INDEX.json'}, 'Archive/index inventory mismatch')
        require(archive.read('ROUND7_SOURCE_INDEX.json') == index_path.read_bytes(), 'Inner/outer source index differs')
        def get(name):
            name = rel(name)
            require(name in index['files'], 'Unindexed evidence dependency: ' + name)
            return archive.read(name)
        for name, entry in index['files'].items():
            safe_name(name); data = get(name); secret_scan(data, name)
            require(hashlib.sha256(data).hexdigest() == entry['public_sha256'] and len(data) == entry['bytes'], 'Public exact content changed')
            source = ROOT / entry.get('source_path', name)
            if source.is_file():
                actual = file_digest(source)
                if actual == entry['source_sha256']:
                    source_verified += 1
                elif actual in (entry['public_sha256'], load(content_path)['files'][name]['git_normalized_body_sha256']):
                    derived_verified += 1
                else:
                    raise ValueError('Existing source/public materialization differs from all pinned identities: ' + name)
        quality = [quality_check(spec, get, index['files']) for spec in index['checks']['quality_analyses']]
    return dict(status='PASS', archive_sha256=file_digest(archive_path), source_index_sha256=file_digest(index_path),
        content_index_sha256=file_digest(content_path), indexed_files=len(index['files']),
        original_source_files_verified=source_verified, derived_public_files_verified=derived_verified,
        quality_recomputations=quality, content_identity_check=content,
        cached_prefix_calibration_not_E2E=True, failed_smokes_excluded_from_quality=True,
        credential_findings=0, Git_operations=0, network_operations=0, GPU_used=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--git-tree-map', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args(); result = verify(a.assets.resolve(), a.git_tree_map)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        with a.output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
    print(json.dumps(dict(status=result['status'], indexed_files=result['indexed_files'],
        quality_analyses=len(result['quality_recomputations']), actual_git_tree_verified=result['content_identity_check']['actual_git_tree_verified'])))
