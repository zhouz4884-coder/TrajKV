"""Summarize only the frozen Doc64 primary family and separate mechanisms.

Inference identities are checked against the declared inference-only hash
list and launch snapshots. Subsequent analysis-code additions are permitted.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
import struct
from collections import defaultdict
from pathlib import Path

from analyze import paired, write_csv
from analyze_docvqa_confirm64 import holm
from analyze_round5 import actual_visual_owner
from analyze_external import score_docvqa

ROOT = Path(__file__).resolve().parents[2]
MARKER = ROOT/'audit/a2/round5_confirmation_v1.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_output_hashes(folder):
    hashes = {path.name: sha(path) for path in sorted(folder.iterdir())
              if path.is_file() and path.name != 'SHA256.json'}
    (folder/'SHA256.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')


def local_path(value):
    normalized = str(value).replace('\\', '/')
    marker = '/TrajKV/'
    return ROOT/normalized.split(marker, 1)[1] if marker in normalized else Path(value)


def checked_axis_mask(units):
    mask = [0]*28
    for axis, layer, head in units:
        if axis not in ('k', 'v') or not isinstance(layer, int) or not isinstance(head, int) or not 0 <= layer < 28 or not 0 <= head < 4:
            raise ValueError('Invalid layer/head/axis in packed selection mask')
        bit = 1 << (2*head+(axis == 'v'))
        if mask[layer] & bit:
            raise ValueError('Duplicate packed selection mask bit')
        mask[layer] |= bit
    return mask


def checked_runner_seed(source):
    seeds = [node.args[0].value for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and node.func.attr == 'manual_seed' and node.args and isinstance(node.args[0], ast.Constant)]
    if seeds != [1212]:
        raise ValueError('Expected one fixed manual seed 1212')
    return 1212


def frozen_asset_checks(frozen):
    if frozen.get('new_doc_outcomes_seen') is not False or frozen.get('new_text_outcomes_seen') is not False:
        raise ValueError('Confirmation was not frozen before new outcomes')
    if frozen.get('methods_change_after_confirmation') is not False:
        raise ValueError('Method-change rule differs from the freeze')
    hashes = {}
    for category in ('input_sha256', 'configuration_sha256', 'inference_source_sha256',
                     'calibration_ranking_inputs_sha256'):
        for relative, expected in frozen[category].items():
            path = ROOT/relative
            if sha(path) != expected:
                raise ValueError('Frozen '+category+' mismatch: '+relative)
            hashes[relative] = expected
    for remote, expected in frozen['external_codec_inputs'].items():
        if sha(local_path(remote)) != expected:
            raise ValueError('Frozen codec input mismatch')
        hashes[str(local_path(remote).relative_to(ROOT))] = expected
    return dict(status='PASS', marker_sha256=sha(MARKER), checked_sha256=hashes,
        source_scope='only declared inference hashes are frozen; analysis files outside this list may change')


def run_manifest_checks(folder, frozen, allowed_variants, samples_sha256):
    """Use actual immutable-launch snapshots, including non-a2 dependencies."""
    if load(folder/'complete.json')['status'] != 'COMPLETE':
        raise ValueError('Run incomplete: '+str(folder))
    manifest = load(folder/'manifest.json')
    if manifest['samples_sha256'] != samples_sha256 or manifest.get('answers_visible_to_runner') is not False:
        raise ValueError('Run sample/answer-visibility mismatch')
    if manifest['config']['variants'] != allowed_variants:
        raise ValueError('Frozen variants changed in run')
    matches = [relative for relative, digest in frozen['configuration_sha256'].items()
               if digest == manifest['config_sha256'] and '/confirm_' in relative
               and '/job_' not in relative and '/confirm_text_queue_' not in relative]
    if len(matches) != 1:
        raise ValueError('Run configuration is not uniquely frozen')
    relative = matches[0]
    if manifest['config'] != load(ROOT/relative):
        raise ValueError('Manifest config differs from its frozen config bytes')
    sample_rows = load(local_path(manifest['config']['arguments']['samples']))
    start = manifest['config'].get('start', 0)
    expected_ids = [row['id'] for row in sample_rows[start:start+manifest['config'].get('count', len(sample_rows))]]
    if manifest['selected_ids'] != expected_ids:
        raise ValueError('Run selected IDs differ from the frozen shard')
    job = 'round5_'+Path(relative).stem+'_v1'
    if job not in frozen['jobs']:
        raise ValueError('Unknown confirmation job')
    receipts = ROOT/'audit/a2/round5_v1/source_receipts'
    source_path, launch_path = receipts/(job+'.sha256.json'), receipts/(job+'.launch.json')
    snapshot, launch = load(source_path), load(launch_path)
    if launch['name'] != job or launch.get('immutable_code_execution') is not True:
        raise ValueError('Immutable launch receipt mismatch')
    if not any(str(launch['source_snapshot'])+'/' in str(x) for x in launch['argv_executed']):
        raise ValueError('Launch did not execute the recorded source snapshot')
    snapshot = {key.replace('\\', '/'): value for key, value in snapshot.items()}
    sources = {key.replace('\\', '/'): value for key, value in manifest['sources'].items()}
    for name, digest in frozen['inference_source_sha256'].items():
        if snapshot.get(name) != digest:
            raise ValueError('Launch inference source differs: '+name)
        if name.startswith('code/a2/') and sources.get(name) != digest:
            raise ValueError('Executed a2 source differs: '+name)
    # The runner sets a fixed seed. Confirm its actual executed source, even
    # though this transitive runner was not separately named in the marker.
    seed_source = ROOT/'code/a2/run_representation_variants.py'
    seed_key = 'code/a2/run_representation_variants.py'
    if snapshot.get(seed_key) != sha(seed_source) or sources.get(seed_key) != sha(seed_source):
        raise ValueError('Executed seeded runner differs from its available source')
    checked_runner_seed(seed_source.read_text())
    expected_external = {value: frozen['external_codec_inputs'][value]
        for variant in allowed_variants for key, value in variant.items()
        if key.endswith('_path') and isinstance(value, str) and value in frozen['external_codec_inputs']}
    if manifest['external_codec_inputs'] != expected_external:
        raise ValueError('Executed external codec input hashes differ')
    return manifest, dict(status='PASS', job=job, config=relative,
        manifest_sha256=sha(folder/'manifest.json'), records_sha256=sha(folder/'records.jsonl'),
        launch_sha256=sha(launch_path), launch_snapshot_sha256=sha(source_path),
        inference_files_checked=len(frozen['inference_source_sha256']), seed=1212,
        seed_scope='literal seed in actual snapshot-matched runner',
        external_codec_inputs=expected_external)


def representation_check(record, config, base_parameter_bytes):
    """Independent actual owner and axis-mask checks for frozen formats."""
    for key in ('session_id', 'variant', 'method', 'visual_tokens', 'owned_initial', 'turns', 'read_events'):
        if record.get(key) is None:
            raise ValueError('Null required record field: '+key)
    if record['variant_config'] != config or record['method'] != config['method']:
        raise ValueError('Record config differs from frozen variant')
    if record['protocol'] != 'image_boundary_free_running' or record.get('task_accuracy_eligible') is False:
        raise ValueError('Non-free or ineligible record')
    if record.get('initial_logits_shared_fp16') is not False or not record.get('read_ledger_exact'):
        raise ValueError('Question replay/read ledger check failed')
    if not record.get('global_extra_budget_ok', True):
        raise ValueError('Read budget failed')
    method, nv = record['method'], int(record['visual_tokens'])
    if nv <= 0 or nv % 128:
        raise ValueError('Invalid visual length')
    nl, nh, d = 28, 4, 128
    e = nl*nh*nv*d
    owned = record['owned_initial']
    for key in ('visual_base', 'visual_residual', 'selector_metadata', 'visual_position', 'visual_fp16_denominator'):
        value = owned.get(key)
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Null/invalid owner: '+key)
    if owned['visual_fp16_denominator'] != 4*e or owned['visual_position'] != 8*nv:
        raise ValueError('Denominator/position owner mismatch')
    if owned.get('full_visual_retained') or owned.get('diagnostic_fp16_reference'):
        raise ValueError('Unexpected source/reference visual retention')
    encoder, residual, lookup = 0, 0, 0
    mask = [0]*nl
    if method == 'full':
        expected = 4*e
    elif method in ('base2', 'native4'):
        bits = 2 if method == 'base2' else 4
        expected = e*(2*bits/8+8/128)
    elif method in ('direction_meta', 'sparse_v_residual', 'mixed_residual'):
        group = nv if config.get('v_group_size') == 'all' else 128
        expected = e*3/8+nl*nh*nv*2+nl*nh*d*2+4*nl*nh*d*(nv//group)+2048
        encoder = base_parameter_bytes
        if config.get('functional_v_factor_path') or config.get('interaction_policy_path'):
            raise ValueError('Unfrozen additional factor/policy in final check')
        if method == 'mixed_residual':
            selected = record['mixed_residual_selected_units']
            requested = config['selected_units']
            lookup = nl*2*nh*4
            encoder += 2*(2*nl*nh)
            if record['mixed_residual_dual_lookup_bytes'] != lookup or record['mixed_residual_encoder_ranking_bytes'] != 448:
                raise ValueError('Mixed lookup/ranking cost mismatch')
            packed_order = [(0 if axis == 'k' else 1)*nl*nh+int(layer)*nh+int(head)
                            for axis, layer, head in requested]
            order_set = set(packed_order)
            if len(order_set) != len(packed_order):
                raise ValueError('Duplicate frozen mixed units')
            packed_order += [value for value in range(2*nl*nh) if value not in order_set]
            digest = hashlib.sha256(struct.pack('<'+'H'*(2*nl*nh), *packed_order)).hexdigest()
            if record['mixed_residual_encoder_ranking_sha256'] != digest:
                raise ValueError('Mixed ranking bytes differ')
            if selected != requested[:len(selected)]:
                raise ValueError('Mixed selected units are not frozen priority prefix')
            mask = checked_axis_mask(selected)
            if record['mixed_residual_K_units'] != sum(a == 'k' for a, _, _ in selected) or record['mixed_residual_V_units'] != sum(a == 'v' for a, _, _ in selected):
                raise ValueError('Mixed axis counts differ')
            if record['mixed_residual_units'] != len(selected) or not record['mixed_residual_budget_verified']:
                raise ValueError('Mixed residual verification failed')
        elif method == 'sparse_v_residual':
            selected = record['sparse_residual_selected_units']
            requested = config['selected_units']
            axis = config['residual_axis']
            lookup = nl*nh*4
            encoder += 2*len(requested)
            if record['sparse_v_lookup_bytes'] != lookup or record['sparse_v_encoder_ranking_bytes'] != 2*len(requested):
                raise ValueError('Sparse lookup/ranking cost mismatch')
            if selected != requested[:len(selected)] or record['sparse_residual_axis'] != axis:
                raise ValueError('Sparse selected units differ from frozen axis/prefix')
            mask = checked_axis_mask([(axis, layer, head) for layer, head in selected])
            if record['sparse_residual_units'] != len(selected) or not record['sparse_v_budget_verified']:
                raise ValueError('Sparse residual verification failed')
        if method in ('mixed_residual', 'sparse_v_residual'):
            residual = len(selected)*nv*18
            allowed = math.floor(config['budget_bits']*(4*e)/16)
            affordable = (allowed-(expected+8*nv+lookup+encoder))//(18*nv)
            count = min(len(requested), affordable)
            limit = config.get('res_units_limit', config.get('residual_units_limit'))
            if limit is not None:
                count = min(count, max(0, int(limit)))
            if len(selected) != count:
                raise ValueError('Selected mask does not use the specified affordable prefix')
    else:
        raise ValueError('Unaccounted frozen format: '+method)
    if owned['visual_base'] != expected or owned['visual_residual'] != residual or owned['selector_metadata'] != lookup:
        raise ValueError('Independent actual owner formula mismatch')
    if record.get('encoder_calibration_parameters_bytes', 0) != encoder:
        raise ValueError('Encoder constant owner mismatch')
    store = actual_visual_owner(record)+encoder
    if store != expected+residual+lookup+8*nv+encoder:
        raise ValueError('Actual visual owner contains unaccounted tensors')
    bits = 16*store/(4*e)
    if method in ('direction_meta', 'sparse_v_residual', 'mixed_residual') and bits > 1.8+1e-12:
        raise ValueError('Actual complete per-image budget exceeds 1.8')
    raw_extra = sum(event.get('selection_bytes', 0)+event['residual_bytes']+
        (event['base_visual_bytes']+event['text_bytes'] if event['pass_kind'] in ('replay', 'repair', 'probe') else 0)
        for event in record['read_events'])
    if raw_extra != record['extra_logical_read_bytes']:
        raise ValueError('Independent read ledger sum differs')
    return dict(complete_visual_bits=bits, complete_visual_bytes=store,
        mask_units=sum(value.bit_count() for value in mask), mask_sha256=hashlib.sha256(bytes(mask)).hexdigest(),
        null_required_fields=False, mask_bits_valid=True)


def named_comparisons(by, pairs, family):
    rows = []
    for left, right in pairs:
        if left not in by or right not in by or by[left].keys() != by[right].keys():
            raise ValueError('Missing complete planned comparison')
        rows.append(dict(left=left, right=right, family=family, **paired(by[left], by[right], repeats=8000)))
    for row, adjusted in zip(rows, holm([row['p'] for row in rows])):
        row['holm_within_named_family'] = adjusted
    return rows


def format_controls():
    # A null owner and illegal residual bits must be rejected before scoring.
    fake = dict(session_id='synthetic', variant='full8', method='full', visual_tokens=128,
        owned_initial=None, turns=[], read_events=[], variant_config=dict(name='full8', method='full'))
    rejected = False
    try:
        representation_check(fake, fake['variant_config'], 0)
    except ValueError:
        rejected = True
    invalid_masks = [[('k', 28, 0)], [('v', 0, 4)], [('x', 0, 0)], [('k', 0, 0), ('k', 0, 0)]]
    masks_rejected = 0
    for units in invalid_masks:
        try:
            checked_axis_mask(units)
        except ValueError:
            masks_rejected += 1
    seed_rejected = 0
    for source in ('torch.manual_seed(0)', 'torch.manual_seed(None)', 'pass'):
        try:
            checked_runner_seed(source)
        except ValueError:
            seed_rejected += 1
    if checked_axis_mask([('k', 0, 0), ('v', 0, 0)])[0] != 3 or checked_runner_seed('torch.manual_seed(1212)') != 1212:
        raise RuntimeError('Valid mask/seed format rejected')
    if not rejected or masks_rejected != 4 or seed_rejected != 3 or not math.isclose(holm([.01, .04, .03])[0], .03):
        raise RuntimeError('Synthetic format/Holm controls failed')
    return dict(status='PASS', required_null_rejected=True, three_item_holm=True,
        invalid_mask_cases_rejected=masks_rejected, invalid_seed_cases_rejected=seed_rejected,
        scope='pure synthetic format only; no confirmation result inspected')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--summary', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    controls = format_controls()
    if a.self_test:
        print(json.dumps(controls)); return
    if a.summary is None or a.output is None:
        p.error('--summary and --output required')
    frozen = load(MARKER)
    asset_audit = frozen_asset_checks(frozen)
    basic = load(a.summary)
    sample_key, ref_key = 'configs/a2/round5_v1/doc_confirm64.json', 'configs/data/round5_doc64_references.jsonl'
    samples = {row['id']: row for row in load(ROOT/sample_key)}
    refs = {row['id']: {int(t['turn']): t['answer'] for t in row['references']}
            for row in map(json.loads, (ROOT/ref_key).read_text(encoding='utf-8').splitlines())}
    names = {v['name'] for v in frozen['variants']}
    if (len(samples), sum(map(len, refs.values())), len(names)) != (64, 256, 12):
        raise ValueError('Frozen Doc64/256/12 shape differs')
    if basic['status'] != 'COMPLETE' or basic['analysis_role'] != 'confirmation' or (basic['images'], basic['questions_per_variant'], basic['variants'], basic['sessions'], basic['scored_answers']) != (64, 256, 12, 768, 3072):
        raise ValueError('Incomplete or wrong-role Doc confirmation analysis')
    if basic['samples_sha256'] != frozen['input_sha256'][sample_key] or basic['references_sha256'] != frozen['input_sha256'][ref_key]:
        raise ValueError('Analyzed sample/reference hashes differ from freeze')
    base_bytes = 4*load(local_path(next(v['metadata_factor_path'] for v in frozen['variants'] if v.get('metadata_factor_path'))))['parameter_count']
    seen, records, runs, owner_checks = set(), [], [], []
    configs = {variant['name']: variant for variant in frozen['variants']}
    for run in basic['runs']:
        folder = Path(run['run'])
        folder = folder/'quality' if (folder/'quality').is_dir() else folder
        manifest, check = run_manifest_checks(folder, frozen, frozen['variants'], frozen['input_sha256'][sample_key])
        runs.append(check)
        for line in (folder/'records.jsonl').read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            key = (r['session_id'], r['variant'])
            if key in seen or key[0] not in samples or key[1] not in names or key[0] not in manifest['selected_ids']:
                raise ValueError('Duplicate/unexpected Doc record')
            if r['image_id'] != samples[key[0]]['image_id'] or {t['turn'] for t in r['turns']} != set(refs[key[0]]):
                raise ValueError('Doc record image/turn coverage differs')
            owner_checks.append(representation_check(r, configs[r['variant']], base_bytes))
            seen.add(key); records.append(r)
    if seen != {(sid, name) for sid in samples for name in names}:
        raise ValueError('Doc confirmation is not complete matched coverage')
    by, score_seen = defaultdict(dict), set()
    score_path = a.summary.parent/'scores.csv'
    predictions = {(r['session_id'], r['variant'], int(turn['turn'])): turn['answer']
                   for r in records for turn in r['turns']}
    for row in csv.DictReader(score_path.open(encoding='utf-8-sig')):
        sid, variant, turn = row['session_id'], row['variant'], int(row['turn'])
        key = (sid, variant, turn)
        if sid not in samples or variant not in names or turn not in refs[sid] or key in score_seen:
            raise ValueError('Unexpected/duplicate scored Doc answer')
        value = float(row['score'])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Null/nonfinite Doc score')
        rescored = score_docvqa(predictions[key], refs[sid][turn])['score']
        if abs(value-rescored) > 1e-12:
            raise ValueError('Scored Doc answer differs from frozen raw answer/reference')
        by[variant][(samples[sid]['source_document_id'], sid, turn)] = value
        score_seen.add(key)
    if len(score_seen) != 3072:
        raise ValueError('Missing scored Doc answers')
    primary = frozen['primary']
    planned = frozen['primary_planned_comparisons']
    if len(planned) != 3 or any(left != primary for left, _ in planned):
        raise ValueError('Primary family identity differs')
    primary_rows = named_comparisons(by, planned, 'frozen_primary_three')
    mechanisms = named_comparisons(by, frozen['mechanism_planned_comparisons'], 'frozen_secondary_mechanisms')
    table = {row['variant']: row for row in basic['table']}
    if set(table) != names:
        raise ValueError('Doc table variant set differs')
    for name, values in by.items():
        if abs(sum(values.values())/len(values)-table[name]['anls']) > 1e-12:
            raise ValueError('Doc table mean differs from complete scored answers')
    comparison = next(row for row in primary_rows if row['right'] == frozen['quality_reference'])
    eligible = all(check['complete_visual_bits'] <= frozen['budget_bits']+1e-12
                   for r, check in zip(records, owner_checks) if r['variant'] == primary)
    target = dict(candidate=primary, reference=frozen['quality_reference'],
        budget_bits=table[primary]['budget_bits_including_encoder_constants'],
        complete_budget_every_image=eligible, margin=frozen['quality_margin'],
        **{k: comparison[k] for k in ('delta', 'ci_low', 'ci_high', 'p', 'images', 'questions')})
    target.update(point_target_met=eligible and target['delta'] >= -target['margin'],
        noninferiority_95_ci=eligible and target['ci_low'] >= -target['margin'],
        interval_scope='lower endpoint of two-sided document-paired 95% bootstrap CI')
    a.output.mkdir(parents=True, exist_ok=False)
    write_csv(a.output/'primary_comparisons.csv', primary_rows)
    write_csv(a.output/'mechanism_comparisons.csv', mechanisms)
    write_csv(a.output/'storage_target.csv', [target])
    summary = dict(status='COMPLETE', analysis_role='confirmation',
        primary_candidate=primary, secondary_low_storage=frozen['secondary_low_storage'],
        candidate_rule='identities fixed by marker before new outcomes; no reselection',
        table=basic['table'], primary_comparisons=primary_rows, mechanism_comparisons=mechanisms,
        storage_target=target, primary_holm_family_size=3, mechanism_holm_family_size=len(mechanisms),
        images=64, questions_per_variant=256, variants=12, sessions=768, scored_answers=3072,
        audit=dict(status='PASS', assets=asset_audit, runs=runs, representation_cases=len(owner_checks),
            mask_bits_valid=True, required_nulls_absent=True, seed=1212),
        controls=controls, input_summary_sha256=sha(a.summary), scores_sha256=sha(score_path),
        analyzer_sha256=sha(__file__), scope='fresh registered source-document Doc64 confirmation on one model; full visual owner includes lookup, residual alpha and encoder constants')
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (a.output/'complete.json').write_text(json.dumps(dict(status='COMPLETE', source_marker_sha256=sha(MARKER))))
    write_output_hashes(a.output)
    print(json.dumps(dict(status='COMPLETE', primary=primary, target=target, primary_comparisons=primary_rows)))


if __name__ == '__main__':
    main()
