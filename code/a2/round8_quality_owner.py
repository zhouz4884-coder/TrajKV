"""R8 actual-owner and artifact audit; imports no model/GPU implementation."""
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np
import analyze_round7_quality as r7

ROOT = Path(__file__).resolve().parents[2]
PATH_MAP = {}


def configure_path_map(path=None):
    global PATH_MAP
    PATH_MAP = json.loads(Path(path).read_text(encoding='utf-8-sig')) if path else {}
    # The inherited non-R8 audit shares this resolver without modifying its
    # frozen source or any execution manifest.
    r7.local_path = local_path


def local_path(value):
    value = str(value).replace('\\', '/')
    exact = PATH_MAP.get('files', {})
    if value in exact:
        path = Path(exact[value]); return path if path.is_absolute() else ROOT / path
    for item in sorted(PATH_MAP.get('prefixes', []), key=lambda row: -len(row['remote'])):
        prefix = item['remote'].rstrip('/')
        if value == prefix or value.startswith(prefix + '/'):
            path = Path(item['local']); path = path if path.is_absolute() else ROOT / path
            return path / value[len(prefix):].lstrip('/')
    if '/TrajKV/' in value:
        return ROOT / value.split('/TrajKV/', 1)[1]
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_sample_source(manifest, samples_path, selected_ids):
    if manifest['samples_sha256'] == r7.sha(samples_path):
        return 'same_input_manifest'
    source = local_path(manifest['config']['arguments']['samples'])
    r7.require(source.is_file() and r7.sha(source) == manifest['samples_sha256'], 'Cannot verify original staged sample bytes; supply --artifact-map')
    original = {row['id']: row for row in r7.read(source)}
    subset = {row['id']: row for row in r7.read(samples_path)}
    r7.require(all(sid in original and sid in subset and original[sid] == subset[sid] for sid in selected_ids), 'Analysis subset differs from original stimuli')
    return 'exact_subset_of_original_stimuli'


def audit_owner(record, variant, external, cache):
    if record['method'] != 'round8_joint':
        return r7.audit_owner(record, variant, external, cache)
    req = r7.require
    req(record['variant_config'] == variant and record['method'] == variant['method'] and record['variant'] == variant['name'], 'R8 producer settings mismatch')
    owner = record['owned_initial']
    req(owner.get('full_visual_retained') is False and record.get('visual_fp16_materialization') is False
        and record.get('key_source_fp16_retained') is False and record.get('depth_residual_source_fp16_retained') is False
        and not owner.get('diagnostic_fp16_reference'), 'R8 deployment retains FP16 source or materializes visual KV')
    req(record.get('question_state_scope') == 'recomputed_under_codec' and record.get('initial_logits_shared_fp16') is False
        and record.get('first_question_suffix_replay_tokens', 0) > 0, 'R8 question state not recomputed under codec')
    def artifact(key):
        value = variant[key]; path = local_path(value)
        req(path.is_file() and external.get(value) == r7.sha(path), 'Missing/changed consumed artifact: ' + value)
        if value not in cache:
            cache[value] = r7.read(path)
        return cache[value]
    factor = artifact('metadata_factor_path')
    arrays = [factor.get(key) for key in ('key_head_factors', 'value_head_factors', 'value_center_offsets')]
    arrays = [array for array in arrays if array is not None]
    layers, heads = len(arrays[0]), len(arrays[0][0]); lh = layers * heads
    req(all(len(array) == layers and all(len(row) == heads for row in array) for array in arrays), 'R8 factor dimensions inconsistent')
    factor_bytes = sum(len(row) for array in arrays for row in array) * 4
    req(factor['parameter_count'] * 4 == factor_bytes, 'R8 factor physical FP32 charge mismatch')
    reference = artifact('metadata_reference_book_path')
    reference_sha = external[variant['metadata_reference_book_path']]
    req(factor['encoder_book_sha256'] == reference_sha and record['metadata_reference_book_sha256'] == reference_sha
        and record['metadata_calibration_sha256'] == external[variant['metadata_factor_path']], 'R8 numerical-inheritance reference provenance mismatch')
    book = np.asarray(artifact('direction_book_path')['centers'], dtype=np.float16)
    shape = tuple(book.shape)
    req(shape in ((16,4),(256,4),(256,8),(32,16,4)) and np.isfinite(book).all(), 'Unsupported/nonfinite actual R8 K book')
    positions = len(shape) == 3; words, dim = shape[-2:]
    kbytes = 128 // dim * (int(math.log2(words))) // 8
    req(record['key_book_shape'] == list(shape) and record['key_book_words'] == words
        and record['direction_book_dimension'] == dim and record['key_position_specific'] == positions
        and record['key_index_bytes_per_token'] == kbytes and record['decoder_book_bytes'] == book.nbytes
        and record['key_book_sha256'] == external[variant['direction_book_path']], 'R8 K shape/payload/book fingerprint mismatch')
    encoding = variant.get('key_encoding', 'reencode'); extra = 0
    req(encoding in ('reencode','decoder-only'),'Unknown R8 K encoding mode')
    req(record['key_encoding'] == encoding, 'R8 key encoding mode differs from config')
    if encoding == 'decoder-only':
        encoder = np.asarray(artifact('encoder_book_path')['centers'], dtype=np.float16)
        req(encoder.shape == (16,4) and external[variant['encoder_book_path']] == reference_sha, 'R8 decoder-only original encoder identity mismatch')
        extra = encoder.nbytes
        req(record['key_encoder_book_sha256'] == reference_sha, 'R8 retained encoder SHA mismatch')
    req(record['key_encoder_extra_book_bytes'] == extra, 'R8 retained encoder must be charged once')
    value_mode = variant.get('value_encoding', 'binary'); vbook_bytes = 0
    req(value_mode in ('binary','cube-identity','decoder-only','reencode'),'Unknown R8 V encoding mode')
    req(record['value_encoding'] == value_mode and record['value_frame'] == 'all'
        and record['value_row_gain_added'] is False, 'R8 V frame/mode/gain changed without declaration')
    if value_mode != 'binary':
        vbook = np.asarray(artifact('value_joint_book_path')['centers'], dtype=np.float16)
        req(vbook.shape == (256,8) and np.isfinite(vbook).all(), 'R8 V book geometry mismatch')
        vbook_bytes = vbook.nbytes
        req(record['value_joint_book_sha256'] == external[variant['value_joint_book_path']], 'R8 V book fingerprint mismatch')
        if value_mode == 'cube-identity':
            expected = np.asarray([[(j >> bit) & 1 for bit in range(8)] for j in range(256)], dtype=np.float16)
            req(np.array_equal(vbook, expected), 'R8 cube identity codeword order changed')
    req(record['value_joint_book_bytes'] == vbook_bytes and record['value_index_bytes_per_token'] == 16, 'R8 V dictionary/row-index bytes mismatch')
    nv = int(record['visual_tokens']); full = lh * nv * 128 * 4
    req(nv > 0 and nv % 32 == 0 and owner['visual_fp16_denominator'] == full, 'R8 current primary reader requires complete real-N bitwords/D128')
    base = lh * nv * (kbytes + 2 + 16) + lh * 128 * 2 + lh * 128 * 4 + book.nbytes + vbook_bytes
    stages = [tuple(item) for item in record['depth_residual_selected_stages']]
    seen = set()
    for axis, layer, head, stage in stages:
        req(axis in ('k','v') and 0 <= layer < layers and 0 <= head < heads and stage in (0,1)
            and (axis,layer,head,stage) not in seen and (stage == 0 or (axis,layer,head,0) in seen), 'R8 illegal stage prefix')
        seen.add((axis,layer,head,stage))
    if variant.get('selected_stages') is not None:
        entries = [tuple(item) for item in variant['selected_stages']]
        requested_seen = set()
        for axis,layer,head,stage in entries:
            req(axis in ('k','v') and 0 <= layer < layers and 0 <= head < heads and stage in (0,1)
                and (axis,layer,head,stage) not in requested_seen
                and (stage == 0 or (axis,layer,head,0) in requested_seen), 'R8 illegal requested stage ranking')
            requested_seen.add((axis,layer,head,stage))
        stage_ids = True; descriptor = struct.pack('<Bf',3,0.)
    else:
        entries = [tuple(item) for item in variant.get('selected_units', [])]; stage_ids = False
        policy = variant.get('depth_policy','incumbent'); fraction = {'incumbent':0.,'v_pair':1.,'v_mixed':float(variant.get('v_mixed_pair_fraction',.5))}[policy]
        descriptor = struct.pack('<Bf',{'incumbent':0,'v_pair':1,'v_mixed':2}[policy],fraction)
    ids = [((int(axis == 'v') * layers + layer) * heads + head) * (2 if stage_ids else 1)
           + (stage if stage_ids else 0) for axis,layer,head,*tail in entries for stage in ([tail[0]] if stage_ids else [0])]
    ranking = struct.pack('<'+'H'*len(ids),*ids)
    lookup = lh * 16; residual = len(stages) * nv * 18
    constants = factor_bytes + extra + len(ranking) + len(descriptor)
    allowed = math.floor(variant.get('budget_bits',1.8)*full/16)
    fixed = base + lookup + nv*8 + constants
    if stage_ids:
        affordable = max(0,(allowed-fixed)//(nv*18))
        limit = variant.get('res_stages_limit',variant.get('res_units_limit',variant.get('residual_units_limit')))
        if limit is not None:
            affordable = min(affordable,max(0,int(limit)))
        req(stages == entries[:affordable], 'R8 actual stage mask is not the legal affordable requested prefix')
    else:
        req(variant.get('residual_layout_reference') in (None,'actual'), 'R8 named policy must use its actual layout')
        plan = r7.independent_plan(variant,layers,heads,nv,base+nv*8,factor_bytes+extra,0,words,full)
        req(stages == plan['stages'], 'R8 named-policy stage mask differs from independent scalar planner')
    for name,value in (('depth_residual_requested_stages',entries),('depth_residual_fixed_complete_bytes',fixed),
                       ('depth_residual_allowed_bytes',allowed)):
        if name in record:
            actual = ([tuple(item) for item in record[name]] if record[name] is not None else None) if name.endswith('requested_stages') else record[name]
            expected = value if not name.endswith('requested_stages') or stage_ids else None
            req(actual == expected,'R8 planner diagnostic differs from independent recount: '+name)
    req(record.get('key_metric_bytes',0) == 0, 'R8 first prototype has no undeclared rank-one metric')
    req(owner['visual_base'] == base and owner['visual_residual'] == residual and owner['selector_metadata'] == lookup
        and owner['visual_position'] == 8*nv, 'R8 actual codes/metadata/books/residual/lookup/position owner mismatch')
    req(record['encoder_calibration_parameters_bytes'] == constants and owner.get('encoder_constant_bytes') == constants,
        'R8 factors/ranking/policy/retained encoder omitted or doubled')
    total = base + residual + lookup + nv*8 + constants
    req(owner['complete_visual_owner_bytes'] == total and record['depth_residual_complete_bytes'] == total,
        'R8 declared complete owner differs from shape recount')
    req(record['depth_residual_encoder_ranking_bytes'] == len(ranking)
        and record['depth_residual_encoder_policy_bytes'] == len(descriptor)
        and record['depth_residual_encoder_ranking_sha256'] == hashlib.sha256(ranking).hexdigest()
        and record['depth_residual_encoder_policy_sha256'] == hashlib.sha256(descriptor).hexdigest(), 'R8 actual compact encoder constant mismatch')
    bits = 16*total/full
    req(r7.same(bits,record['depth_residual_actual_budget_bits']) and total <= allowed, 'R8 complete budget violated')
    return dict(session_id=record['session_id'],variant=record['variant'],method=record['method'],role='actual packed runtime',
        visual_tokens=nv,visual_tokens_mod32=nv%32,visual_fp16_denominator=full,
        visual_base_bytes=base,residual_bytes=residual,lookup_bytes=lookup,visual_position_bytes=nv*8,
        encoder_parameters_bytes=constants,complete_visual_owner_bytes=total,actual_visual_bits=bits,
        auxiliary_visual_owner_bytes=0,total_unique_recorded_for_diagnostics_only=owner.get('total_unique'),
        key_book_words=words,key_index_bytes_per_token=kbytes,key_metric_bytes=0,
        decoder_K_book_bytes=book.nbytes,decoder_V_book_bytes=vbook_bytes,retained_K_encoder_bytes=extra,
        physical_tail_storage_bytes=0,actual_packed_owner_reconstructed=True,
        residual_mask_sha256=hashlib.sha256(json.dumps(stages,separators=(',',':')).encode()).hexdigest(),
        k2_reference_mask_verified=None,owner_audit='PASS',owner_scope='Actual K/V dictionary/index layouts and all persistent encoder constants; no total_unique duplication')
