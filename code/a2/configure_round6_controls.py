"""Freeze same-format, every-prefix axis-matched round-six controls locally.

No SSH, GPU, model outputs, or reference annotations are read. The two controls
preserve the incumbent's K/V sequence and replace only the order within an axis.
The existing MixedResidualBank performs the actual runtime owner verification.
"""
import argparse
import copy
import hashlib
import json
import math
import random
import struct
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / 'configs/a2/round6_v1'
AUDIT = ROOT / 'audit/a2/round6_controls_v1.json'
NL, NH, DIM = 28, 4, 128
CAPACITY = 2 * NL * NH
NAMES = ('r6_incumbent_mixed_gradient', 'r6_axis_matched_spread',
         'r6_axis_matched_random1212')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def local_project_path(remote):
    return ROOT / remote.split('/TrajKV/', 1)[1]


def ranking_bytes(order):
    ids = [(0 if axis == 'k' else 1) * NL * NH + layer * NH + head
           for axis, layer, head in order]
    return struct.pack('<' + 'H' * CAPACITY, *ids)


def replace_within_axes(incumbent, order):
    positions = {'k': 0, 'v': 0}
    result = []
    for axis, _, _ in incumbent:
        layer, head = order[positions[axis]]
        positions[axis] += 1
        result.append([axis, layer, head])
    return result


def shape_owner(order, prefix, nv, factor_bytes, book_bytes):
    """Exact owned tensor shapes in mixed_residual_bank.py, not GPU measurement."""
    selected = order[:prefix]
    layer_counts = [sum(layer == index for _, layer, _ in selected) for index in range(NL)]
    # K: uint8[N,32], F16[N,1], F16[1,128]; V: packed1
    # int32[128,N/32], two F16[128,1] whole-image metadata arrays.
    visual_base = NL * NH * (nv * (32 + 2 + 16) + DIM * 2 + DIM * 4) + book_bytes
    residual = sum(count * nv * ((DIM // 32) * 4 + 2) for count in layer_counts)
    lookup = NL * 2 * NH * 4
    positions = 8 * nv
    ranking = len(ranking_bytes(order))
    encoder = factor_bytes + ranking
    total = visual_base + residual + lookup + positions + encoder
    formula = (NL * NH * nv * 50 + NL * NH * DIM * 2
               + 4 * NL * NH * DIM + book_bytes + lookup + 8 * nv
               + factor_bytes + ranking + prefix * nv * (DIM // 8 + 2))
    if total != formula:
        raise ValueError('Tensor shape owner disagrees with frozen bank formula')
    return dict(visual_base=visual_base, visual_residual=residual,
                dual_lookup=lookup, visual_position=positions,
                encoder_constants=encoder, total_including_encoder=total,
                visual_fp16_denominator=2 * NL * NH * nv * DIM * 2)


def build():
    source_path = ROOT / 'configs/a2/round5_v1/mixed_dev64.json'
    freeze_path = ROOT / 'audit/a2/round5_confirmation_v1.json'
    source, freeze = read(source_path), read(freeze_path)
    incumbent = next(v for v in source['variants'] if v['name'] == freeze['primary'])
    if freeze['primary'] != 'mixed_gradient_max' or incumbent['method'] != 'mixed_residual':
        raise ValueError('Unexpected frozen incumbent')
    frozen = next(v for v in freeze['variants'] if v['name'] == freeze['primary'])
    if frozen != incumbent:
        raise ValueError('Development incumbent differs from round5 freeze')
    order = incumbent['selected_units']
    universe = {(axis, layer, head) for axis in ('k', 'v')
                for layer in range(NL) for head in range(NH)}
    if len(order) != CAPACITY or set(map(tuple, order)) != universe:
        raise ValueError('Complete unique 224-entry mixed ordering required')
    # Reuse exactly the round5 bit-reversal spread and Random(1212) order;
    # both axes get the same predetermined within-axis order.
    spread = []
    for i in range(128):
        index = int(f'{i:07b}'[::-1], 2)
        if index < NL * NH:
            spread.append((index // NH, index % NH))
    randomized = [(i // NH, i % NH) for i in range(NL * NH)]
    random.Random(1212).shuffle(randomized)
    orders = [order, replace_within_axes(order, spread),
              replace_within_axes(order, randomized)]
    variants = []
    for name, candidate in zip(NAMES, orders):
        value = copy.deepcopy(incumbent)
        value.update(name=name, selected_units=candidate)
        variants.append(value)

    external = {}
    for key in ('metadata_factor_path', 'direction_book_path'):
        remote = incumbent[key]
        local = local_project_path(remote)
        digest = sha(local)
        if freeze['external_codec_inputs'][remote] != digest:
            raise ValueError('Frozen external codec input changed: ' + key)
        external[local.relative_to(ROOT).as_posix()] = digest
    factors = read(local_project_path(incumbent['metadata_factor_path']))
    factor_bytes = 4 * factors['parameter_count']
    counted = sum(len(row) for key in ('key_head_factors', 'value_head_factors', 'value_center_offsets')
                  for row in factors.get(key, []))
    if counted != factors['parameter_count'] or factors['value_group_size'] != 'all':
        raise ValueError('Unexpected factor family/window')
    book = read(local_project_path(incumbent['direction_book_path']))['centers']
    if len(book) != 256 or any(len(row) != 4 for row in book):
        raise ValueError('Unexpected frozen shared decoder book')
    book_bytes = 256 * 4 * 2
    for candidate in orders:
        if len(candidate) != CAPACITY or set(map(tuple, candidate)) != universe:
            raise ValueError('Control has duplicated/missing units')
        if [u[0] for u in candidate] != [u[0] for u in order]:
            raise ValueError('Control changes the K/V sequence')
    prefixes = []
    for prefix in range(CAPACITY + 1):
        counts = {axis: sum(u[0] == axis for u in order[:prefix]) for axis in ('k', 'v')}
        for candidate in orders[1:]:
            if any(sum(u[0] == axis for u in candidate[:prefix]) != counts[axis] for axis in counts):
                raise ValueError('Prefix axis count mismatch')
        prefixes.append(dict(prefix=prefix, K_units=counts['k'], V_units=counts['v']))
    cost_checks = []
    for nv in range(128, 4097, 128):
        for prefix in range(CAPACITY + 1):
            costs = [shape_owner(candidate, prefix, nv, factor_bytes, book_bytes) for candidate in orders]
            if any(value != costs[0] for value in costs[1:]):
                raise ValueError('Same-prefix actual tensor shape costs differ')
        fixed = shape_owner(order, 0, nv, factor_bytes, book_bytes)
        allowed = math.floor(incumbent['budget_bits'] * fixed['visual_fp16_denominator'] / 16)
        affordable = min(CAPACITY, max(0, (allowed - fixed['total_including_encoder']) // (18 * nv)))
        actual = shape_owner(order, affordable, nv, factor_bytes, book_bytes)
        cost_checks.append(dict(visual_tokens=nv, base_fits_budget=fixed['total_including_encoder'] <= allowed,
            affordable_units=affordable, **prefixes[affordable], **actual,
            actual_bits_including_encoder=16 * actual['total_including_encoder'] / actual['visual_fp16_denominator']))

    remote_root = incumbent['metadata_factor_path'].split('/results/', 1)[0]
    outputs = {}
    for label, samples, count, turns in [
        ('doc16', 'configs/a2/docvqa_next_dev64.json', 16, 6),
        ('doc64', 'configs/a2/docvqa_next_dev64.json', 64, 6),
        ('text64', 'configs/a2/round5_v1/text_transfer64.json', 64, 1),
        ('seen_doc64', 'configs/a2/round5_v1/doc_confirm64.json', 64, 6),
    ]:
        rows = read(ROOT / samples)
        if len(rows) < count or any(any(k in row for k in ('answer', 'answers', 'references')) for row in rows):
            raise ValueError('Incomplete or answer-bearing runner source')
        outputs[FOLDER / f'controls_{label}.json'] = dict(source,
            arguments=dict(source['arguments'], samples=remote_root + '/' + samples, turns=turns),
            variants=copy.deepcopy(variants), start=0, count=count)
    input_paths = [source_path, freeze_path, ROOT / 'code/a2/mixed_residual_bank.py',
                   ROOT / 'code/a2/configure_sparse_round5.py', ROOT / 'code/a2/round5_runner.py',
                   ROOT / 'configs/a2/docvqa_next_dev64.json',
                   ROOT / 'configs/a2/round5_v1/text_transfer64.json',
                   ROOT / 'configs/a2/round5_v1/doc_confirm64.json']
    report = dict(schema='trajkv.round6.axis-matched-controls.v1',
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        incumbent=freeze['primary'], variants=list(NAMES), capacity=CAPACITY,
        ranking_constant_bytes=CAPACITY * 2, dual_lookup_bytes=NL * 2 * NH * 4,
        base_factor_bytes=factor_bytes, decoder_book_bytes=book_bytes,
        random_seed=1212, axis_order_reused_for_both_axes=True,
        spread_rule='same round5 7-bit reversal of flat layer*4+head indices; discard >=112',
        random_rule='same round5 Python Random(1212).shuffle of flat layer-major 112 units; identical within-axis sequence for K and V',
        every_prefix_axis_counts_verified=True, all_units_unique_and_complete=True,
        every_prefix_shape_owner_costs_equal=True, prefix_checks=len(prefixes),
        tested_visual_token_counts=list(range(128, 4097, 128)),
        shape_owner_checks=(CAPACITY + 1) * 32 * len(orders),
        owner_verification_scope='exact allocated tensor shape arithmetic from frozen MixedResidualBank; runtime bank additionally verifies owned bytes; this audit is not a GPU measurement',
        arbitrary_N_cost_match='symbolic: all orders have identical fixed owners and 18*N bytes per selected unit; preserved axis sequence therefore matches every affordable prefix for any legal N',
        model_outcomes_read=False, reference_answers_read=False, SSH_used=False, GPU_used=False,
        protocol='known development/seen-cohort same-format ablation; no new blind-confirmation claim',
        prefixes=prefixes, budget_costs=cost_checks,
        ranking_sha256={name: hashlib.sha256(ranking_bytes(candidate)).hexdigest()
                        for name, candidate in zip(NAMES, orders)},
        input_sha256={path.relative_to(ROOT).as_posix(): sha(path) for path in input_paths} | external,
        source_sha256=sha(__file__))
    return outputs, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    outputs, report = build()
    encoded = {path: json.dumps(value, indent=2).encode('utf-8') for path, value in outputs.items()}
    report['configuration_sha256'] = {path.relative_to(ROOT).as_posix(): hashlib.sha256(payload).hexdigest()
                                      for path, payload in encoded.items()}
    if not args.dry_run:
        if AUDIT.exists():
            old = read(AUDIT)
            for key in ('created_at_utc',):
                report[key] = old[key]
        encoded[AUDIT] = json.dumps(report, indent=2).encode('utf-8')
        for path, payload in encoded.items():
            if path.exists() and path.read_bytes() != payload:
                raise FileExistsError('Refusing to overwrite different frozen file: ' + str(path))
        for path, payload in encoded.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                with path.open('xb') as stream:
                    stream.write(payload)
    print(json.dumps(dict(status='DRY_RUN' if args.dry_run else 'FROZEN', variants=list(NAMES),
        config_files=[str(path.relative_to(ROOT)) for path in outputs],
        prefix_checks=report['prefix_checks'], shape_owner_checks=report['shape_owner_checks'],
        ranking_constant_bytes=report['ranking_constant_bytes'], dual_lookup_bytes=report['dual_lookup_bytes'],
        references_read=False, model_outcomes_read=False, uploaded=False)))


if __name__ == '__main__':
    main()
