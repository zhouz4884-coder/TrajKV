"""Independent complete-owner arithmetic for R8 formats; not reader/GPU proof."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def cost(nv, layers, heads, kd, kc, vd=None, vc=None, *, kpos=1, decoder_only=False,
         stages=43, factors=None, ranking_entries=None):
    dim, lh = 128, layers * heads
    assert dim % kd == 0 and (vd is None or dim % vd == 0)
    kbits = int(math.log2(kc)); assert 2 ** kbits == kc
    k_per_token_bits = (dim // kd) * kbits
    assert k_per_token_bits % 8 == 0, 'This contract requires true packing, not byte-per-symbol padding'
    kpayload = lh * nv * k_per_token_bits // 8
    if vd is None:
        vpayload = lh * dim * ((nv + 31) // 32) * 4
        vbook = 0
    else:
        vbits = int(math.log2(vc)); assert 2 ** vbits == vc
        bits_per_token = dim // vd * vbits
        assert bits_per_token % 8 == 0
        vpayload = lh * nv * bits_per_token // 8
        vbook = vc * vd * 2
    fields = dict(K_codes=kpayload, K_row_gain=lh * nv * 2, K_mean=lh * dim * 2,
        V_codes=vpayload, V_channel_scale_min=lh * dim * 4,
        K_global_or_position_books=kpos * kc * kd * 2,
        K_retained_old_encoder_book=128 if decoder_only else 0,
        V_book=vbook, shared_visual_positions=nv * 8,
        selected_sign_alpha_pool=stages * nv * 18,
        full_stage_lookup=layers * 2 * heads * 2 * 4,
        complete_uint16_ranking=2 * (2 * layers * heads if ranking_entries is None else ranking_entries), policy_descriptor=5,
        encoder_factors=3 * lh * 4 if factors is None else factors)
    total = sum(fields.values()); denominator = lh * nv * dim * 4
    return dict(visual_tokens=nv, layers=layers, kv_heads=heads, head_dim=dim,
        K_format=dict(dim=kd, words=kc, index_bytes_per_token=k_per_token_bits // 8, position_books=kpos),
        V_format=dict(dim=vd, words=vc, index_layout='channel-tail-bitwords' if vd is None else 'token-block-packed-indices'),
        actual_owner_components=fields, complete_visual_owner_bytes=total,
        visual_FP16_denominator_bytes=denominator, complete_bits_per_KV_scalar=16 * total / denominator,
        no_semantic_visual_padding=True, no_V_row_gain=True,
        binary_tail_storage_bytes=lh * 16 * (((nv + 31) // 32) * 32 - nv) if vd is None else 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT / 'audit/a2/round8_independent_budget_v1.json')
    p.add_argument('--nv', nargs='+', type=int, default=[768, 1024, 1280, 1792])
    p.add_argument('--layers', type=int, default=28)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--ranking-entries', type=int, default=224,
                   help='224 for old full-unit encoder ranking, 43 for the new explicit exact stage list')
    a = p.parse_args(); assert not a.output.exists()
    specs = [('00_K4_16_Vbinary', 4, 16, None, None, 1, False),
             ('10_K8_256_Vbinary', 8, 256, None, None, 1, False),
             ('01_K4_16_V8_256', 4, 16, 8, 256, 1, False),
             ('11_K8_256_V8_256', 8, 256, 8, 256, 1, False),
             ('K4_global_fitted_same_geometry', 4, 16, None, None, 1, False),
             ('K4_position_matched4096B', 4, 16, None, None, 32, False),
             ('K8_decoder_only_retains128B_encoder', 8, 256, None, None, 1, True),
             ('joint_gain_same_fields', 4, 16, None, None, 1, False),
             ('K8_V8_64_subbit', 8, 256, 8, 64, 1, False),
             ('K8_V8_16_halfbit', 8, 256, 8, 16, 1, False),
             ('K8_V16_256_halfbit', 8, 256, 16, 256, 1, False)]
    rows = []
    for nv in a.nv:
        for name, kd, kc, vd, vc, positions, decoder in specs:
            row = cost(nv, a.layers, a.heads, kd, kc, vd, vc, kpos=positions, decoder_only=decoder, ranking_entries=a.ranking_entries)
            row['variant_contract'] = name; rows.append(row)
    if (a.layers, a.heads) == (28, 4) and 1792 in a.nv:
        baseline = next(row for row in rows if row['visual_tokens'] == 1792 and row['variant_contract'] == '00_K4_16_Vbinary')
        expected = 8315013 + 2 * (a.ranking_entries - 224)
        assert baseline['complete_visual_owner_bytes'] == expected
        four = [row for row in rows if row['visual_tokens'] == 1792][:4]
        assert [row['complete_visual_owner_bytes'] - expected for row in four] == [0, 3968, 4096, 8064]
    result = dict(schema='trajkv.round8.independent-budget-contract.v1', status='PASS', rows=rows,
        baseline='r7_joint_k1_gt_cond: fixed 43 exact stages (23K,20V); actual stage IDs must be bound by root config',
        missing_3589B_explained=dict(lookup=1792, ranking=448, policy=5, encoder_factors=1344),
        accounting_scope='One cache owner; global books once; no total_unique duplication or cross-cache amortization',
        correctness_scope='Independent integer accounting only; actual tensor inventory and new reader identity/GPU validation remain separate',
        ranking_entries=a.ranking_entries,
        GPU_used=False, model_scores_read=False, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status='PASS', contracts=len(rows), baseline_N1792_bytes=8315013 + 2 * (a.ranking_entries - 224),
        main_four_book_deltas=[0,3968,4096,8064], GPU_used=False)))


if __name__ == '__main__':
    main()
