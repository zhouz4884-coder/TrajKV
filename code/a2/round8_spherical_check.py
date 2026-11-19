"""CPU math/dispatch controls and optional CUDA encoder arithmetic; no model/fit."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from round8_codec import (decode_key, encode_key as legacy_encode, key_amplitude,
                         key_template, pack_nibbles, unpack_nibbles)
from round8_spherical_codec import encode_key
from round8_spherical_runner import scoped_dispatch, spherical_class, validate_variant


def row_gain(x, t, factor):
    return (key_amplitude(x, t, 'norm').half().float() * factor).half()


def dense_reference(x, book, factor, sweeps, hard):
    """Independent one-row oracle enumerates all complete128-D templates."""
    code, gain = legacy_encode(x, book, gain_factor=factor)
    ids = unpack_nibbles(code).clone()
    t = key_template(code, book)
    xd = x.double()
    def cosine(value):
        td = value.double()
        return float((xd * td).sum() / td.square().sum().sqrt())
    def error(value, scale):
        return float((xd - (value * scale.float()).half().double()).square().sum())
    for _ in range(sweeps):
        for slot in range(32):
            sl = slice(4 * slot, 4 * slot + 4)
            old_id = int(ids[0, slot])
            current_score, current_error = cosine(t), error(t, gain)
            best_id, best_score, best = old_id, current_score, t
            for index in range(16):
                if index == old_id:
                    continue
                candidate = t.clone(); candidate[:, sl] = book[index].float()
                score = cosine(candidate)
                if score > best_score:
                    best_id, best_score, best = index, score, candidate
            if best_id == old_id:
                continue
            proposal_gain = row_gain(x, best, factor)
            if not hard or error(best, proposal_gain) < current_error:
                ids[0, slot] = best_id; t, gain = best, proposal_gain
    return pack_nibbles(ids), gain


def wht(x):
    y = x.float()
    shape = y.shape
    for width in (1, 2, 4, 8, 16, 32, 64):
        view = y.reshape(*shape[:-1], 128 // (2 * width), 2, width)
        a, b = view[..., 0, :], view[..., 1, :]
        y = torch.cat((a + b, a - b), -1).reshape(shape)
    return y * 128 ** -.5


def controls(output):
    torch.set_num_threads(1); torch.manual_seed(20261118)
    x = torch.tensor([2., 1.]).repeat(64)[None]
    def witness(pair):
        book = torch.full((16, 4), 20., dtype=torch.float16)
        book[0] = 1.; book[1] = torch.tensor(pair).repeat(2).half()
        return book
    accepts = witness((1.0146484375, 1.013671875))
    rejects = witness((1.009765625, 1.0087890625))
    cases = []
    for name, book in [('accepts_nonempty', accepts), ('cosine_up_hard_down_reject', rejects)]:
        old_code, old_gain = legacy_encode(x, book)
        old_error = float((x.double() - decode_key(old_code, old_gain, book).double()).square().sum())
        code, gain, stats = encode_key(x, book, sweeps=1, return_stats=True)
        dc, dg = dense_reference(x, book, 1., 1, True)
        if not torch.equal(code, dc) or not torch.equal(gain, dg):
            raise RuntimeError('Incremental coordinate result differs from dense full-template candidate control')
        if name == 'accepts_nonempty' and not (stats['accepted_slot_updates'] > 0 and stats['source_base_SSE_after'] < old_error):
            raise RuntimeError('Nonempty accepted-improvement witness failed')
        if name != 'accepts_nonempty' and not (stats['accepted_slot_updates'] == 0 and stats['cosine_best_proposals_rejected_by_hard_SSE'] > 0
                and torch.equal(code, old_code) and torch.equal(gain, old_gain)):
            raise RuntimeError('FP16 harm was not rejected with original bits preserved')
        cases.append(dict(name=name, old_base_SSE=old_error, **stats))
    for sweeps in (1, 2):
        for _ in range(3):
            source = torch.randn(1, 128); book = torch.randn(16, 4).half(); factor = torch.tensor([[1.073]])
            code, gain, stats = encode_key(source, book, gain_factor=factor, sweeps=sweeps, return_stats=True)
            dc, dg = dense_reference(source, book, factor, sweeps, True)
            if not torch.equal(code, dc) or not torch.equal(gain, dg) or stats['max_valid_row_base_SSE_increase'] > 0:
                raise RuntimeError('Random dense candidate/sweep or base-loss guard control failed')
    # Strict ties, duplicates, zero source and already-optimal rows are no-ops.
    book = torch.ones((16, 4), dtype=torch.float16)
    for source in (torch.zeros(2, 128), torch.ones(2, 128) * 2, torch.randn(2, 128)):
        oc, og = legacy_encode(source, book)
        c, g = encode_key(source, book, sweeps=2)
        if not torch.equal(c, oc) or not torch.equal(g, og):
            raise RuntimeError('Duplicate/tie/zero no-op changed original fields')
    # A deliberate coordinate trap prevents treating no improvement as a bound.
    trap = witness((2., 1.)); oc, og = legacy_encode(x, trap)
    c, g = encode_key(x, trap, sweeps=2)
    global_code = pack_nibbles(torch.ones(1, 32, dtype=torch.uint8))
    perfect = decode_key(global_code, torch.ones(1, 1).half(), trap)
    if not torch.equal(c, oc) or not torch.equal(g, og) or not torch.equal(perfect.float(), x):
        raise RuntimeError('Local-trap scope control failed')
    # Broadcast factors, real WHT coordinates and row chunking.
    native = torch.randn(1, 2, 3, 128).half()
    mean = native.float().mean(-2, keepdim=True).half()
    source = wht(native.float() - mean.float())
    book = torch.randn(16, 4).half(); factor = torch.tensor([.94, 1.07])[None, :, None, None]
    c, g, st = encode_key(source, book, gain_factor=factor, sweeps=2, row_chunk=2, return_stats=True)
    cf, gf = encode_key(source.reshape(-1, 128), book,
        gain_factor=factor.expand(1, 2, 3, 1).reshape(-1, 1), sweeps=2, row_chunk=2)
    if not torch.equal(c.reshape(-1, 16), cf) or not torch.equal(g.reshape(-1, 1), gf) or c.numel() != 6 * 16:
        raise RuntimeError('Broadcast/WHT shape or physical nibble owner mismatch')
    rotated = decode_key(c, g, book).float()
    baseline_c, baseline_g = legacy_encode(source, book, gain_factor=factor)
    before_rows = (source.double() - decode_key(baseline_c, baseline_g, book).double()).square().sum(-1)
    after_rows = (source.double() - rotated.double()).square().sum(-1)
    if not bool((after_rows <= before_rows).all()):
        raise RuntimeError('Returned packed fields increased an independently recomputed base-SSE')
    error_rot = (source - rotated).double().square().sum()
    error_native_bridge = (native.float() - mean.float() - wht(rotated)).double().square().sum()
    if not torch.allclose(error_rot, error_native_bridge, rtol=1e-5, atol=1e-5):
        raise RuntimeError('Centered WHT coordinate interpretation failed')
    # Legacy dispatch preserves both values and its original autograd behavior.
    grad_source = source.clone().requires_grad_(True)
    lc, lg = encode_key(grad_source, book, assignment='l2', gain_factor=factor)
    oc, og = legacy_encode(grad_source, book, gain_factor=factor)
    if not torch.equal(lc, oc) or not torch.equal(lg, og) or lg.requires_grad != og.requires_grad:
        raise RuntimeError('l2 direct dispatch changed the old function contract')
    original = lambda *args, **kwargs: ('legacy',)
    module = SimpleNamespace(encode_key=original)
    try:
        with scoped_dispatch(module, lambda *args, **kw: ('new',)):
            raise ValueError('intentional scoped restoration check')
    except ValueError:
        pass
    if module.encode_key is not original:
        raise RuntimeError('Constructor encode dispatcher did not restore on error')
    # CPU stub constructor exercises the actual class wrapper without a model.
    path = output / 'synthetic_book.json'
    path.write_text(json.dumps(dict(centers=accepts.float().tolist())), encoding='utf-8')
    stages = [['k', index // 4, index % 4, 0] for index in range(43)]
    cfg = dict(name='CPU_spherical_stub', method='round8_joint', key_assignment='row_cosine_cd',
        direction_book_path=str(path), metadata_reference_book_path=str(path), selected_stages=stages,
        row_cosine_sweeps=1, value_encoding='binary')
    bank_module = SimpleNamespace(encode_key=legacy_encode)
    class Base:
        settings = {}
        def _initialize(self, src, dictionary):
            self.code, self.gain = bank_module.encode_key(src, dictionary,
                assignment=type(self).settings.get('key_assignment', 'l2'))
            self.selected_residual_stages = [tuple(item) for item in type(self).settings.get('selected_stages', [])]
        def result_metadata(self):
            return dict(legacy=True)
    cls = spherical_class(Base, bank_module); cls.settings = cfg
    instance = cls(); instance._initialize(x, accepts)
    meta = instance.result_metadata()
    if bank_module.encode_key is not legacy_encode or meta['row_cosine_extra_retained_tensor_constant_bytes'] != 0:
        raise RuntimeError('Scoped class/metadata invariant failed')
    cls.settings = dict(key_assignment='l2')
    instance = cls(); instance._initialize(x, accepts)
    if instance.result_metadata() != dict(legacy=True):
        raise RuntimeError('Original l2 metadata changed')
    # The real inherited planner must keep the full224-entry rank charge even
    # when only43 stages are selected. The new wrapper never truncates config.
    from round6_depth_codec import plan_depth_stages
    frozen_path = Path(__file__).resolve().parents[2] / 'configs/a2/round7_v1/confirm_doc128_s0.json'
    if frozen_path.is_file():
        frozen = json.loads(frozen_path.read_text(encoding='utf-8-sig'))
        reference = next(v for v in frozen['variants'] if v['name'] == 'r7_joint_k1_gt_cond')
        priority = reference['selected_stages']
    else:
        priority = [[axis, layer, head, 0] for layer in range(28) for head in range(4) for axis in ('k', 'v')]
    cfg224 = dict(cfg, selected_stages=priority, res_stages_limit=43)
    declared = validate_variant(cfg224)
    nv, lh = 1792, 28 * 4
    base = lh * nv * 34 + lh * 128 * 6 + 128 + nv * 8
    plan = plan_depth_stages(layers=28, heads=4, visual_tokens=nv,
        visual_full_bytes=lh * nv * 512, base_bytes=base, factor_bytes=1344,
        budget_bits=1.8, selected_stages=priority, stage_limit=43)
    if len(priority) != 224 or plan['selected_stages'] != declared['reference_stages'] or len(plan['ranking_constant']) != 448 or plan['predicted_bytes'] != 8315013:
        raise RuntimeError('Full224 ranking plus selected43 owner contract was lost')
    return dict(status='PASS', scope='CPU synthetic math and stub integration only; no GPU/model/calibration/quality measurement',
        dense_oracle_cases=8, accepted_and_rejected_witnesses=cases,
        strict_tie_zero_duplicate_noop=True, local_trap_not_global_optimality=True,
        broadcast_WHT_packed16B_and_gain_fields=True, direct_legacy_l2_behavior_preserved=True,
        scoped_dispatch_restored_on_error=True, inherited43_stage_and_metadata_stub=True,
        full224_requested_ranking_bytes=448, selected43_complete_N1792_bytes=plan['predicted_bytes'],
        full_requested_priority_not_truncated=True,
        round8_sources_not_edited=True, added_retained_tensor_bytes=0)


def gpu_controls():
    if not torch.cuda.is_available():
        raise RuntimeError('--gpu requires actual CUDA; no CPU substitute')
    torch.manual_seed(20261118)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    with torch.inference_mode():
        for nv in (32, 256, 1792):
            source = torch.randn(1, 4, nv, 128, device='cuda')
            book = torch.randn(16, 4, device='cuda', dtype=torch.float16)
            factor = torch.tensor([.91, .98, 1.04, 1.11], device='cuda')[None, :, None, None]
            old_code, old_gain = legacy_encode(source, book, gain_factor=factor)
            old_decoded = decode_key(old_code, old_gain, book)
            old_error = (source.double() - old_decoded.double()).square().sum(-1)
            for sweeps in (1, 2):
                code, gain, stats = encode_key(source, book, gain_factor=factor,
                    sweeps=sweeps, hard_accept=True, return_stats=True)
                actual = decode_key(code, gain, book)
                error = (source.double() - actual.double()).square().sum(-1)
                same = (code == old_code).all(-1)
                if code.dtype != torch.uint8 or gain.dtype != torch.float16 or code.numel() != 4 * nv * 16 or not bool((error <= old_error).all()):
                    raise RuntimeError('Actual CUDA packed fields/base-SSE guard control failed')
                if not torch.equal(gain[same], old_gain[same]):
                    raise RuntimeError('CUDA unchanged rows lost original gain bits')
                rows.append(dict(nv=nv, sweeps=sweeps, base_SSE_nondecreasing_rows=0,
                    unchanged_J_gain_bitwise=True, code_bytes_per_token=16,
                    actual_source_base_SSE_before=float(old_error.sum()),
                    actual_source_base_SSE_after=float(error.sum()), stats=stats))
        # Exact candidate enumeration is bounded to two analytic rows; it is
        # not mistaken for a practical deployment algorithm or speed result.
        source = torch.tensor([2., 1.], device='cuda').repeat(64)[None]
        for pair in ((1.0146484375, 1.013671875), (1.009765625, 1.0087890625)):
            book = torch.full((16, 4), 20., device='cuda', dtype=torch.float16)
            book[0] = 1.; book[1] = torch.tensor(pair, device='cuda').repeat(2).half()
            code, gain = encode_key(source, book, sweeps=1, hard_accept=True)
            expected_code, expected_gain = dense_reference(source, book, 1., 1, True)
            if not torch.equal(code, expected_code) or not torch.equal(gain, expected_gain):
                raise RuntimeError('CUDA dense complete-template candidate oracle differs')
            rows.append(dict(analytic_pair=list(pair), CUDA_dense_full_template_equal=True))
    return dict(passed=True, cases=rows,
        scope='CUDA synthetic encoder arithmetic only; unchanged serving reader already has its separate R8 control; no model/task/timing claim')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    report = controls(args.output)
    report['gpu_executed'] = bool(args.gpu)
    report['GPU_controls'] = gpu_controls() if args.gpu else None
    report['sources'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                         for name in ('round8_spherical_codec.py', 'round8_spherical_runner.py', 'round8_spherical_check.py',
                                      'round8_codec.py', 'round8_bank.py')}
    (args.output / 'check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    gpu_cases = len(report['GPU_controls']['cases']) if args.gpu else 0
    (args.output / 'complete.json').write_text(json.dumps(dict(status='PASS', gpu_executed=bool(args.gpu),
        cpu_dense_oracle_cases=report['dense_oracle_cases'], GPU_cases=gpu_cases)), encoding='utf-8')
    print(json.dumps(dict(status='PASS', dense_oracle_cases=report['dense_oracle_cases'],
        gpu_executed=bool(args.gpu), GPU_cases=gpu_cases)), flush=True)


if __name__ == '__main__':
    main()
