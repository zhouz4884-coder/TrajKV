"""One query-free per-image RD prototype, without editing frozen runners.

Normal CLI: --config CONFIG --output NEW_RUN.
CPU byte control: --byte-control --output NEW_CONTROL.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


def reserve_encoder_bytes(full_visual_bytes, budget_bits, extra_bytes):
    """Construct a super-budget whose integer cap is exactly cap-extra.

    The retained FP32 w owner is added back after the existing bank encodes.
    Avoid a one-byte floating-point floor discrepancy at the boundary.
    """
    if full_visual_bytes <= 0 or not math.isfinite(budget_bits) or budget_bits <= 0 or extra_bytes < 0:
        raise ValueError('Positive denominator/budget and nonnegative extra bytes required')
    allowed = math.floor(budget_bits * full_visual_bytes / 16)
    reduced = allowed - int(extra_bytes)
    if reduced <= 0:
        raise ValueError('Sensitivity constants exhaust the complete visual budget')
    effective_bits = 16 * reduced / full_visual_bytes
    for _ in range(32):
        actual = math.floor(effective_bits * full_visual_bytes / 16)
        if actual == reduced:
            return dict(original_budget_bits=budget_bits, original_allowed_bytes=allowed,
                sensitivity_bytes=int(extra_bytes), reduced_allowed_bytes=reduced,
                super_budget_bits=effective_bits)
        effective_bits = math.nextafter(effective_bits, math.inf if actual < reduced else -math.inf)
    raise RuntimeError('Cannot represent exact reserved integer budget')


def byte_control():
    from round6_depth_codec import plan_depth_stages
    rows = []
    for layers, heads, nv in [(28, 4, n) for n in (128, 576, 1024, 1280, 1792, 2304, 4096)] + [(32, 32, 576)]:
        full = layers * heads * nv * 128 * 4
        weights = 2 * layers * heads * 4
        reserve = reserve_encoder_bytes(full, 1.8, weights)
        base = layers * heads * nv * 50 + layers * heads * 128 * 6 + 2048 + nv * 8
        factors = 3 * layers * heads * 4
        units = [(axis, layer, head) for axis in ('k', 'v') for layer in range(layers) for head in range(heads)]
        order = [(*unit, stage) for stage in (0, 1) for unit in units]
        common = dict(layers=layers, heads=heads, visual_tokens=nv,
            visual_full_bytes=full, base_bytes=base, selected_stages=order)
        via_super = plan_depth_stages(**common, factor_bytes=factors, budget_bits=reserve['super_budget_bits'])
        direct = plan_depth_stages(**common, factor_bytes=factors + weights, budget_bits=1.8)
        if via_super['allowed_bytes'] != reserve['reduced_allowed_bytes']:
            raise RuntimeError('Super integer cap does not equal the exact reserved cap')
        if via_super['selected_stages'] != direct['selected_stages']:
            raise RuntimeError('Reserve/add-back changed the selected affordable stage prefix')
        if via_super['predicted_bytes'] + weights != direct['predicted_bytes'] or via_super['fixed_bytes'] + weights != direct['fixed_bytes']:
            raise RuntimeError('Sensitivity owner was omitted or counted twice')
        if direct['predicted_bytes'] > reserve['original_allowed_bytes']:
            raise RuntimeError('Restored complete owner exceeds original cap')
        rows.append(dict(layers=layers, heads=heads, visual_tokens=nv, sensitivity_bytes=weights,
            ranking_bytes=len(order) * 2, descriptor_bytes=5,
            selected_stages=len(direct['selected_stages']), complete_bytes=direct['predicted_bytes'],
            original_allowed_bytes=reserve['original_allowed_bytes'],
            actual_bits=16 * direct['predicted_bytes'] / full, reserve_equals_direct=True))
    return dict(status='PASS', actual_cuda=False, cases=rows,
        scope='CPU integer byte-cap and prefix controls only; no reader/model/quality claim')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--byte-control', action='store_true')
    args = parser.parse_args()
    if args.byte_control:
        report = byte_control()
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'byte_control.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(dict(status=report['status'], cases=len(report['cases']), actual_cuda=False)), flush=True)
        return
    if args.config is None:
        parser.error('--config required for actual GPU runs')
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    variants = [v for v in config['variants'] if v['method'] == 'round6_image_rd']
    if len(variants) != 1:
        raise ValueError('Run exactly one per-image RD prototype variant; no hyperparameter family')
    if not variants[0].get('image_rd_sensitivity_path'):
        raise ValueError('The frozen image_rd_sensitivity_path is required')
    # Import/registration are deliberately delayed so CPU cap checks work on
    # hosts without Triton, transformers or a CUDA driver.
    import round6_runner
    import factory
    previous = factory.bank_class

    def choose(method):
        if method == 'round6_image_rd':
            from round6_image_rd_bank import ImageRDResidualBank
            return ImageRDResidualBank
        return previous(method)

    factory.bank_class = choose
    external_keys = ('gamma_factor_path', 'depth_policy_path', 'image_rd_sensitivity_path',
        'functional_v_factor_path', 'spectrum_artifact_path', 'metric_pq_artifact_path', 'interaction_policy_path')
    extras = {v[key]: sha(v[key]) for v in config['variants'] for key in external_keys if v.get(key)}
    import run_representation_variants
    run_representation_variants.main()
    for path, digest in extras.items():
        if sha(path) != digest:
            raise RuntimeError('Frozen external input changed during prototype run: ' + path)
    manifest_path = args.output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['external_codec_inputs'].update(extras)
    manifest['image_rd_scope'] = 'One query-free current-image visual-KV RD prototype; w fully charged; extra prepass and serving encode measured together; development only'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
