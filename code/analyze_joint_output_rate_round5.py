"""Gaussian K/V payload allocation in a *common local output-error* metric.

Input metrics come from real same-query attention/WO Jacobians.  K's visual
token Jacobian sum is subsampled and rescaled; V's attention overlap retains
GQA cross-head terms.  The design model nevertheless assumes zero-mean,
independent token errors with a shared source-error covariance, linearizes K,
and combines pooled source covariance with an averaged metric.  It is neither
an actual codec nor a prediction of free-generation ANLS.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code/a2'))
from query_spectrum_math import gaussian_waterfill, moment, symmetric


def plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def positive_matrix(value):
    """Project only negative roundoff directions, without inventing a ridge."""
    value = symmetric(torch.as_tensor(value).double())
    raw, vectors = torch.linalg.eigh(value)
    positive = raw.clamp_min(0.)
    reconstructed = (vectors * positive.unsqueeze(-2)) @ vectors.transpose(-1, -2)
    diagnostics = {'raw_min_eigenvalue': raw.amin(-1), 'negative_eigenvalue_count': (raw < 0).sum(-1),
                   'negative_eigenvalue_abs_sum': (-raw.clamp_max(0.)).sum(-1),
                   'positive_trace': positive.sum(-1)}
    return reconstructed, positive, vectors, diagnostics


def output_spectrum(covariance, metric):
    covariance, values, vectors, cdiag = positive_matrix(covariance)
    metric, _, _, mdiag = positive_matrix(metric)
    root = (vectors * values.sqrt().unsqueeze(-2)) @ vectors.transpose(-1, -2)
    weighted = symmetric(root @ metric @ root)
    spectrum = torch.linalg.eigvalsh(weighted).flip(-1).clamp_min(0.)
    return spectrum, {'source_covariance': cdiag, 'local_output_metric': mdiag}


def normalized_metric(row, key):
    metric = torch.as_tensor(row[key]).double()
    count = torch.as_tensor(row['output_metric_count']).double()
    if not bool((count > 0).all()):
        raise ValueError('No common-output observations at layer ' + str(row['layer']))
    while count.ndim < metric.ndim:
        count = count.unsqueeze(-1)
    return symmetric(metric / count)


def safe_ratio(numerator, denominator):
    if isinstance(denominator, torch.Tensor):
        return torch.where(denominator > 0, numerator / denominator.clamp_min(1e-300), torch.zeros_like(numerator))
    return numerator / denominator if denominator > 0 else None


def spectrum_description(mu):
    d = mu.shape[-1]
    total = mu.sum(-1)
    ranks = sorted(set(rank for rank in (4, 8, 16, 32, 64, d) if rank <= d))
    return {'trace': total,
            'effective_rank': safe_ratio(total.square(), mu.square().sum(-1)),
            'top_r_energy_fraction': {str(rank): safe_ratio(mu[..., :rank].sum(-1), total) for rank in ranks},
            'zero_metric': total == 0}


def build(statistics, payload_bits_kv=1.5, fixed_k_bits=2., fixed_v_bits=1.):
    if payload_bits_kv < 0 or fixed_k_bits < 0 or fixed_v_bits < 0:
        raise ValueError('Payload rates must be nonnegative')
    if abs((fixed_k_bits + fixed_v_bits) / 2 - payload_bits_kv) > 1e-9:
        raise ValueError('The fixed K/V comparison must use the same average payload budget')
    rows = sorted(statistics['layers'], key=lambda row: row['layer'])
    if not rows or [row['layer'] for row in rows] != list(range(len(rows))):
        raise ValueError('Contiguous, nonempty layer statistics required')
    all_k, all_v, reports = [], [], []
    dimension = None
    for row in rows:
        _, ck = moment(row['k_sum'], row['k_outer_sum'], row['k_count'], centered=True)
        _, cv = moment(row['v_sum'], row['v_outer_sum'], row['v_count'], centered=True)
        mk = normalized_metric(row, 'k_output_metric_sum')
        mv = normalized_metric(row, 'v_output_metric_sum')
        if not (ck.shape == cv.shape == mk.shape == mv.shape):
            raise ValueError('Source/metric shape mismatch at layer ' + str(row['layer']))
        dimension = ck.shape[-1]
        ek, dk = output_spectrum(ck, mk)
        ev, dv = output_spectrum(cv, mv)
        all_k.append(ek)
        all_v.append(ev)
        # One common water level *within each layer/KV-head*, budget 2*D*B.
        # At D=128 and B=1.5, every such unit has 384 payload bits.
        joint = gaussian_waterfill(torch.cat((ek, ev), -1), payload_bits_kv)
        bk = joint['allocation_bits'][..., :dimension].mean(-1)
        bv = joint['allocation_bits'][..., dimension:].mean(-1)
        fixed_k = gaussian_waterfill(ek, fixed_k_bits)
        fixed_v = gaussian_waterfill(ev, fixed_v_bits)
        fixed_separate_distortion = fixed_k['absolute_distortion'] + fixed_v['absolute_distortion']
        uniform_distortion = ek.sum(-1) * (2 ** (-2 * fixed_k_bits)) + ev.sum(-1) * (2 ** (-2 * fixed_v_bits))
        baseline_trace = ek.sum(-1) + ev.sum(-1)
        reports.append({'layer': row['layer'], 'metric_observations': row['output_metric_count'],
            'K_output_spectrum': ek, 'V_output_spectrum': ev,
            'K_spectrum_description': spectrum_description(ek), 'V_spectrum_description': spectrum_description(ev),
            'matrix_diagnostics': {'K': dk, 'V': dv},
            'within_unit_joint_waterfill': {
                'theta': joint['theta'], 'K_mean_payload_bits': bk, 'V_mean_payload_bits': bv,
                'K_coordinate_bits': joint['allocation_bits'][..., :dimension],
                'V_coordinate_bits': joint['allocation_bits'][..., dimension:],
                'absolute_distortion': joint['absolute_distortion'],
                'relative_distortion': safe_ratio(joint['absolute_distortion'], baseline_trace),
                'total_payload_bits': joint['allocation_bits'].sum(-1),
                'budget_residual_bits': joint['allocation_bits'].sum(-1) - 2 * dimension * payload_bits_kv},
            'fixed_KV_rate_spectral_waterfill': {
                'K_mean_payload_bits': fixed_k_bits, 'V_mean_payload_bits': fixed_v_bits,
                'theta_K': fixed_k['theta'], 'theta_V': fixed_v['theta'],
                'absolute_distortion': fixed_separate_distortion,
                'relative_distortion': safe_ratio(fixed_separate_distortion, baseline_trace)},
            'fixed_KV_rate_coordinate_uniform': {
                'absolute_distortion': uniform_distortion,
                'relative_distortion': safe_ratio(uniform_distortion, baseline_trace)},
            'joint_to_fixed_separate_distortion_ratio': safe_ratio(joint['absolute_distortion'], fixed_separate_distortion)})
    all_k, all_v = torch.stack(all_k), torch.stack(all_v)
    layers, heads, d = all_k.shape
    pair_spectra = torch.stack((all_k, all_v), -2)
    # Optional additional model: a common global theta can transfer bits across
    # layers and KV-heads. Summed local residual errors are not a final-model loss.
    global_fit = gaussian_waterfill(pair_spectra.reshape(-1), payload_bits_kv)
    global_allocation = global_fit['allocation_bits'].reshape(layers, heads, 2, d)
    per_unit_k = torch.stack([row['within_unit_joint_waterfill']['K_mean_payload_bits'] for row in reports])
    per_unit_v = torch.stack([row['within_unit_joint_waterfill']['V_mean_payload_bits'] for row in reports])
    joint_dist = sum(float(row['within_unit_joint_waterfill']['absolute_distortion'].sum()) for row in reports)
    separate_dist = sum(float(row['fixed_KV_rate_spectral_waterfill']['absolute_distortion'].sum()) for row in reports)
    uniform_dist = sum(float(row['fixed_KV_rate_coordinate_uniform']['absolute_distortion'].sum()) for row in reports)
    total_trace = float(pair_spectra.sum())
    shift = float(per_unit_v.mean()) - fixed_v_bits
    tolerance = .05
    direction = ('more_V_payload_in_this_Gaussian_reference' if shift > tolerance else
                 'more_K_payload_in_this_Gaussian_reference' if shift < -tolerance else
                 'near_fixed_KV_split_in_this_Gaussian_reference')
    report = {
        'status': 'COMPLETE', 'layers': layers, 'kv_heads': heads, 'source_dimension': d,
        'statistics_scope': statistics.get('metadata', {}),
        'budget': {'mean_payload_bits_per_KV_scalar': payload_bits_kv,
            'payload_bits_per_layer_KV_head': 2 * d * payload_bits_kv,
            'fixed_K_mean_payload_bits': fixed_k_bits, 'fixed_V_mean_payload_bits': fixed_v_bits,
            'metadata_basis_gain_mean_codebook_bytes_included': False},
        'metric_units': 'squared local same-query attention output after WO, averaged over output_metric_count',
        'source_covariance_scope': {
            'K': 'pooled moments of per-image centered visual K; per-image means stored by codec are excluded',
            'V': 'pooled covariance after one global mean per layer/head; this is not per-image/per-window centroid residual covariance'},
        'model_assumptions': [
            'Gaussian ideal source rate-distortion with continuous, possibly fractional coordinate rates; finite PQ/scalar codecs need not attain it',
            'zero-mean independent visual-token quantization errors with a shared covariance',
            'first-order K softmax Jacobian at full-cache queries; V attention held fixed',
            'GQA Q-head cross terms retained within each KV-head; cross-K/V and cross-KV-head error terms not part of this separable quadratic model',
            'K token-Jacobian sum estimated from uniformly spaced visual tokens and scaled Nv/sample_count',
            'averaged local output metric combined with pooled source covariance; their conditional dependence is not retained',
            'global allocation additionally sums local errors across layers, which is not the final-model loss'],
        'not_a_deployable_codec': True, 'not_an_ANLS_prediction': True,
        'within_unit_joint_summary': {
            'K_mean_payload_bits': float(per_unit_k.mean()), 'V_mean_payload_bits': float(per_unit_v.mean()),
            'K_rate_min_max': [float(per_unit_k.min()), float(per_unit_k.max())],
            'V_rate_min_max': [float(per_unit_v.min()), float(per_unit_v.max())],
            'V_rate_shift_vs_fixed': shift, 'design_direction_only': direction,
            'summed_absolute_distortion': joint_dist, 'relative_distortion': safe_ratio(joint_dist, total_trace),
            'fixed_KV_spectral_summed_distortion': separate_dist,
            'fixed_KV_coordinate_uniform_summed_distortion': uniform_dist,
            'joint_to_fixed_spectral_distortion_ratio': safe_ratio(joint_dist, separate_dist),
            'joint_to_fixed_uniform_distortion_ratio': safe_ratio(joint_dist, uniform_dist),
            'K_total_spectral_trace': float(all_k.sum()), 'V_total_spectral_trace': float(all_v.sum())},
        'global_joint_reference': {
            'additional_scope': 'one theta across K,V,layers,heads; average rate fixed globally, not in every unit',
            'theta': global_fit['theta'],
            'K_mean_payload_bits': float(global_allocation[..., 0, :].mean()),
            'V_mean_payload_bits': float(global_allocation[..., 1, :].mean()),
            'K_rate_by_layer_KV_head': global_allocation[..., 0, :].mean(-1),
            'V_rate_by_layer_KV_head': global_allocation[..., 1, :].mean(-1),
            'total_payload_bits': float(global_allocation.sum()),
            'absolute_distortion': global_fit['absolute_distortion'],
            'relative_distortion': global_fit['relative_distortion'],
            'coordinate_bits': global_allocation},
        'per_layer': reports}
    return plain(report)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stats', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--payload-bits-kv', type=float, default=1.5)
    p.add_argument('--fixed-k-bits', type=float, default=2.)
    p.add_argument('--fixed-v-bits', type=float, default=1.)
    a = p.parse_args()
    torch.set_num_threads(4)
    a.output.mkdir(parents=True, exist_ok=False)
    stats = torch.load(a.stats, map_location='cpu', weights_only=False)
    report = build(stats, a.payload_bits_kv, a.fixed_k_bits, a.fixed_v_bits)
    report['statistics_sha256'] = hashlib.sha256(a.stats.read_bytes()).hexdigest()
    (a.output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (a.output / 'complete.json').write_text(json.dumps({'status': 'COMPLETE', 'statistics_sha256': report['statistics_sha256']}), encoding='utf-8')
    print(json.dumps({'status': 'COMPLETE', 'within_unit_joint_summary': report['within_unit_joint_summary'],
                      'global_K_rate': report['global_joint_reference']['K_mean_payload_bits'],
                      'global_V_rate': report['global_joint_reference']['V_mean_payload_bits']}))


if __name__ == '__main__':
    main()
