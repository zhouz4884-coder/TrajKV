"""Fit round-5 source/query bases and export Gaussian design references.

No benchmark score is predicted here.  Statistics must come from real model
forwards on the declared calibration split.  The output .pt supplies paired
encoder/decoder transforms to QuerySpectrumPort, a materialized quality port.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code/a2'))
from query_spectrum_math import moment, pca_basis, weighted_basis, gaussian_waterfill, weighted_projection_error


def _pick(record, names, default=None):
    for name in names:
        if name in record:
            return record[name]
    if default is not None:
        return default
    raise KeyError('Missing statistics; expected one of ' + repr(names))


def _query(record, regime):
    prefix = 'q_' + regime
    # Counts include all sampled positions and the seven grouped Q-heads.
    count = _pick(record, (prefix + '_count', 'count_' + prefix), default=0)
    if not bool((torch.as_tensor(count) > 0).all()):
        return None
    sums = _pick(record, (prefix + '_sum', 'sum_' + prefix))
    outers = _pick(record, (prefix + '_outer_sum', prefix + '_cross_sum', 'sum_' + prefix + '_outer'))
    _, second = moment(sums, outers, count, centered=False)
    return second


def _plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def build(statistics, ridge=1e-4, mix_prefill=.5):
    layers = sorted(statistics['layers'], key=lambda row: row['layer'])
    if [row['layer'] for row in layers] != list(range(len(layers))):
        raise ValueError('Contiguous layer indices required')
    transforms = {name: {'encode': [], 'decode': []} for name in
                  ('k_pca', 'query_aware_prefill', 'query_aware_decode', 'query_aware_mixed')}
    report = []
    fallbacks = []
    for row in layers:
        _, ck = moment(row['k_sum'], row['k_outer_sum'], row['k_count'], centered=True)
        qpre, qdec = _query(row, 'prefill'), _query(row, 'decode')
        if qpre is None and qdec is None:
            raise ValueError('No observed query moments for layer ' + str(row['layer']))
        if qpre is None:
            qpre = qdec
            fallbacks.append({'layer': row['layer'], 'missing': 'prefill', 'used': 'decode'})
        if qdec is None:
            qdec = qpre
            fallbacks.append({'layer': row['layer'], 'missing': 'decode', 'used': 'prefill'})
        # Physical attention scores are q.k/sqrt(D).  Uniform scaling leaves
        # eigenvectors unchanged but matters for reporting absolute distortion.
        d = ck.shape[-1]
        qm = {'prefill': qpre / d, 'decode': qdec / d,
              'mixed': (mix_prefill * qpre + (1 - mix_prefill) * qdec) / d}
        pca_e, pca_d, pca_values = pca_basis(ck)
        transforms['k_pca']['encode'].append(pca_e.float())
        transforms['k_pca']['decode'].append(pca_d.float())
        layer_report = {'layer': row['layer'], 'k_covariance_trace': ck.diagonal(dim1=-2, dim2=-1).sum(-1),
                        'k_pca_eigenvalues': pca_values, 'query_regimes': {}}
        for regime, mq in qm.items():
            encode, decode, values, raw, regularized = weighted_basis(ck, mq, ridge)
            name = 'query_aware_' + regime
            transforms[name]['encode'].append(encode.float())
            transforms[name]['decode'].append(decode.float())
            total = values.sum(-1, keepdim=True).clamp_min(1e-300)
            ranks = (4, 8, 16, 32, 64, ck.shape[-1])
            ranks = sorted(set(r for r in ranks if r <= ck.shape[-1]))
            ideal = {str(rate): gaussian_waterfill(values, rate) for rate in (.5, 1., 1.5, 2., 3., 4.)}
            layer_report['query_regimes'][regime] = {
                'eigenvalues': values, 'energy_fraction': values.cumsum(-1) / total,
                'gaussian_weighted_mse_reference': ideal,
                'unregularized_source_min_eigenvalue': raw.amin(-1),
                'regularized_source_min_eigenvalue': regularized.amin(-1),
                'projection_relative_distortion': {str(r): weighted_projection_error(ck, mq, encode, decode, r) for r in ranks},
                'pca_projection_relative_distortion': {str(r): weighted_projection_error(ck, mq, pca_e, pca_d, r) for r in ranks}}
        # Optional V reference: identity is explicitly a source-MSE proxy.
        if all(key in row for key in ('v_sum', 'v_outer_sum', 'v_count')):
            _, cv = moment(row['v_sum'], row['v_outer_sum'], row['v_count'], centered=True)
            metric = row.get('v_output_metric', row.get('wo_metric', row.get('wo_gram')))
            v_label = ('provided_local_output_metric' if 'v_output_metric' in row else
                       'unweighted_WO_gram_proxy' if metric is not None else 'identity_source_mse_proxy')
            mv = torch.eye(cv.shape[-1]).expand_as(cv) if metric is None else torch.as_tensor(metric).double()
            _, _, vv, _, _ = weighted_basis(cv, mv, ridge)
            layer_report['value_reference'] = {'metric_scope': v_label, 'eigenvalues': vv,
                'gaussian_weighted_mse_reference': {str(rate): gaussian_waterfill(vv, rate) for rate in (.5, 1., 1.5, 2., 3.)}}
            if all(key in row for key in ('v_error_sum', 'v_error_outer_sum', 'v_error_count')):
                _, ce = moment(row['v_error_sum'], row['v_error_outer_sum'], row['v_error_count'], centered=True)
                _, _, ve, _, _ = weighted_basis(ce, mv, ridge)
                layer_report['value_residual_reference'] = {
                    'metric_scope': v_label, 'source_scope': 'centered empirical V coding residual covariance',
                    'weighted_eigenvalues': ve,
                    'effective_rank': ve.sum(-1).square() / ve.square().sum(-1).clamp_min(1e-300),
                    'rank2_fraction': ve[..., :2].sum(-1) / ve.sum(-1).clamp_min(1e-300),
                    'scope': 'second-moment design proxy, not full attention-conditioned output error'}
        report.append(_plain(layer_report))
    for item in transforms.values():
        item['encode'] = torch.stack(item['encode'])
        item['decode'] = torch.stack(item['decode'])
    artifact = {'format': 'trajkv_query_spectrum_v1', 'transforms': transforms,
                'metadata': {'source_metadata': statistics.get('metadata', {}), 'source_mean_scope': 'per-image visual K mean',
                    'query_metric': 'uncentered E[qq^T]/D, seven Q-head samples per KV-head',
                    'query_mix_prefill': mix_prefill, 'source_eigen_floor_relative_to_max': ridge,
                    'query_regime_fallbacks': fallbacks,
                    'row_vector_convention': 'y=(k-mu)@A; reconstruction=y_hat@B+mu; full rank A@B=I',
                    'key_payload_bits_current': 2, 'value_payload_bits_current': 1,
                    'materialized_quality_prototype': True}}
    summary = {'status': 'COMPLETE', 'layers': len(layers), 'statistics_scope': statistics.get('metadata', {}),
               'model_reference': 'Gaussian ideal weighted squared-error rate-distortion; independent source/query second-moment design approximation',
               'not_a_benchmark_prediction': True, 'not_a_deployable_codec': True,
               'joint_water_level': {'computed': False, 'reason': 'K score error and V output error have different units; a common output/loss metric is required before combining spectra'},
               'per_layer': report}
    aggregate = {}
    for regime in ('prefill', 'decode', 'mixed'):
        entries = [item['query_regimes'][regime] for item in report]
        energy = torch.tensor([entry['energy_fraction'] for entry in entries], dtype=torch.float64)
        ranks = sorted(set(r for r in (4, 8, 16, 32, 64, energy.shape[-1]) if r <= energy.shape[-1]))
        aggregate[regime] = {
            'mean_top_r_energy': {str(r): float(energy[..., r - 1].mean()) for r in ranks},
            'mean_actual_projection_D': {str(r): float(torch.tensor([entry['projection_relative_distortion'][str(r)] for entry in entries]).mean()) for r in ranks},
            'mean_pca_projection_D': {str(r): float(torch.tensor([entry['pca_projection_relative_distortion'][str(r)] for entry in entries]).mean()) for r in ranks},
            'mean_gaussian_D_at_K_payload2': float(torch.tensor([entry['gaussian_weighted_mse_reference']['2.0']['relative_distortion'] for entry in entries]).mean()),
            'aggregation': 'unweighted macro average over layer/KV-head; per-group spectra retained'}
    summary['aggregate'] = aggregate
    return artifact, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ridge', type=float, default=1e-4)
    parser.add_argument('--mix-prefill', type=float, default=.5)
    args = parser.parse_args()
    if not 0 < args.ridge < 1 or not 0 <= args.mix_prefill <= 1:
        raise ValueError('Invalid regularization or query mixture')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    statistics = torch.load(args.stats, map_location='cpu', weights_only=False)
    artifact, summary = build(statistics, args.ridge, args.mix_prefill)
    fingerprint = hashlib.sha256(args.stats.read_bytes()).hexdigest()
    artifact['metadata']['statistics_sha256'] = fingerprint
    summary['statistics_sha256'] = fingerprint
    torch.save(artifact, args.output / 'transforms.pt')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (args.output / 'complete.json').write_text(json.dumps({'status': 'COMPLETE', 'statistics_sha256': fingerprint}), encoding='utf-8')
    print(json.dumps({'status': 'COMPLETE', 'layers': summary['layers'], 'output': str(args.output), 'aggregate': summary['aggregate']}))


if __name__ == '__main__':
    main()
