"""Prepare compact encoder-only 4D query-metric blocks from real T0 moments."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code/a2'))
from query_spectrum_math import hadamard_matrix, symmetric


def query_moment(row, regime):
    prefix = 'q_' + regime
    count = torch.as_tensor(row.get(prefix + '_count', 0)).double()
    if not bool((count > 0).all()):
        return None
    outer = torch.as_tensor(row[prefix + '_outer_sum']).double()
    while count.ndim < outer.ndim:
        count = count.unsqueeze(-1)
    return symmetric(outer / count)


def prepare(statistics, regime='mixed', shrinkage=.05, mode='query'):
    if not 0 <= shrinkage <= 1:
        raise ValueError('Shrinkage must be in [0,1]')
    if regime not in ('prefill', 'decode', 'mixed') or mode not in ('query', 'identity'):
        raise ValueError('Invalid metric mode/regime')
    hmat = hadamard_matrix(128, dtype=torch.float64)
    blocks = []
    fallbacks, zero_blocks = [], 0
    for row in sorted(statistics['layers'], key=lambda item: item['layer']):
        heads = torch.as_tensor(row['k_sum']).shape[0]
        if mode == 'identity':
            block = torch.eye(4, dtype=torch.float64).expand(heads, 32, 4, 4).clone()
        else:
            prefill, decode = query_moment(row, 'prefill'), query_moment(row, 'decode')
            if prefill is None and decode is None:
                raise ValueError('No query observations for layer ' + str(row['layer']))
            if prefill is None:
                prefill = decode
                fallbacks.append({'layer': row['layer'], 'missing': 'prefill', 'used': 'decode'})
            if decode is None:
                decode = prefill
                fallbacks.append({'layer': row['layer'], 'missing': 'decode', 'used': 'prefill'})
            moment = prefill if regime == 'prefill' else decode if regime == 'decode' else (prefill + decode) * .5
            rotated = symmetric(hmat.T @ moment @ hmat)
            block = torch.stack([rotated[:, i:i + 4, i:i + 4] for i in range(0, 128, 4)], 1)
            # Clean roundoff before shrinkage, then normalize each block trace.
            values, vectors = torch.linalg.eigh(block)
            values = values.clamp_min(0.)
            block = (vectors * values.unsqueeze(-2)) @ vectors.transpose(-1, -2)
            trace = block.diagonal(dim1=-2, dim2=-1).sum(-1)
            zero = trace <= 1e-20
            zero_blocks += int(zero.sum())
            identity = torch.eye(4, dtype=block.dtype).expand_as(block)
            block = block / (trace / 4).clamp_min(1e-20)[..., None, None]
            block = torch.where(zero[..., None, None], identity, block)
            block = (1 - shrinkage) * block + shrinkage * identity
        blocks.append(block.half().contiguous())
    metric = torch.stack(blocks)
    metadata = {'mode': mode, 'query_regime': regime, 'mixed_prefill_weight': .5,
                'shrinkage_to_identity': shrinkage, 'query_regime_fallbacks': fallbacks,
                'zero_blocks_replaced_by_identity': zero_blocks,
                'source_metadata': statistics.get('metadata', {}),
                'coordinate_system': 'normalized Walsh-Hadamard K residual; contiguous four-dimensional PQ blocks',
                'assignment_scope': 'block diagonal E[q_rot q_rot^T]; initial amplitude fixed at source_norm/sqrt128; final original norm-match gain retained',
                'not_a_global_rate_distortion_optimum': True,
                'encoder_only': True, 'serving_reader_change': False,
                'encoder_metric_bytes': metric.numel() * metric.element_size(),
                'metric_dtype': str(metric.dtype),
                'payload_bits_K': 2, 'payload_bits_V': 1}
    return {'format': 'trajkv_metric_pq_v1', 'metric_blocks': metric, 'metadata': metadata}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stats', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--regime', choices=('prefill', 'decode', 'mixed'), default='mixed')
    p.add_argument('--mode', choices=('query', 'identity'), default='query')
    p.add_argument('--shrinkage', type=float, default=.05)
    a = p.parse_args()
    torch.set_num_threads(4)
    a.output.mkdir(parents=True, exist_ok=False)
    source = torch.load(a.stats, map_location='cpu', weights_only=False)
    artifact = prepare(source, a.regime, a.shrinkage, a.mode)
    artifact['metadata']['statistics_sha256'] = hashlib.sha256(a.stats.read_bytes()).hexdigest()
    torch.save(artifact, a.output / 'encoder_metric.pt')
    (a.output / 'summary.json').write_text(json.dumps(artifact['metadata'], indent=2), encoding='utf-8')
    print(json.dumps(artifact['metadata']))


if __name__ == '__main__':
    main()
