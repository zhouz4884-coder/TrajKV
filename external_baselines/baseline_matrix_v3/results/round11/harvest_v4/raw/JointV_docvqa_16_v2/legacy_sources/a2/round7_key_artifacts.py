"""Explicit Round7 book/factor/metric provenance; never weaken old hash checks."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch


FACTOR_NAMES = ('key_head_factors', 'value_head_factors', 'value_center_offsets')


def factor_values_hash(data):
    fields = {key: data.get(key) for key in (*FACTOR_NAMES, 'key_gain_mode', 'value_group_size', 'parameter_count')}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verify_factor_inheritance(data):
    record = data['round7_inheritance']
    if record.get('schema') != 'trajkv_round7_explicit_factor_inheritance_v1':
        raise ValueError('Unrecognized factor inheritance declaration')
    if record.get('inherited_values_sha256') != factor_values_hash(data):
        raise ValueError('Inherited factor numerical values or settings changed')
    for key in ('source_factor_sha256', 'source_encoder_book_sha256', 'target_encoder_book_sha256'):
        if len(record.get(key, '')) != 64:
            raise ValueError('Full source/target SHA256 required for inherited factors')
    if record['target_encoder_book_sha256'] != data['encoder_book_sha256']:
        raise ValueError('Inherited factor target book declaration mismatch')


def inherit_factors(source, book, output):
    source, book, output = Path(source), Path(book), Path(output)
    if output.exists():
        raise FileExistsError(output)
    data = json.loads(source.read_text(encoding='utf-8'))
    target = copy.deepcopy(data)
    target['encoder_book_sha256'] = hashlib.sha256(book.read_bytes()).hexdigest()
    target['round7_inheritance'] = dict(schema='trajkv_round7_explicit_factor_inheritance_v1',
        source_factor_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_encoder_book_sha256=data['encoder_book_sha256'],
        target_encoder_book_sha256=target['encoder_book_sha256'],
        inherited_values_sha256=factor_values_hash(data),
        scope='All numerical key/value factors copied unchanged; no refit or new-book calibration claim')
    verify_factor_inheritance(target)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(target, indent=2, allow_nan=False), encoding='utf-8')
    return dict(path=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest(), inheritance=target['round7_inheritance'])


def load_rank1_metric(path, layers, heads, device):
    path = Path(path)
    record = json.loads(path.read_text(encoding='utf-8'))
    if record.get('schema') != 'trajkv_round7_rank1_metric_v1' or record.get('basis') != 'centered_wht':
        raise ValueError('Explicit Round7 rank-one metric in centered_wht basis required')
    if record.get('calibration_only') is not True:
        raise ValueError('Rank-one metric must explicitly declare calibration_only=true')
    u = torch.tensor(record['u'], dtype=torch.float16, device=device).contiguous()
    gamma = torch.tensor(record['gamma'], dtype=torch.float32, device=device)
    if gamma.ndim == 0:
        gamma = gamma.expand(layers, heads).clone()
    gamma = gamma.contiguous()
    if tuple(u.shape) != (layers, heads, 128) or tuple(gamma.shape) != (layers, heads):
        raise ValueError('Rank-one metric must contain u[L,Hkv,128] and gamma scalar or [L,Hkv]')
    if not bool(torch.isfinite(u).all()) or not bool(torch.isfinite(gamma).all()) or not bool((gamma >= 0).all()):
        raise ValueError('Finite rank-one metric and nonnegative gamma required')
    metadata = {key: value for key, value in record.items() if key not in ('u', 'gamma')}
    metadata.update(deployed_u_dtype='FP16', deployed_gamma_dtype='FP32',
                    metric='M=I+gamma*u*u^T; u is used as stored without silent normalization',
                    tensor_bytes=u.numel() * 2 + gamma.numel() * 4)
    return u, gamma, metadata, hashlib.sha256(path.read_bytes()).hexdigest()


def train_book(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    # Generate and hash the exact fixed-seed CPU Gaussian input, also for GPU fit.
    generator = torch.Generator(device='cpu').manual_seed(args.seed)
    train_cpu = torch.randn(args.vectors, 4, generator=generator)
    test_cpu = torch.randn(args.validation_vectors, 4, generator=generator)
    input_hash = hashlib.sha256(train_cpu.numpy().tobytes() + test_cpu.numpy().tobytes()).hexdigest()
    x, test = train_cpu.to(device), test_cpu.to(device)
    initialization = torch.randperm(len(train_cpu), generator=generator)[:args.words]
    centers = x[initialization.to(device)].clone()
    history = []
    with torch.inference_mode():
        for _ in range(args.iterations):
            distance = x.square().sum(-1, keepdim=True) + centers.square().sum(-1)[None] - 2 * (x @ centers.T)
            index = distance.argmin(-1)
            count = torch.bincount(index, minlength=args.words).float()
            total = torch.zeros_like(centers).index_add_(0, index, x)
            updated = total / count[:, None].clamp_min(1)
            centers = torch.where(count[:, None] > 0, updated, centers)
            history.append(float((x - centers[index]).square().mean()))
        centers = centers.half().float()
        distance = test.square().sum(-1, keepdim=True) + centers.square().sum(-1)[None] - 2 * (test @ centers.T)
        mse = float((test - centers[distance.argmin(-1)]).square().mean())
    record = dict(schema='trajkv.direction-book.v1', dimension=4, centroids=args.words,
        nominal_bits_per_scalar=(4 if args.words == 16 else 8) / 4, centers=centers.cpu().tolist(),
        storage_dtype='FP16', actual_book_bytes=args.words * 4 * 2,
        inputHash=input_hash, input_hash_scope='SHA256 concatenated train then validation FP32 little-endian row-major tensors',
        seed=args.seed, training_vectors=args.vectors, validation_vectors=args.validation_vectors,
        iterations=args.iterations, training_mse_history=history, validation_mse=mse,
        fit_device=str(device), cpu_threads=args.threads,
        training_distribution='fixed-seed standard Gaussian prior; no model/query/task data',
        scope='quantizer constant fit only; not model accuracy or deployment efficiency')
    path = args.output / f'gaussian_vq_g4_{args.words}.json'
    path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    result = dict(status='COMPLETE', book=path.name, inputHash=input_hash,
                  book_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), validation_mse=mse,
                  actual_book_bytes=record['actual_book_bytes'])
    (args.output / 'complete.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    book = sub.add_parser('book')
    book.add_argument('--output', type=Path, required=True)
    book.add_argument('--device', default='cpu')
    book.add_argument('--words', type=int, choices=(16, 256), default=16)
    book.add_argument('--seed', type=int, default=20261116)
    book.add_argument('--vectors', type=int, default=65536)
    book.add_argument('--validation-vectors', type=int, default=16384)
    book.add_argument('--iterations', type=int, default=40)
    book.add_argument('--threads', type=int, default=1)
    factors = sub.add_parser('inherit-factors')
    factors.add_argument('--source', type=Path, required=True)
    factors.add_argument('--book', type=Path, required=True)
    factors.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = train_book(args) if args.command == 'book' else inherit_factors(args.source, args.book, args.output)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
