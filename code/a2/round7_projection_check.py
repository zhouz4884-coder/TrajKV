"""CPU controls for the reused projection-preserving gain, not task quality."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch
import round7_key_codec as codec

PREVIOUS_CODEC_SHA256 = 'ff0633da7f4f3448746e81c1d93842881e3469b2e1c31e323acddec0707e5bec'
PREVIOUS_ENCODE_KEY_AST_SHA256 = '87d2c3e16cd908b1b7a4079776a9902f8cb4cab906ad1098b12fbe8b9612a993'


def previous_amplitude(source, template, mode='norm', u=None, gamma=None):
    source, template = source.float(), template.float()
    if mode == 'norm':
        return source.norm(dim=-1, keepdim=True) / template.norm(dim=-1, keepdim=True).clamp_min(1e-20)
    numerator = (source * template).sum(-1, keepdim=True)
    denominator = template.square().sum(-1, keepdim=True)
    if mode == 'metric_ls':
        if u is None or gamma is None:
            raise ValueError('metric_ls requires an explicit charged rank-one metric')
        numerator = numerator + gamma * (source * u).sum(-1, keepdim=True) * (template * u).sum(-1, keepdim=True)
        denominator = denominator + gamma * (template * u).sum(-1, keepdim=True).square()
    elif mode != 'ls':
        raise ValueError('key_amplitude must be norm/ls/metric_ls')
    return numerator / denominator.clamp_min(1e-20)


def controls():
    torch.manual_seed(20261116)
    source, template = torch.zeros(5, 128), torch.zeros(5, 128)
    source[0, :2], template[0, :2] = torch.tensor([2., 1.]), torch.tensor([1., 1.])
    source[1, :2], template[1, :2] = torch.tensor([2., 1.]), torch.tensor([-1., -1.])
    source[2, :2], template[2, :2] = torch.tensor([2., 1.]), torch.tensor([1., -2.])
    source[3, 0], template[3, 0] = 1e-7, 1e-7
    template[4, 0] = 1.
    dot = (source * template).sum(-1, keepdim=True)
    norm = source.norm(dim=-1, keepdim=True)
    fallback = previous_amplitude(source, template)
    expected = torch.where(dot > 1e-12, norm.square() / dot.clamp_min(1e-12), fallback)
    actual = codec.key_amplitude(source, template, 'projection')
    if not torch.equal(actual, expected) or not torch.equal(actual[1:], fallback[1:]):
        raise RuntimeError('Projection formula or nonpositive/tiny-dot norm fallback differs')
    if not torch.allclose(actual[0] * dot[0], norm[0].square(), atol=1e-6, rtol=1e-6):
        raise RuntimeError('Positive-dot analytical projection identity failed')
    rows, rounding_sensitive = [], 0
    real_amplitude = codec.key_amplitude
    raw = torch.randn(1, 2, 19, 128)
    factor = torch.tensor([.923713, 1.07723])[None, :, None, None]
    u = torch.randn(1, 2, 1, 128)
    u = u / u.norm(dim=-1, keepdim=True)
    gamma = torch.tensor([.5, 3.])[None, :, None, None]
    for words in (16, 256):
        book = torch.randn(words, 4).half()
        for mode in ('norm', 'ls', 'metric_ls'):
            metric = dict(u=u, gamma=gamma) if mode == 'metric_ls' else {}
            actual_code, actual_gain = codec.encode_key(raw, book, amplitude=mode, gain_factor=factor, **metric)
            try:
                codec.key_amplitude = previous_amplitude
                previous_code, previous_gain = codec.encode_key(raw, book, amplitude=mode, gain_factor=factor, **metric)
            finally:
                codec.key_amplitude = real_amplitude
            if not torch.equal(actual_code, previous_code) or not torch.equal(actual_gain, previous_gain):
                raise RuntimeError('An old amplitude mode changed encoded indices or folded FP16 gains')
            rows.append(dict(book_words=words, mode=mode, indices_bitwise_unchanged=True, gain_bitwise_unchanged=True))
        code, gain = codec.encode_key(raw, book, amplitude='projection', gain_factor=factor)
        ids = codec.unpack_nibbles(code) if words == 16 else code
        selected = book.float()[ids.long()].flatten(-2)
        dot = (raw * selected).sum(-1, keepdim=True)
        norm = raw.norm(dim=-1, keepdim=True)
        proposal = torch.where(dot > 1e-12, norm.square() / dot.clamp_min(1e-12), previous_amplitude(raw, selected))
        expected = (proposal.half().float() * factor).half()
        alternative = (proposal * factor).half()
        if not torch.equal(gain, expected):
            raise RuntimeError('Projection gain did not round to FP16 before frozen factor folding')
        rounding_sensitive += int((expected != alternative).sum())
    if not rounding_sensitive:
        raise RuntimeError('Control failed to distinguish the two required FP16 rounding stages')
    tree = ast.parse(Path(codec.__file__).read_text(encoding='utf-8'))
    encode_ast = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'encode_key')
    current_ast_hash = hashlib.sha256(ast.dump(encode_ast, include_attributes=False).encode()).hexdigest()
    if current_ast_hash != PREVIOUS_ENCODE_KEY_AST_SHA256:
        raise RuntimeError('The existing encoder implementation changed beyond its amplitude helper')
    return dict(status='PASS', scope='CPU formula, fallback, rounding and legacy-mode compatibility only; no GPU/model/task result',
        formula_name='projection_preserving', provenance='reuse of existing DirectionBank projection gain; not a new formula',
        formula='norm(r)^2 / dot(r,t) for dot>1e-12, otherwise original norm gain',
        analytical_cases=5, nonpositive_or_tiny_dot_fallback_cases=4,
        legacy_modes=rows, projection_book_sizes=[16, 256],
        fp16_fold_rounding_sensitive_tokens=rounding_sensitive,
        old_encode_key_AST_unchanged=True, encode_key_AST_sha256=current_ast_hash,
        previous_codec_sha256=PREVIOUS_CODEC_SHA256,
        new_deployment_tensor_constants=0, reader_modified=False, gpu_executed=False,
        claims_excluded=['unbiasedness', 'Euclidean optimality', 'task-quality benefit'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = controls()
    report['sources'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('round7_projection_check.py', 'round7_key_codec.py', 'round7_key_bank.py', 'round7_tail_key_bank.py',
                     'round7_key_attention.py', 'round7_tail_key_attention.py', 'direction_bank.py')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(dict(status=report['status'], legacy_mode_cases=len(report['legacy_modes']),
        fp16_rounding_sensitive_tokens=report['fp16_fold_rounding_sensitive_tokens'])), flush=True)


if __name__ == '__main__':
    main()
