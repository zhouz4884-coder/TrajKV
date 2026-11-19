"""Source-balanced native-coordinate dictionary calibration for Round8.

Fixed assignments/gains/residuals define weighted least-squares centroid
proposals. Every proposal is quantized to FP16, decoded in the declared hard
order, and accepted only on the source-balanced aggregate training objective.
This does not assert a global optimum or predict downstream model quality.
Select sources are scored only after fitting, never used for updates/acceptance.
The module imports on CPU without Transformers, Triton, or a model.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def cartesian_book(old):
    if tuple(old.shape) != (16, 4):
        raise ValueError('Cartesian initialization requires the frozen 16x4 book')
    return torch.stack([torch.cat((old[i], old[j])) for j in range(16) for i in range(16)]).half().contiguous()


def binary_cube(device='cpu'):
    ids = torch.arange(256, device=device)
    shifts = torch.arange(8, device=device)
    return ((ids[:, None] >> shifts) & 1).half().contiguous()


def binary_indices(binary):
    parts = binary.reshape(*binary.shape[:-1], 16, 8).long()
    shifts = torch.arange(8, device=binary.device)
    return (parts << shifts).sum(-1).byte().contiguous()


def key4_to_key8(index):
    return (index[..., 0::2] | (index[..., 1::2] << 4)).contiguous()


def unpack_key(code, book):
    words = book.shape[-2]
    if words == 16:
        return torch.stack((code & 15, code >> 4), -1).flatten(-2).contiguous()
    return code


def signs(code):
    shifts = torch.arange(32, device=code.device)
    return (((code.long()[..., None] >> shifts) & 1).flatten(-2).float() * 2 - 1)


def hard_residual(base, codes, alphas):
    value = base
    for stage in range(codes.shape[1]):
        value = (value.float() + signs(codes[:, stage]) * alphas[:, stage, None].float()).half()
    return value


def residual_offset(codes, alphas):
    result = torch.zeros((len(codes), 128), device=codes.device)
    for stage in range(codes.shape[1]):
        result += signs(codes[:, stage]) * alphas[:, stage, None].float()
    return result


@dataclass
class NativeData:
    source: torch.Tensor
    old_base: torch.Tensor
    ids4: torch.Tensor
    binary: torch.Tensor
    gain: torch.Tensor
    factor: torch.Tensor
    minimum: torch.Tensor
    scale: torch.Tensor
    residual_codes: torch.Tensor
    residual_alphas: torch.Tensor
    depth: torch.Tensor
    weight: torch.Tensor
    source_index: torch.Tensor
    source_ids: list


@dataclass
class Fields:
    index: torch.Tensor
    gain: torch.Tensor
    codes: torch.Tensor
    alphas: torch.Tensor


def load_native(dump, source_ids, axis, device, manifest):
    """Each source has total weight 1/S, independent of sampled token count."""
    expected = list(map(str, source_ids))
    if not expected or len(set(expected)) != len(expected):
        raise ValueError('Nonempty, unique source membership required')
    entries = {str(x['source_id']): x for x in manifest['entries']}
    if len(entries) != len(manifest['entries']):
        raise ValueError('Duplicate sources in native dump')
    if not set(expected) <= entries.keys():
        raise ValueError('Split has sources missing from native dump: ' + str(sorted(set(expected)-entries.keys())))
    all_fields = {key: [] for key in ('source', 'old_base', 'ids4', 'binary', 'gain', 'factor', 'minimum', 'scale',
        'residual_codes', 'residual_alphas', 'depth', 'weight', 'source_index')}
    for number, identity in enumerate(expected):
        entry = entries[identity]
        path = dump / entry['file']
        if sha(path) != entry['sha256']:
            raise ValueError('Native shard SHA mismatch: ' + entry['file'])
        shard = torch.load(path, map_location='cpu', weights_only=False)
        if shard['schema'] != 'trajkv.round8.native-dump.v1' or shard['source_id'] != identity:
            raise ValueError('Native shard identity/schema mismatch')
        if not shard.get('diagnostic_source_owner') or shard.get('deployable') is not False:
            raise ValueError('Source tensors must be explicitly diagnostic owners')
        if shard['selected_stages'] != manifest['selected_stages']:
            raise ValueError('Frozen stage mask changed between source shards')
        count = sum(len(u[axis + '_source']) for u in shard['units'])
        if count <= 0:
            raise ValueError('Empty source')
        for unit in shard['units']:
            n = len(unit[axis + '_source'])
            source = unit[axis + '_source']
            if source.shape != (n, 128) or not bool(torch.isfinite(source).all()):
                raise ValueError('Finite native D128 source required')
            values = dict(source=source.float(), old_base=unit[axis+'_old_base'].half(),
                ids4=unit['k_index4'].byte(), binary=unit['v_binary'].byte(),
                gain=unit['k_gain'].half(), factor=unit['k_factor'].float().reshape(1, 1).expand(n, 1),
                minimum=unit['v_min'].half().reshape(1, 128).expand(n, 128),
                scale=unit['v_scale'].half().reshape(1, 128).expand(n, 128),
                depth=torch.full((n,), len(unit[axis + '_residual_codes']), dtype=torch.long),
                weight=torch.full((n,), 1. / (len(expected) * count), dtype=torch.float64),
                source_index=torch.full((n,), number, dtype=torch.long))
            codes = torch.zeros((n, 2, 4), dtype=torch.int32)
            alphas = torch.zeros((n, 2), dtype=torch.float16)
            for stage, (code, alpha) in enumerate(zip(unit[axis+'_residual_codes'], unit[axis+'_residual_alphas'])):
                codes[:, stage] = code
                alphas[:, stage] = alpha
            values.update(residual_codes=codes, residual_alphas=alphas)
            for key, value in values.items():
                all_fields[key].append(value)
        del shard
    merged = {key: torch.cat(value).contiguous().to(device) for key, value in all_fields.items()}
    return NativeData(**merged, source_ids=expected)


def initial_fields(data, axis, book):
    if axis == 'k':
        index = key4_to_key8(data.ids4) if book.shape[-1] == 8 else data.ids4.clone()
    else:
        index = binary_indices(data.binary)
    return Fields(index, data.gain.clone(), data.residual_codes.clone(), data.residual_alphas.clone())


def template(book, index):
    if book.ndim == 2:
        return book.float()[index.long()].flatten(-2)
    position = torch.arange(book.shape[0], device=book.device)
    return book.float()[position[None], index.long()].flatten(-2)


def decode_base(data, axis, book, fields, sl):
    t = template(book, fields.index[sl])
    if axis == 'k':
        return (t * fields.gain[sl].float()).half()
    multiplied = (data.scale[sl].float() * t).half()
    return (data.minimum[sl].float() + multiplied.float()).half()


@torch.no_grad()
def hard_scores(data, axis, book, fields, chunk=2048):
    result = torch.zeros(len(data.source_ids), device=data.source.device, dtype=torch.float64)
    for start in range(0, len(data.source), chunk):
        sl = slice(start, start+chunk)
        decoded = hard_residual(decode_base(data, axis, book, fields, sl), fields.codes[sl], fields.alphas[sl])
        sse = (data.source[sl] - decoded.float()).square().sum(-1).double()
        # Report per-source mean squared error per scalar, then mean sources.
        result.index_add_(0, data.source_index[sl], sse * data.weight[sl] * len(data.source_ids) / 128)
    per_source = result.cpu().tolist()
    return dict(source_balanced_mse=sum(per_source) / len(per_source),
                per_source_mse=dict(zip(data.source_ids, per_source)))


@torch.no_grad()
def centroid_proposal(data, axis, book, fields, chunk=2048):
    """Weighted normal equations in native coordinates, not normalized LBG.

    K: sum(w*g*(z-r))/sum(w*g^2).
    V: sum(w*S*(v-m-r))/sum(w*S^2), per coordinate.
    FP16 sequential rounding is assessed after this real-valued proposal.
    """
    dim, words = book.shape[-1], book.shape[-2]
    positions = 128 // dim
    tables = book.shape[0] if book.ndim == 3 else 1
    numerator = torch.zeros((tables*words, dim), device=book.device, dtype=torch.float64)
    denominator = torch.zeros_like(numerator)
    for start in range(0, len(data.source), chunk):
        sl = slice(start, start+chunk)
        target = data.source[sl] - residual_offset(fields.codes[sl], fields.alphas[sl])
        coefficient = fields.gain[sl].expand(-1, 128).float() if axis == 'k' else data.scale[sl].float()
        if axis == 'v':
            target -= data.minimum[sl].float()
        weight = data.weight[sl, None, None]
        coefficient, target = coefficient.reshape(-1, positions, dim).double(), target.reshape(-1, positions, dim).double()
        index = fields.index[sl].long()
        if tables > 1:
            index = index + torch.arange(positions, device=book.device)[None] * words
        numerator.index_add_(0, index.flatten(), (weight * coefficient * target).reshape(-1, dim))
        denominator.index_add_(0, index.flatten(), (weight * coefficient.square()).reshape(-1, dim))
    previous = book.double().reshape(-1, dim)
    candidate = torch.where(denominator > 0, numerator / denominator.clamp_min(1e-300), previous)
    if not bool(torch.isfinite(candidate).all()):
        raise ValueError('Nonfinite centroid proposal')
    return candidate.reshape(book.shape).half().contiguous()


@torch.no_grad()
def reencode_fields(data, axis, book, chunk=2048):
    """Rebuild own J/g/R fields with the actual shared Round8 codec.

    K uses norm gain and its original functional factor with both FP16 gain
    roundings. V has frozen all-image m/S and no row gain. Every selected sign
    stage is regenerated against the candidate's actual FP16 base.
    """
    from round8_codec import encode_key, encode_value_joint8, encode_sign_depth
    indexes, gains, codes, alphas = [], [], [], []
    for start in range(0, len(data.source), chunk):
        sl = slice(start, start+chunk)
        if axis == 'k':
            packed, gain = encode_key(data.source[sl], book, amplitude='norm',
                                      gain_factor=data.factor[sl], chunk=chunk)
            index = unpack_key(packed, book)
            base = (template(book, index) * gain.float()).half()
        else:
            # Metadata may differ by row (source and layer/head). Treat each
            # row as a one-token batch/head, keeping the exact codec layout.
            source = data.source[sl].half()[None, :, None, :]
            minimum = data.minimum[sl][None, :, :, None]
            scale = data.scale[sl][None, :, :, None]
            index = encode_value_joint8(source, scale, minimum, book, chunk=chunk)[0, :, 0]
            gain = data.gain[sl]
            multiplied = (data.scale[sl].float() * template(book, index)).half()
            base = (data.minimum[sl].float() + multiplied.float()).half()
        rc = torch.zeros((len(base), 2, 4), device=base.device, dtype=torch.int32)
        ra = torch.zeros((len(base), 2), device=base.device, dtype=torch.float16)
        for depth in (1, 2):
            selected = data.depth[sl] == depth
            if bool(selected.any()):
                local_c, local_a, _, _ = encode_sign_depth(data.source[sl][selected], base[selected], depth)
                for stage in range(depth):
                    rc[selected, stage], ra[selected, stage] = local_c[stage], local_a[stage]
        indexes.append(index)
        gains.append(gain)
        codes.append(rc)
        alphas.append(ra)
    return Fields(*(torch.cat(x).contiguous() for x in (indexes, gains, codes, alphas)))


@torch.no_grad()
def fit_book(data, axis, initial, *, mode='decoder-only', iterations=8, chunk=2048):
    if mode not in ('decoder-only', 'reencode'):
        raise ValueError('Mode must declare decoder-only or reencode')
    book = initial.half().contiguous().clone()
    fields = initial_fields(data, axis, book) if mode == 'decoder-only' else reencode_fields(data, axis, book, chunk)
    before = hard_scores(data, axis, book, fields, chunk)
    logs = [dict(iteration=0, accepted=True, **before)]
    for iteration in range(1, iterations+1):
        candidate = centroid_proposal(data, axis, book, fields, chunk)
        # Evaluate several finite candidates along the LS direction. Rounding
        # and sequential residuals mean the real LS point need not improve.
        best_book, best_fields, best = book, fields, before
        accepted_fraction = 0.
        candidates = []
        for fraction in (1., .5, .25):
            trial = (book.float() + fraction * (candidate.float()-book.float())).half().contiguous()
            trial_fields = fields if mode == 'decoder-only' else reencode_fields(data, axis, trial, chunk)
            score = hard_scores(data, axis, trial, trial_fields, chunk)
            candidates.append(dict(fraction=fraction, source_balanced_mse=score['source_balanced_mse']))
            if score['source_balanced_mse'] < best['source_balanced_mse']:
                best_book, best_fields, best, accepted_fraction = trial, trial_fields, score, fraction
        logs.append(dict(iteration=iteration, accepted=accepted_fraction > 0,
                         fraction=accepted_fraction, candidates=candidates, **best))
        book, fields, before = best_book, best_fields, best
        print(json.dumps(dict(axis=axis, mode=mode, iteration=iteration,
            book_shape=list(book.shape), accepted=accepted_fraction > 0,
            source_balanced_mse=best['source_balanced_mse'])), flush=True)
        if accepted_fraction == 0.:
            break
    return book, fields, logs


def evaluate_book(data, axis, book, mode, chunk):
    fields = initial_fields(data, axis, book) if mode == 'decoder-only' else reencode_fields(data, axis, book, chunk)
    return hard_scores(data, axis, book, fields, chunk)


def write_book(path, book, metadata):
    centers = book.detach().half().cpu().contiguous()
    words, dim = centers.shape[-2:]
    value = dict(schema='trajkv.direction-book.v1', dimension=dim, centroids=words,
        centers=centers.float().tolist(), storage_dtype='float16',
        nominal_bits_per_scalar=(words.bit_length()-1)/dim,
        actual_book_bytes=centers.numel()*2, centers_fp16_le_sha256=tensor_sha(centers),
        table_layout='position-specific' if centers.ndim == 3 else 'global', **metadata)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')
    return dict(file=path.name, sha256=sha(path), centers_fp16_le_sha256=value['centers_fp16_le_sha256'],
                actual_book_bytes=value['actual_book_bytes'], shape=list(centers.shape))


def numerical_check():
    """CPU algebra check; no native/model/GPU samples or LBG are generated."""
    scale = torch.tensor([[1., 10.], [10., 1.]], dtype=torch.float64)
    normal = torch.tensor([[1., 1.], [2., 2.]], dtype=torch.float64)
    weighted = (scale.square()*normal).sum(0)/scale.square().sum(0)
    unweighted = normal.mean(0)
    weighted_sse = float((scale*(normal-weighted)).square().sum())
    unweighted_sse = float((scale*(normal-unweighted)).square().sum())
    grad = (scale.square()*(weighted-normal)).sum(0)
    # Nonuniform source cardinalities still have exactly equal source mass.
    source_masses = [sum([1/(2*n)]*n) for n in (3, 11)]
    g = torch.tensor([.5, 2., 1.], dtype=torch.float64)
    z = torch.tensor([[1., 4.], [3., -1.], [2., 7.]], dtype=torch.float64)
    w = torch.tensor([.5, .25, .25], dtype=torch.float64)
    center = (w[:, None]*g[:, None]*z).sum(0)/(w*g.square()).sum()
    k_normal_residual = (w[:, None]*g[:, None]*(g[:, None]*center-z)).sum(0)
    old = torch.arange(64, dtype=torch.float32).reshape(16, 4).div(37).sub(.8).half()
    ids = torch.arange(32, dtype=torch.uint8).remainder(16).reshape(1, 32)
    joined = key4_to_key8(ids)
    gain = torch.tensor([[.79]], dtype=torch.float16)
    k4 = (template(old, ids)*gain.float()).half()
    k8 = (template(cartesian_book(old), joined)*gain.float()).half()
    pos = (template(old[None].repeat(32, 1, 1), ids)*gain.float()).half()
    bit = torch.arange(128).remainder(3).eq(0).byte().reshape(1, 128)
    vc = template(binary_cube(), binary_indices(bit))
    assert weighted_sse < unweighted_sse and float(grad.abs().max()) < 1e-12
    assert max(abs(x-.5) for x in source_masses) < 1e-12
    assert float(k_normal_residual.abs().max()) < 1e-12
    assert torch.equal(k4, k8) and torch.equal(k4, pos) and torch.equal(vc, bit.float())
    return dict(status='PASS', scope='CPU algebra only; not a model, native codec or quality measurement',
        v_weighted_center=weighted.tolist(), v_weighted_sse=weighted_sse,
        v_unweighted_sse=unweighted_sse, v_normal_equation_max_residual=float(grad.abs().max()),
        k_normal_equation_max_residual=float(k_normal_residual.abs().max()), source_masses=source_masses,
        toy_fp16_cartesian_key_identity=True, toy_fp16_position_key_identity=True,
        toy_binary_cube_layout_identity=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dump', type=Path)
    p.add_argument('--splits', type=Path)
    p.add_argument('--old-book', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--iterations', type=int, default=8)
    p.add_argument('--chunk', type=int, default=2048)
    p.add_argument('--modes', nargs='+', choices=('decoder-only', 'reencode'), default=['decoder-only', 'reencode'])
    p.add_argument('--assets', nargs='+', choices=('k_global4_16', 'k_global8_256', 'k_position4_16', 'v_global8_256'),
                   default=['k_global4_16', 'k_global8_256', 'k_position4_16', 'v_global8_256'])
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    if args.check_only:
        (args.output/'numerical_check.json').write_text(json.dumps(numerical_check(), indent=2), encoding='utf-8')
        return
    if not all((args.dump, args.splits, args.old_book)) or args.iterations < 0 or args.chunk <= 0:
        p.error('dump, splits, old-book, nonnegative iterations and positive chunk are required')
    torch.set_num_threads(8)
    if args.device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
    splits = json.loads(args.splits.read_text(encoding='utf-8-sig'))
    train, select = list(map(str, splits['train'])), list(map(str, splits['select']))
    if set(train) & set(select):
        raise ValueError('Source leakage: train and select overlap')
    manifest = json.loads((args.dump/'manifest.json').read_text())
    if manifest['schema'] != 'trajkv.round8.native-dump.v1' or manifest['old_book_sha256'] != sha(args.old_book):
        raise ValueError('Native dump schema/frozen encoder identity mismatch')
    old = torch.tensor(json.loads(args.old_book.read_text())['centers'], device=args.device, dtype=torch.float16)
    if tuple(old.shape) != (16, 4):
        raise ValueError('Round8 baseline is the shipped K1 G4_16 book')
    provenance = dict(calibration_schema='trajkv.round8.source-balanced-native-fit.v1',
        old_encoder_book_sha256=sha(args.old_book), dump_manifest_sha256=sha(args.dump/'manifest.json'),
        source_split_sha256=sha(args.splits), training_source_ids=train, select_source_ids=select,
        labels_used=False, model_weights_frozen=True, source_script_sha256=sha(__file__),
        objective='mean over sources of mean native-coordinate squared error per scalar; exact declared FP16 decode',
        source_weight='each source has total mass 1/number_sources, divided equally among sampled native layer/head/token vectors',
        key_frame='FP16 all-image mean before actual WHT; gain is final folded old gain for decoder-only',
        value_frame='frozen all-image per-channel FP16 min/scale including functional fold; no V row gain',
        fixed_stage_mask=manifest['selected_stages'],
        conditional_update='source-weighted native-coordinate LS proposal; FP16 candidates accepted only on training aggregate hard loss',
        select_policy='read and score once after each endpoint is fitted; never used for training, stopping or acceptance',
        scope='coordinate reconstruction calibration; actual packed model quality and efficiency measured separately')
    outputs = []
    for axis in ('k', 'v'):
        assets = [x for x in args.assets if x.startswith(axis+'_')]
        if not assets:
            continue
        data = load_native(args.dump, train, axis, args.device, manifest)
        for asset in assets:
            initial = {'k_global4_16': lambda: old.clone(),
                'k_global8_256': lambda: cartesian_book(old),
                'k_position4_16': lambda: old[None].repeat(32, 1, 1).contiguous(),
                'v_global8_256': lambda: binary_cube(args.device)}[asset]()
            identity_fields = initial_fields(data, axis, initial)
            for start in range(0, len(data.source), args.chunk):
                sl = slice(start, start+args.chunk)
                reconstructed = decode_base(data, axis, initial, identity_fields, sl)
                if not torch.equal(reconstructed, data.old_base[sl]):
                    raise ValueError('Native initializer hard-base identity failed before fitting: '+asset)
            for mode in args.modes:
                book, _, logs = fit_book(data, axis, initial, mode=mode, iterations=args.iterations, chunk=args.chunk)
                name = asset+'_'+mode.replace('-', '_')
                # Load no select tensors until the final centers are frozen.
                selection = load_native(args.dump, select, axis, args.device, manifest)
                selected_score = evaluate_book(selection, axis, book, mode, args.chunk)
                del selection
                fields_policy = ('original J/g/frame and all residual codes/alphas fixed; decoder table only' if mode == 'decoder-only' else
                    'own J, K norm gain with original functional factor, own residual codes/alphas; same frozen mean/V frame/stage mask')
                metadata = dict(provenance, asset=asset, axis=axis, fit_mode=mode,
                    fields_policy=fields_policy, initialization_fp16_le_sha256=tensor_sha(initial),
                    initialization=('Cartesian C8[i+16j]=concat(C4[i],C4[j])' if asset == 'k_global8_256' else
                        'binary cube, low-coordinate bit first' if axis == 'v' else 'frozen G4_16 replicated only for position control'),
                    train_history=logs, select_score=selected_score,
                    retains_old_encoder_book=mode == 'decoder-only' and axis == 'k',
                    old_encoder_constant_bytes=128 if mode == 'decoder-only' and axis == 'k' else 0)
                entry = write_book(args.output/(name+'.json'), book, metadata)
                entry.update(asset=asset, mode=mode, final_train_mse=logs[-1]['source_balanced_mse'],
                    select_mse=selected_score['source_balanced_mse'])
                outputs.append(entry)
                (args.output/(name+'_fit_log.json')).write_text(json.dumps(dict(train=logs, select=selected_score), indent=2), encoding='utf-8')
        del data
    (args.output/'complete_books.json').write_text(json.dumps(dict(status='COMPLETE', schema=provenance['calibration_schema'],
        books=outputs, provenance=provenance, numerical_check=numerical_check()), indent=2), encoding='utf-8')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
