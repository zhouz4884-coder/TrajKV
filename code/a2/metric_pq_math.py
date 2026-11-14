"""Encoder-only query-metric assignment for the existing 256x4 PQ layout."""
import torch


def euclidean_codes(parts, book, chunk_size=65536):
    """Exact expression/chunking used by MetadataDirectionBank's assignment."""
    flat = parts.reshape(-1, 4)
    selected = []
    for start in range(0, len(flat), chunk_size):
        part = flat[start:start + chunk_size]
        distance = part.square().sum(-1, keepdim=True) + book.square().sum(-1)[None] - 2 * (part @ book.T)
        selected.append(distance.argmin(-1).byte())
    return torch.cat(selected).reshape(*parts.shape[:-1])


def metric_codes(normalized_rotated, book, metric_blocks, max_distance_elements=16777216):
    """Assign block codes at fixed initial amplitude, then caller renormalizes.

    normalized_rotated: [1, H, N, 128]; metric_blocks: [H, 32, 4, 4].
    The full-identity case uses precisely the baseline Euclidean path.  A zero
    block falls back to the identity.  Query statistics are calibration-only;
    current/future online queries never enter this function.
    """
    if normalized_rotated.ndim != 4 or normalized_rotated.shape[0] != 1 or normalized_rotated.shape[-1] != 128:
        raise ValueError('The current metric PQ layout requires batch1,D128')
    h, n = normalized_rotated.shape[1:3]
    expected = (h, 32, 4, 4)
    if tuple(metric_blocks.shape) != expected or tuple(book.shape) != (256, 4):
        raise ValueError('Metric/book architecture mismatch')
    metric = metric_blocks.float()
    if not bool(torch.isfinite(metric).all()):
        raise ValueError('Nonfinite encoding metric')
    identity = torch.eye(4, dtype=metric.dtype, device=metric.device).expand_as(metric)
    zero = metric.abs().sum((-2, -1)) <= 1e-20
    metric = torch.where(zero[..., None, None], identity, metric)
    parts = normalized_rotated.reshape(1, h, n, 32, 4)
    if torch.equal(metric, identity):
        return euclidean_codes(parts, book).reshape(1, h, n, 32).contiguous(), int(zero.sum())
    # Hxblock batched products; chunk N so distance scratch remains bounded.
    parts = parts[0].permute(0, 2, 1, 3)
    metric_book = torch.matmul(book[None, None], metric)
    book_quadratic = (metric_book * book[None, None]).sum(-1)
    tile = max(1, min(n, int(max_distance_elements) // (h * 32 * 256)))
    selected = []
    for start in range(0, n, tile):
        part = parts[:, :, start:start + tile]
        weighted = torch.matmul(part, metric)
        distance = ((weighted * part).sum(-1, keepdim=True) + book_quadratic[..., None, :]
                    - 2 * (weighted @ book.T))
        selected.append(distance.argmin(-1).byte())
    codes = torch.cat(selected, -1).permute(0, 2, 1)[None].contiguous()
    return codes, int(zero.sum())
