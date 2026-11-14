"""Numerical helpers for the round-5 query-weighted *quality* experiment.

All matrices use row-vector conventions: y = (k-mu) @ encode and
k_hat = y_hat @ decode + mu.  Gaussian reverse-water-filling is only an
idealized weighted-squared-error reference, not a prediction of ANLS or the
distortion attained by a finite 256x4 product code.
"""
import torch


def symmetric(x):
    return (x + x.transpose(-1, -2)) * .5


def _count_tensor(count, values):
    c = torch.as_tensor(count, dtype=values.dtype, device=values.device)
    while c.ndim < values.ndim:
        c = c.unsqueeze(-1)
    return c.clamp_min(1.)


def moment(sum_x, sum_outer, count, centered=False):
    """Moment from sufficient statistics; q uses centered=False.

    Per-image-centered K sufficient statistics should already have the image
    mean removed by the collector.  Subtracting its residual global mean is
    harmless and protects against accumulating floating-point offsets.
    """
    sx = torch.as_tensor(sum_x).double()
    xx = torch.as_tensor(sum_outer).double()
    mean = sx / _count_tensor(count, sx)
    value = xx / _count_tensor(count, xx)
    if centered:
        value = value - mean.unsqueeze(-1) * mean.unsqueeze(-2)
    return mean, symmetric(value)


def source_roots(covariance, ridge=1e-4):
    covariance = symmetric(covariance.double())
    values, vectors = torch.linalg.eigh(covariance)
    floor = values.amax(-1, keepdim=True).clamp_min(1e-12) * ridge
    regularized = values.clamp_min(floor)
    root = (vectors * regularized.sqrt().unsqueeze(-2)) @ vectors.transpose(-1, -2)
    invroot = (vectors * regularized.rsqrt().unsqueeze(-2)) @ vectors.transpose(-1, -2)
    return root, invroot, values, regularized


def weighted_basis(covariance, query_second_moment, ridge=1e-4):
    """Biorthogonal source-white/query-weighted bases in original coordinates.

    If encode=A and decode=B, full-rank A@B=I.  Rank r takes A[..., :r]
    and B[..., :r, :], giving the original-coordinate oblique projector.
    """
    root, invroot, raw, regularized = source_roots(covariance, ridge)
    weighted = symmetric(root @ symmetric(query_second_moment.double()) @ root)
    values, vectors = torch.linalg.eigh(weighted)
    values = values.flip(-1).clamp_min(0.)
    vectors = vectors.flip(-1)
    encode = invroot @ vectors
    decode = vectors.transpose(-1, -2) @ root
    return encode, decode, values, raw, regularized


def pca_basis(covariance):
    values, vectors = torch.linalg.eigh(symmetric(covariance.double()))
    return vectors.flip(-1), vectors.flip(-1).transpose(-1, -2), values.flip(-1).clamp_min(0.)


def gaussian_waterfill(eigenvalues, bits_per_source_scalar):
    """Gaussian weighted MSE reference, per-source payload bits (not KV b_eff).

    Supports any leading dimensions.  Each row has D source coordinates; the
    rate is sum(b_i)/D, including coordinates assigned zero bits.  Degenerate
    all-zero metrics have zero absolute distortion and an undefined normalized
    distortion, represented as zero with an explicit zero_metric flag.
    """
    if bits_per_source_scalar < 0:
        raise ValueError('A nonnegative payload rate is required')
    mu = torch.as_tensor(eigenvalues).double().clamp_min(0.)
    d = mu.shape[-1]
    maximum = mu.amax(-1, keepdim=True)
    positive = mu > 0
    log_mu = torch.where(positive, mu.clamp_min(1e-300).log2(), -torch.inf)
    high = maximum.clamp_min(1e-300).log2()
    low = high - (2 * d * float(bits_per_source_scalar) + 80)
    for _ in range(100):
        middle = (low + high) * .5
        rate = ((log_mu - middle) * .5).clamp_min(0.).sum(-1, keepdim=True) / d
        # More rate than requested: raise theta.
        low = torch.where(rate > bits_per_source_scalar, middle, low)
        high = torch.where(rate > bits_per_source_scalar, high, middle)
    log_theta = (low + high) * .5
    allocation = ((log_mu - log_theta) * .5).clamp_min(0.)
    allocation = torch.where(positive, allocation, torch.zeros_like(allocation))
    theta = log_theta.exp2()
    distortion = torch.minimum(mu, theta).sum(-1)
    total = mu.sum(-1)
    return {'theta': theta.squeeze(-1), 'allocation_bits': allocation,
            'rate': allocation.mean(-1), 'absolute_distortion': distortion,
            'relative_distortion': distortion / total.clamp_min(1e-300),
            'zero_metric': total == 0}


def weighted_projection_error(covariance, query_moment, encode, decode, rank):
    """Exact second-moment error under the independent-q/source quadratic model."""
    d = covariance.shape[-1]
    projector = encode[..., :rank] @ decode[..., :rank, :]
    error = torch.eye(d, dtype=projector.dtype, device=projector.device) - projector
    residual_covariance = error.transpose(-1, -2) @ covariance @ error
    numerator = (residual_covariance * query_moment.transpose(-1, -2)).sum((-2, -1))
    denominator = (covariance * query_moment.transpose(-1, -2)).sum((-2, -1))
    return numerator.clamp_min(0.) / denominator.clamp_min(1e-300)


def hadamard_matrix(dimension, dtype=torch.float32, device='cpu'):
    if dimension < 1 or dimension & (dimension - 1):
        raise ValueError('Hadamard dimension must be a power of two')
    value = torch.ones((1, 1), dtype=dtype, device=device)
    while value.shape[0] < dimension:
        value = torch.cat((torch.cat((value, value), 1), torch.cat((value, -value), 1)), 0)
    return value / dimension ** .5
