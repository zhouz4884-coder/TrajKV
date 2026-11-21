"""Single-owner FP16 state and bounded segmented readout correctness reference.

Not a fused kernel or packed codec. Positions are original post-RoPE positions.
The production path must replace the Python block loop, not claim its speed.
"""
import torch


class SingleState:
    def __init__(self, key, value, visual, reserve=32):
        if key.shape != value.shape or key.ndim != 4:
            raise ValueError('Expected equal B,H,N,D KV tensors')
        visual = visual.to(device=key.device, dtype=torch.long)
        if visual.ndim != 1 or visual.unique().numel() != visual.numel():
            raise ValueError('Visual positions must be unique')
        if visual.numel() and (visual.min() < 0 or visual.max() >= key.shape[2]):
            raise ValueError('Visual positions out of range')
        self.visual_pos = visual.sort().values.clone()
        mask = torch.ones(key.shape[2], device=key.device, dtype=torch.bool)
        mask[self.visual_pos] = False
        text = torch.nonzero(mask).flatten()
        self.visual_k = key.index_select(2, self.visual_pos)
        self.visual_v = value.index_select(2, self.visual_pos)
        self.reserve = reserve
        self.length = key.shape[2]
        self.text_length = text.numel()
        capacity = self.text_length + reserve
        self.text_k = key.new_empty((*key.shape[:2], capacity, key.shape[-1]))
        self.text_v = torch.empty_like(self.text_k)
        self.text_pos = torch.empty(capacity, device=key.device, dtype=torch.long)
        self.text_k[:, :, :self.text_length].copy_(key.index_select(2, text))
        self.text_v[:, :, :self.text_length].copy_(value.index_select(2, text))
        self.text_pos[:self.text_length].copy_(text)

    def append(self, key, value):
        if key.shape != value.shape or key.shape[:2] != self.text_k.shape[:2] or key.shape[-1] != self.text_k.shape[-1]:
            raise ValueError('Append shape mismatch')
        n = key.shape[2]
        end = self.text_length + n
        if end > self.text_k.shape[2]:
            capacity = max(end + self.reserve, self.text_k.shape[2] * 2)
            for name in ('text_k', 'text_v'):
                old = getattr(self, name)
                new = old.new_empty((*old.shape[:2], capacity, old.shape[-1]))
                new[:, :, :self.text_length].copy_(old[:, :, :self.text_length])
                setattr(self, name, new)
            pos = self.text_pos.new_empty(capacity)
            pos[:self.text_length].copy_(self.text_pos[:self.text_length])
            self.text_pos = pos
        self.text_k[:, :, self.text_length:end].copy_(key)
        self.text_v[:, :, self.text_length:end].copy_(value)
        self.text_pos[self.text_length:end] = torch.arange(self.length, self.length+n, device=key.device)
        self.text_length = end
        self.length += n

    def blocks(self, block_size, selected=None):
        """selected contains per-KV-head offsets into visual_pos, not global IDs."""
        if block_size <= 0:
            raise ValueError('Positive block_size required')
        count = self.visual_pos.numel() if selected is None else selected.shape[-1]
        for start in range(0, count, block_size):
            if selected is None:
                yield (self.visual_k[:, :, start:start+block_size],
                       self.visual_v[:, :, start:start+block_size], self.visual_pos[start:start+block_size])
            else:
                ids = selected[..., start:start+block_size]
                gather = ids[..., None].expand(*ids.shape, self.visual_k.shape[-1])
                yield self.visual_k.gather(2, gather), self.visual_v.gather(2, gather), self.visual_pos[ids]
        for start in range(0, self.text_length, block_size):
            end = min(start+block_size, self.text_length)
            yield self.text_k[:, :, start:end], self.text_v[:, :, start:end], self.text_pos[start:end]

    def storage_bytes(self):
        tensors = [self.visual_k, self.visual_v, self.visual_pos, self.text_k, self.text_v, self.text_pos]
        storage = {(str(t.device), t.untyped_storage().data_ptr()): t.untyped_storage().nbytes() for t in tensors}
        return sum(storage.values())


def readout(query, query_pos, blocks):
    """Stable online softmax; handles a fully future-masked intermediate block."""
    shape = (*query.shape[:-1], 1)
    m = torch.full(shape, -torch.inf, device=query.device, dtype=torch.float32)
    denominator = torch.zeros_like(m)
    numerator = torch.zeros_like(query, dtype=torch.float32)
    for key, value, positions in blocks:
        groups = query.shape[1] // key.shape[1]
        if groups * key.shape[1] != query.shape[1]:
            raise ValueError('Invalid GQA head ratio')
        k = key.float().repeat_interleave(groups, 1)
        v = value.float().repeat_interleave(groups, 1)
        if positions.ndim == 1:
            positions = positions[None, None, :].expand(key.shape[0], key.shape[1], -1)
        positions = positions.repeat_interleave(groups, 1)
        scores = query.float() @ k.transpose(-1, -2) * query.shape[-1] ** -.5
        scores.masked_fill_(positions[:, :, None, :] > query_pos[None, None, :, None], -torch.inf)
        block_m = scores.amax(-1, keepdim=True)
        new_m = torch.maximum(m, block_m)
        safe_m = torch.where(torch.isfinite(new_m), new_m, 0.)
        scale = torch.exp(m-safe_m)
        weights = torch.exp(scores-safe_m)
        numerator = numerator * scale + weights @ v
        denominator = denominator * scale + weights.sum(-1, keepdim=True)
        m = new_m
    return (numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny)).to(query.dtype)
