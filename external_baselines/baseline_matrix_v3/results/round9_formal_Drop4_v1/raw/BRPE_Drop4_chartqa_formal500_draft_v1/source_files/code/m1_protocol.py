"""Reusable native M1 image-prefix/suffix protocol; no method or label logic."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch


@dataclass
class M1PreparedBatch:
    inputs: Any
    prefix_inputs: dict
    visual_positions: torch.Tensor
    boundary: int
    positions: torch.Tensor
    prompt: str

    @property
    def ids(self):
        return self.inputs["input_ids"]

    @property
    def mask(self):
        return self.inputs["attention_mask"]

    @property
    def full_length(self):
        return self.ids.shape[-1]

    @property
    def suffix_length(self):
        return self.full_length - self.boundary


def prepare_m1_inputs(processor, image, question, *, image_token_id=151667,
                      max_patches=8, device="cuda:0") -> M1PreparedBatch:
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt", crop_to_patches=True,
                       max_patches=max_patches).to(device)
    visual = (inputs["input_ids"][0] == image_token_id).nonzero(as_tuple=False).flatten()
    if visual.numel() < 2:
        raise ValueError("M1 requires at least two real visual placeholder tokens")
    boundary = int(visual[-1]) + 1
    positions = torch.arange(inputs["input_ids"].shape[-1], device=inputs["input_ids"].device)[None]
    prefix = {**inputs, "input_ids": inputs["input_ids"][:, :boundary],
              "attention_mask": inputs["attention_mask"][:, :boundary]}
    return M1PreparedBatch(inputs, prefix, visual, boundary, positions, prompt)


def m1_native_full_forward(model, batch: M1PreparedBatch, *, logits_to_keep=1):
    return model(**batch.inputs, position_ids=batch.positions, use_cache=True,
                 return_dict=True, logits_to_keep=logits_to_keep)


def m1_prefix_forward(model, batch: M1PreparedBatch):
    return model(**batch.prefix_inputs, position_ids=batch.positions[:, :batch.boundary],
                 use_cache=True, return_dict=True, logits_to_keep=1)


def m1_forward_suffix(model, batch: M1PreparedBatch, cache, *, logits_to_keep=1):
    return model(input_ids=batch.ids[:, batch.boundary:], attention_mask=batch.mask,
                 position_ids=batch.positions[:, batch.boundary:], past_key_values=cache,
                 use_cache=True, return_dict=True, logits_to_keep=logits_to_keep)


def m1_generate_native(model, batch: M1PreparedBatch, *, max_new_tokens=32):
    return model.generate(**batch.inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)


def m1_generate_cached(model, batch: M1PreparedBatch, cache, *, max_new_tokens=32):
    # Native generation preparation slices unprocessed question tokens according
    # to logical cache length, then extends native serial positions each step.
    # Passing a static position_ids tensor would risk reusing its final position.
    return model.generate(input_ids=batch.ids, attention_mask=batch.mask, past_key_values=cache,
                          max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)


def clone_m1_prefix_cache(cache, boundary=None):
    """Independent native DynamicCache copy for same-prefix protocol controls.

    Diagnostic/harness helper. A compression run must release prototype/copies
    before its owner audit; this function makes no compression claim.
    """
    from transformers.cache_utils import DynamicCache
    result = DynamicCache()
    for index, layer in enumerate(cache.layers):
        length = layer.keys.shape[-2] if boundary is None else boundary
        result.update(layer.keys[:, :, :length].contiguous().clone(),
                      layer.values[:, :, :length].contiguous().clone(), index)
    return result
