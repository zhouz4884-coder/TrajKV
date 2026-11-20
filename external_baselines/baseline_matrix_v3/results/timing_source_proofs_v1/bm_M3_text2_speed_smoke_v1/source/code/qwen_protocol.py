"""Native Qwen2.5-VL/Qwen3-VL prefix and suffix protocol.

Native processors, RoPE/QK norm, vision and DeepStack remain active. Cached
generation delegates to HF with the complete prompt history and saved native
RoPE delta, while image tensors are consumed only by the image prefix.
"""
from __future__ import annotations
from dataclasses import dataclass
import inspect
from typing import Any
import torch


@dataclass
class QwenPreparedBatch:
    inputs: Any
    prefix_inputs: dict
    visual_positions: torch.Tensor
    boundary: int
    positions: torch.Tensor
    rope_deltas: torch.Tensor
    prompt: str

    @property
    def ids(self): return self.inputs['input_ids']

    @property
    def mask(self): return self.inputs['attention_mask']

    @property
    def full_length(self): return self.ids.shape[-1]

    @property
    def suffix_length(self): return self.full_length-self.boundary


def _native_rope(model, inputs):
    signature=inspect.signature(model.model.get_rope_index).parameters
    kwargs={k:v for k,v in inputs.items() if k in signature}
    return model.model.get_rope_index(**kwargs)


def prepare_qwen_inputs(processor, image, question, *, model, image_token_id=None,
                        device='cuda:0', min_pixels=None, max_pixels=None) -> QwenPreparedBatch:
    if model.config.model_type not in ('qwen2_5_vl','qwen3_vl'):
        raise ValueError('UNSUPPORTED_MODEL: native Qwen2.5-VL or Qwen3-VL required')
    messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':question}]}]
    prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    kwargs={}
    if min_pixels is not None:kwargs['min_pixels']=int(min_pixels)
    if max_pixels is not None:kwargs['max_pixels']=int(max_pixels)
    inputs=processor(text=[prompt],images=[image],return_tensors='pt',**kwargs).to(device)
    if 'mm_token_type_ids' not in inputs:
        raise ValueError('INPUT_MISMATCH: native processor modality IDs are required for mRoPE')
    if inputs['input_ids'].shape[0]!=1 or not bool((inputs['attention_mask']==1).all()):
        raise ValueError('UNSUPPORTED_CACHE_PROTOCOL: only unpadded batch-one image requests')
    token=model.config.image_token_id if image_token_id is None else image_token_id
    visual=(inputs['input_ids'][0]==token).nonzero(as_tuple=False).flatten()
    if visual.numel()<2:raise ValueError('INPUT_MISMATCH: no native image placeholders')
    boundary=int(visual[-1])+1
    if boundary>=inputs['input_ids'].shape[-1]:raise ValueError('INPUT_MISMATCH: missing question suffix')
    positions,deltas=_native_rope(model,inputs)
    if positions.shape!=(3,1,inputs['input_ids'].shape[-1]):
        raise ValueError('UNSUPPORTED_MODEL: unexpected native three-axis position shape')
    prefix={}
    sequence_fields={'input_ids','attention_mask','mm_token_type_ids','token_type_ids'}
    for key,value in inputs.items():
        if key=='position_ids':raise ValueError('UNSUPPORTED_MODEL: processor-supplied positions require explicit audit')
        prefix[key]=value[..., :boundary] if key in sequence_fields else value
    prefix_positions,prefix_deltas=_native_rope(model,prefix)
    if not torch.equal(prefix_positions,positions[..., :boundary]) or not torch.equal(prefix_deltas,deltas):
        raise ValueError('INPUT_MISMATCH: image-prefix native mRoPE differs from full prompt prefix')
    return QwenPreparedBatch(inputs,prefix,visual,boundary,positions,deltas,prompt)


def _activate_native_delta(model,batch):
    model.model.rope_deltas=batch.rope_deltas.detach().to(batch.ids.device).clone()


def qwen_native_full_forward(model,batch:QwenPreparedBatch,*,logits_to_keep=1):
    _activate_native_delta(model,batch)
    return model(**batch.inputs,position_ids=batch.positions.to(batch.ids.device),use_cache=True,
                 return_dict=True,logits_to_keep=logits_to_keep)


def qwen_prefix_forward(model,batch:QwenPreparedBatch):
    _activate_native_delta(model,batch)
    return model(**batch.prefix_inputs,position_ids=batch.positions[..., :batch.boundary].to(batch.ids.device),
                 use_cache=True,return_dict=True,logits_to_keep=1)


def qwen_forward_suffix(model,batch:QwenPreparedBatch,cache,*,logits_to_keep=1):
    if cache.get_seq_length()!=batch.boundary:
        raise ValueError('INPUT_MISMATCH: suffix must start from the original image-prefix boundary')
    _activate_native_delta(model,batch)
    return model(input_ids=batch.ids[:, batch.boundary:],attention_mask=batch.mask,
                 position_ids=batch.positions[..., batch.boundary:].to(batch.ids.device),past_key_values=cache,
                 use_cache=True,return_dict=True,logits_to_keep=logits_to_keep)


def qwen_generate_native(model,batch:QwenPreparedBatch,*,max_new_tokens=32,**generation_kwargs):
    # No past cache: native HF preparation computes the original image positions.
    return model.generate(**batch.inputs,max_new_tokens=max_new_tokens,do_sample=False,
                          use_cache=True,**generation_kwargs)


def qwen_generate_cached(model,batch:QwenPreparedBatch,cache,*,max_new_tokens=32,**generation_kwargs):
    if cache.get_seq_length()!=batch.boundary:
        raise ValueError('INPUT_MISMATCH: cached generation must recompute the entire question suffix')
    _activate_native_delta(model,batch)
    # Native HF preparation slices full IDs at the LOGICAL cache length and
    # derives each later text position from the original delta. Full IDs retain
    # prompt history for M2's native repetition penalty 1.05. No consumed image
    # tensor or stale first-answer logits are supplied.
    return model.generate(input_ids=batch.ids,attention_mask=batch.mask,
                          past_key_values=cache,max_new_tokens=max_new_tokens,
                          do_sample=False,use_cache=True,**generation_kwargs)


# Neutral aliases let a runner choose one protocol module per model.
native_full_forward=qwen_native_full_forward
prefix_forward=qwen_prefix_forward
forward_suffix=qwen_forward_suffix
generate_native=qwen_generate_native
generate_cached=qwen_generate_cached
