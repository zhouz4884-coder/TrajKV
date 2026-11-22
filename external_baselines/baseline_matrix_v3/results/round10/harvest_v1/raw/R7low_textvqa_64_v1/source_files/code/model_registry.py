"""Locked native model identities and cache contracts for matrix v3.

Geometry is verified against upstream HF configuration, not inferred from the
parameter-count label. Local paths are supplied by a job, never implicit caches.
"""
from __future__ import annotations
from copy import deepcopy

MODEL_REGISTRY = {
    'M1': {
        'model_id': 'M1', 'hf_id': 'OpenGVLab/InternVL3-8B-hf',
        'architecture': 'InternVLForConditionalGeneration',
        'layers': 28, 'query_heads': 28, 'kv_heads': 4, 'head_dim': 128,
        'image_token_id': 151667, 'deepstack_visual_indexes': [],
        'sliding_attention': False,
        'native_adapter': 'internvl3',
        'revision': '259a3b64a14623c0ec91a045cb43f7c5af5fa6af',
        'revision_status': 'EXACT_REQUIRED_TREE_VERIFIED_LFS_AWARE',
    },
    'M2': {
        'model_id': 'M2', 'hf_id': 'Qwen/Qwen2.5-VL-7B-Instruct',
        'revision': 'cc594898137f460bfe9f0759e9844b3ce807cfb5',
        'architecture': 'Qwen2_5_VLForConditionalGeneration',
        'layers': 28, 'query_heads': 28, 'kv_heads': 4, 'head_dim': 128,
        'image_token_id': 151655, 'vision_start_token_id': 151652,
        'vision_end_token_id': 151653, 'native_adapter': 'qwen2_5_vl',
        'mrope_section': [16, 24, 24], 'mrope_interleaved': False,
        'deepstack_visual_indexes': [], 'sliding_attention': False,
        'processor_spatial_factor': 28,
    },
    'M3': {
        'model_id': 'M3', 'hf_id': 'Qwen/Qwen3-VL-4B-Instruct',
        'revision': 'ebb281ec70b05090aa6165b016eac8ec08e71b17',
        'architecture': 'Qwen3VLForConditionalGeneration',
        'layers': 36, 'query_heads': 32, 'kv_heads': 8, 'head_dim': 128,
        'image_token_id': 151655, 'vision_start_token_id': 151652,
        'vision_end_token_id': 151653, 'native_adapter': 'qwen3_vl',
        'mrope_section': [24, 20, 20], 'mrope_interleaved': True,
        'deepstack_visual_indexes': [5, 11, 17], 'sliding_attention': False,
        'processor_spatial_factor': 32,
    },
}


def get_model_spec(model_id: str) -> dict:
    return deepcopy(MODEL_REGISTRY[model_id])


def cache_layers(cache):
    """Yield native post-RoPE (K,V), including every DynamicCache layer."""
    if hasattr(cache, 'layers'):
        for layer in cache.layers:
            yield layer.keys, layer.values
    elif hasattr(cache, 'key_cache'):
        yield from zip(cache.key_cache, cache.value_cache)
    else:
        yield from cache


def qwen_language_model(model):
    return model.model.language_model


def qwen_text_layers(model):
    return qwen_language_model(model).layers


def qwen_visual_positions(input_ids, spec):
    """Indices into the ORIGINAL native prompt; no position compaction."""
    return (input_ids[0] == spec['image_token_id']).nonzero(as_tuple=False).flatten()
