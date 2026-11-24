"""Above-budget native-K mass/routing oracles over immutable R7 low43.

Only a clone of native visual K is retained, explicitly exposed as k_tail and
charged. Native V is never retained. These are mechanism oracles, not codecs.
"""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import torch
import r7_legacy_bridge_v1 as bridge

MODES = ('mass-only', 'route-only', 'control')
MODE_IDS = {'control': 0, 'mass-only': 1, 'route-only': 2}
RUNTIME_FILES = ('round12_oracle_bank_v1.py', 'round12_oracle_attention_v1.py')


def require(value, message):
    if not value:
        raise ValueError('ROUND12_ORACLE: ' + message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique_capacity(owners):
    objects = {}
    for owner in owners:
        for tensor in owner.named_tensors().values():
            if tensor.numel():
                objects.setdefault(bridge._storage(tensor), tensor.untyped_storage().nbytes())
    return sum(objects.values())


class OracleOwner(bridge.R7LegacyOwner):
    def named_tensors(self, include_constants=True):
        values = super().named_tensors(include_constants)
        # k_tail is classified as full-precision visual payload by ownership.py.
        values['k_tail'] = self.state.oracle_native_k
        return values

    owner_tensors = named_tensors


@torch.no_grad()
def build_round12_oracle_owners(native_prefix_cache, visual_positions, variant, *, candidate,
        legacy_a2, legacy_experiments, legacy_scripts, kivi_source,
        expected_asset_sha256=None, expected_legacy_source_sha256=None):
    mode = candidate.get('mode')
    require(mode in MODES and variant.get('name') == 'r7_joint_k1_gt_cond', 'oracle mode/original low lineage')
    owners0, metadata = bridge.build_r7_owners(native_prefix_cache, visual_positions, variant,
        legacy_a2=legacy_a2, legacy_experiments=legacy_experiments,
        legacy_scripts=legacy_scripts, kivi_source=kivi_source,
        expected_asset_sha256=expected_asset_sha256,
        expected_legacy_source_sha256=expected_legacy_source_sha256)
    selected = metadata['selected_stages']
    require(len(selected) == 43 and sum(x[0] == 'k' for x in selected) == 23 and
            sum(x[0] == 'v' for x in selected) == 20, 'original concrete K23/V20 coverage')
    n = int(visual_positions.numel())
    baseline = 4590 * n + 89733
    require(unique_capacity(owners0) == baseline, 'frozen low complete owner differs')
    shared = owners0[0].shared
    shared['oracle_format'] = torch.tensor([1, MODE_IDS[mode]], device=visual_positions.device, dtype=torch.uint8)
    original_storages = {bridge._storage(t) for layer in native_prefix_cache.layers for t in (layer.keys, layer.values)}
    result = []
    with bridge._legacy_paths(Path(legacy_a2), Path(legacy_experiments), Path(legacy_scripts)):
        reader = importlib.import_module('round12_oracle_attention_v1').attention
        for layer, owner in enumerate(owners0):
            state = owner.state
            # index_select creates a disjoint actual visual-only allocation.
            native_k = native_prefix_cache.layers[layer].keys.index_select(2, visual_positions).detach().contiguous()
            require(bridge._storage(native_k) not in original_storages and tuple(native_k.shape) == (1, 4, n, 128)
                    and native_k.dtype == torch.float16 and native_k.is_cuda, 'native K reference is not a disjoint CUDA Half visual clone')
            state.oracle_native_k = native_k
            state.oracle_mode = MODE_IDS[mode]
            result.append(OracleOwner(layer, state, shared, reader, owner.boundary))
    owners0.clear()
    native_bytes = 28 * 4 * n * 128 * 2
    expected = baseline + native_bytes + 2
    actual = unique_capacity(result)
    require(actual == expected, f'actual oracle owner {actual} != charged {expected}')
    allowed = set(bridge._STATE_TENSORS) | {'oracle_native_k'}
    for owner in result:
        require(all(not isinstance(v, torch.Tensor) or not v.numel() or k in allowed
                    for k, v in vars(owner.state).items()), 'unreported oracle state tensor')
        require(not any(hasattr(owner.state, key) for key in ('visual_k', 'visual_v')), 'full native source retained')
        for name, tensor in owner.named_tensors().items():
            require(bridge._storage(tensor) not in original_storages and not tensor.requires_grad and tensor.grad_fn is None,
                    'original full-prefix alias/autograd retained: ' + name)
    metadata.update(returned_owners_unique_tensor_capacity_bytes=actual, reader_unchanged=False,
        original_FP16_visual_storage_retained_by_owners=False,
        original_full_native_prefix_storage_retained=False,
        oracle_cloned_FP16_visual_K_reference_retained=True)
    metadata['round12_oracle'] = {
        'schema': 'trajkv.round12.oracle.v1', 'mode': mode,
        'mechanism_oracle_only': True, 'production_compression_method': False,
        'GT_or_future_answer_used': False, 'native_K_known_at_initial_image_prefix': True,
        'actual_complete_visual_owner_bytes': actual, 'original_low43_bytes': baseline,
        'native_K_reference_bytes': native_bytes, 'mode_descriptor_bytes': 2,
        'reference_is_disjoint_clone_of_original_post_RoPE_visual_K': True,
        'native_V_reference_retained': False,
        'same_current_student_query_and_text_state_for_local_LSE_pair': True,
        'packed_V_and_text_read_unchanged': True,
        'per_query_visual_partition_correction': ('logZ_F-logZ_C' if mode == 'mass-only' else
            'logZ_C-logZ_F' if mode == 'route-only' else 'none'),
        'route_scores': 'native_post_RoPE_K' if mode == 'route-only' else 'original_packed_K',
        'generated_trajectories_may_diverge_not_additive_causal_shares': True,
        'source_sha256': {name: sha(Path(__file__).with_name(name)) for name in RUNTIME_FILES},
        'CUDA_control_or_equivalence_not_established_by_factory': True,
    }
    require(all(sha(v['path']) == v['sha256'] for v in metadata['fixed_assets'].values()), 'frozen asset changed')
    return result, metadata
