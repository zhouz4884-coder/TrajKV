"""Fixed Full-q/text local counterfactuals, returning only the Full output.

Every candidate sees the same actual Full query and text cache. Candidate
outputs are measured and discarded; they never enter the model's residual
stream or cache. These are local mechanism statistics, not task scores or a
low-bit deployment implementation.
"""
import hashlib
import json
import math
from pathlib import Path

import torch

from affine_rate_bank import fold
from fast_wht import wht
from masked_bank import A2Bank, unique_bytes
from quantizers import binary_centroids


class FixedQueryBank(A2Bank):
    sink = None

    def _initialize(self, model, cache, visual, method, legacy_scripts,
                    kivi_source, keep_reference=False):
        A2Bank._initialize(self, model, cache, visual, 'full', legacy_scripts,
                           kivi_source, False)
        cfg = type(self).settings
        device = self.states[0].visual_k.device
        self.entries = []
        self.local_metrics = []
        self.sample_counts = {}
        self.finalized = False
        self.modules = [layer.self_attn for layer in model.model.language_model.layers]
        book_path = Path(cfg['direction_book_path'])
        book = torch.tensor(json.loads(book_path.read_text())['centers'],
                            device=device, dtype=torch.float16).float()
        if book.shape != (256, 4):
            raise ValueError('Fixed-query PQ control requires the frozen 256x4 book')
        fit_path = Path(cfg['metadata_factor_path'])
        fit = json.loads(fit_path.read_text())
        if fit['encoder_book_sha256'] != hashlib.sha256(book_path.read_bytes()).hexdigest():
            raise ValueError('Frozen metadata book mismatch')
        if fit['value_group_size'] != cfg.get('v_group_size', 'all'):
            raise ValueError('Frozen metadata value-window mismatch')
        mode = fit['key_gain_mode']
        if mode not in ('none', 'centered', 'whole'):
            raise ValueError('Unrecognized frozen K gain mode')
        self.key_gain_mode = mode
        artifact_path = Path(cfg['spectrum_artifact_path'])
        artifact = torch.load(artifact_path, map_location='cpu', weights_only=False)
        if artifact.get('format') != 'trajkv_query_spectrum_v1':
            raise ValueError('Unrecognized spectrum artifact')
        name = cfg.get('query_transform', 'query_aware_mixed')
        basis = artifact['transforms'][name]
        self.artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        self.sampled_events = []
        self.phase_scope = 'Full free-running first question, final question position and first decode calls'
        for i, state in enumerate(self.states):
            source = state.visual_k.float()
            mean = source.mean(-2, keepdim=True).half().contiguous()
            residual = source-mean.float()
            factor = (torch.tensor(fit['key_head_factors'][i], device=device,
                                  dtype=torch.float32)[None, :, None, None]
                      if fit.get('key_head_factors') is not None else None)
            final_mean = ((mean.float()*factor).half() if factor is not None and mode == 'whole' else mean)
            rotated = wht(residual)
            initial = (rotated.norm(dim=-1, keepdim=True)/math.sqrt(128)).clamp_min(1e-20)
            parts = (rotated/initial).reshape(-1, 4)
            indices = []
            for start in range(0, len(parts), 65536):
                x = parts[start:start+65536]
                dist = x.square().sum(-1, keepdim=True)+book.square().sum(-1)[None]-2*(x@book.T)
                indices.append(dist.argmin(-1))
            template = book[torch.cat(indices)].reshape(rotated.shape)
            # Mirror the materialized port's declared original-space norm rule.
            unscaled = wht(template)
            gain = (residual.norm(dim=-1, keepdim=True)/unscaled.norm(dim=-1, keepdim=True).clamp_min(1e-20)).half()
            if factor is not None:
                gain = (gain.float()*factor).half()
            pq = (wht((template*gain.float()).half().float())+final_mean.float()).half()
            candidates = {'pq_fwht': pq}
            encode = basis['encode'][i].to(device=device, dtype=torch.float32)[None]
            decode = basis['decode'][i].to(device=device, dtype=torch.float32)[None]
            if encode.shape[-2:] != (128, 128) or decode.shape[-2:] != (128, 128):
                raise ValueError('Basis must match the D128 model')
            for rank in (16, 32, 64):
                latent = (residual@encode[..., :rank]).half()
                projected = latent.float()@decode[..., :rank, :]
                scaled = projected if factor is None else projected*factor
                candidates[f'query_r{rank}'] = (scaled+final_mean.float()).half().contiguous()
                if rank in (16, 32):
                    norm_gain = (residual.norm(dim=-1, keepdim=True)/projected.norm(dim=-1, keepdim=True).clamp_min(1e-20)).half()
                    restored = projected*norm_gain.float()
                    if factor is not None:
                        restored = restored*factor
                    candidates[f'query_r{rank}_norm'] = (restored+final_mean.float()).half().contiguous()
            values = state.visual_v.transpose(2, 3).contiguous()
            group = visual.numel() if cfg.get('v_group_size', 'all') == 'all' else 128
            record = binary_centroids(values, group_size=group)
            if fit.get('value_head_factors') is not None:
                vg = torch.tensor(fit['value_head_factors'][i], device=device, dtype=torch.float32)[None, :, None, None]
                vo = torch.tensor(fit['value_center_offsets'][i], device=device, dtype=torch.float32)[None, :, None, None]
                record = fold(record, 1, vg, vo)
            shift = torch.arange(32, device=device)
            codes = ((record[0].long()[..., None] >> shift) & 1).reshape(values.shape).half()
            decoded = ((codes*record[1].repeat_interleave(group, -1)).half()
                       +record[2].repeat_interleave(group, -1)).half().transpose(2, 3).contiguous()
            if not all(bool(torch.isfinite(x).all()) for x in list(candidates.values())+[decoded]):
                raise RuntimeError('Nonfinite fixed-query reconstruction (including norm restoration)')
            self.entries.append({'keys': candidates, 'value1': decoded})

    def read(self, q, layer):
        phase = 'question_prefill' if self.phase == 'question_prefill' else self.phase
        if self.turn == 1 and phase in ('question_prefill', 'decode'):
            key = (layer, phase)
            count = self.sample_counts.get(key, 0)
            limit = 1 if phase == 'question_prefill' else int(type(self).settings.get('decode_samples', 4))
            if count < limit:
                self.sample_counts[key] = count+1
                self._measure(q[:, :, -1:].float(), layer, phase, count)
        # The only output committed to the residual stream is the real Full path.
        return A2Bank.read(self, q, layer)

    def _measure(self, q, layer, phase, step):
        state = self.states[layer]
        entry = self.entries[layer]
        group = q.shape[1]//state.visual_k.shape[1]
        original_k = state.visual_k.float().repeat_interleave(group, 1)
        original_v = state.visual_v.float().repeat_interleave(group, 1)
        quantized_v = entry['value1'].float().repeat_interleave(group, 1)
        text_k = state.text_k[:, :, :state.text_length].float().repeat_interleave(group, 1)
        text_v = state.text_v[:, :, :state.text_length].float().repeat_interleave(group, 1)
        nv = original_k.shape[-2]
        visual_scores = (q@original_k.transpose(-1, -2))/math.sqrt(128)
        text_scores = (q@text_k.transpose(-1, -2))/math.sqrt(128)
        positions = torch.cat((state.visual_pos, state.text_pos[:state.text_length]))
        invalid = positions[None, None, None, :] >= state.length
        scores = torch.cat((visual_scores, text_scores), -1).masked_fill(invalid, -torch.inf)
        lp = scores.log_softmax(-1)
        weights = lp.exp()
        baseline_values = torch.cat((original_v, text_v), 2)
        value1_values = torch.cat((quantized_v, text_v), 2)
        baseline = weights@baseline_values
        baseline_v1 = weights@value1_values
        mass = weights[..., :nv].sum(-1)
        wo = self.modules[layer].o_proj.weight.detach().float()
        project = lambda x: x.transpose(1, 2).reshape(1, -1)@wo.T
        reference_y = project(baseline)
        reference_v1_y = project(baseline_v1)
        visual_centered = visual_scores-visual_scores.mean(-1, keepdim=True)
        reference_score_sse = float(visual_scores.square().sum())
        reference_score_centered_sse = float(visual_centered.square().sum())
        keys = {'full_k': state.visual_k, **entry['keys']}
        for candidate, packed_k in keys.items():
            current_k = packed_k.float().repeat_interleave(group, 1)
            new_visual_scores = (q@current_k.transpose(-1, -2))/math.sqrt(128)
            error = new_visual_scores-visual_scores
            mean_error = error.mean(-1, keepdim=True)
            centered_error = error-mean_error
            changed = torch.cat((new_visual_scores, text_scores), -1).masked_fill(invalid, -torch.inf)
            new_lp = changed.log_softmax(-1)
            new_weights = new_lp.exp()
            kl_terms = torch.where(weights > 0, weights*(lp-new_lp), torch.zeros_like(weights))
            alpha = new_weights[..., :nv].sum(-1)
            common = dict(layer=layer, phase=phase, step=step, turn=self.turn,
                candidate=candidate, head_count=q.shape[1], visual_tokens=nv,
                score_elements=error.numel(), score_error_sse=float(error.square().sum()),
                score_centered_error_sse=float(centered_error.square().sum()),
                score_offset_error_sse=float(mean_error.square().sum()*nv),
                score_reference_sse=reference_score_sse,
                score_reference_centered_sse=reference_score_centered_sse,
                attention_kl_sum=float(kl_terms.sum()),
                visual_mass_error_sse=float((alpha-mass).square().sum()),
                visual_mass_abs_error_sum=float((alpha-mass).abs().sum()),
                visual_mass_reference_sum=float(mass.sum()), visual_mass_candidate_sum=float(alpha.sum()),
                top1_same_fraction=float((new_visual_scores.argmax(-1)==visual_scores.argmax(-1)).float().mean()),
                reference_output_sse=float(reference_y.square().sum()), output_elements=reference_y.numel(),
                scope='same Full q/text, all heads combined through W_O; no candidate output committed')
            output16 = project(new_weights@baseline_values)
            output1 = project(new_weights@value1_values)
            joint_cross = output1-output16-(reference_v1_y-reference_y)
            for name, output, same_v_reference in [('v16', output16, reference_y),
                                                    ('v1', output1, reference_v1_y)]:
                self.local_metrics.append(dict(common, value_representation=name,
                    output_error_sse=float((output-reference_y).square().sum()),
                    k_effect_output_sse=float((output-same_v_reference).square().sum()),
                    v_only_output_sse=float((reference_v1_y-reference_y).square().sum()),
                    kv_interaction_output_sse=float(joint_cross.square().sum())))

    def owned(self):
        result = super().owned()
        tensors = [x for entry in self.entries for x in list(entry['keys'].values())+[entry['value1']]]
        added = unique_bytes(tensors)
        result['fixed_query_candidate_reference_bytes'] = added
        result['total_unique'] += added
        result['runtime'] = 'full_fp16_fixed_query_local_counterfactual_collector'
        return result

    def result_metadata(self):
        if not self.finalized:
            self.finalized = True
            if type(self).sink is not None:
                type(self).sink.append({'local_metrics': self.local_metrics,
                                       'sample_counts': dict(self.sample_counts)})
        result = super().result_metadata()
        result.update(task_accuracy_eligible=False,
            backend='full_fp16_fixed_query_counterfactuals',
            fixed_query_metric_rows=len(self.local_metrics),
            fixed_query_scope=self.phase_scope, candidate_outputs_committed=False,
            fixed_query_key_gain_mode=self.key_gain_mode,
            fixed_query_spectrum_sha256=self.artifact_sha256,
            norm_restore_scope='FP16 per-token residual-norm gain before the same frozen K head factor',
            metric_scope='FP32 dense local scores/output on shared Full states; not deployment reader arithmetic or end-to-end candidate answers')
        return result
