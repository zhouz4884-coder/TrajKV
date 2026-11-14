"""FP16 full-model calibration collector; not a low-bit serving candidate.

Queries come from actual full-cache question/decode forwards. Frozen low-bit
reconstruction errors are measured on that same query/text cache. No answers
or future evaluation queries enter the collector.
"""
import json, math
from pathlib import Path
import torch
from masked_bank import A2Bank, unique_bytes
from fast_wht import wht
from quantizers import binary_centroids
from affine_rate_bank import fold


class Round5MomentsBank(A2Bank):
    sink = None

    def _initialize(self, model, cache, visual, method, legacy_scripts,
                    kivi_source, keep_reference=False):
        super()._initialize(model, cache, visual, 'full', legacy_scripts,
                            kivi_source, False)
        cfg = type(self).settings
        device = self.states[0].visual_k.device
        book = torch.tensor(json.loads(Path(cfg['direction_book_path']).read_text())['centers'],
                            device=device, dtype=torch.float16).float()
        fit = json.loads(Path(cfg['metadata_factor_path']).read_text())
        self.fit = fit
        self.layers = []
        self.errors = []
        self.packed_keys = []
        self.local_metrics = []
        self.sample_counts = {}
        self.finalized = False
        self.output_metric_seen = set()
        self.model_modules = [x.self_attn for x in model.model.language_model.layers]
        for i, state in enumerate(self.states):
            k, v = state.visual_k.float(), state.visual_v.float()
            centered = k - k.mean(-2, keepdim=True)
            def moments(x):
                z = x[0]
                return z.sum(-2).cpu(), (z.transpose(-1, -2) @ z).cpu()
            ks, ko = moments(centered)
            vs, vo = moments(v)
            mean = k.mean(-2, keepdim=True).half()
            z = wht(k - mean.float())
            norm = z.norm(dim=-1, keepdim=True)
            unit = (norm / math.sqrt(128)).clamp_min(1e-20)
            parts = (z / unit).reshape(-1, 4)
            indices = []
            for start in range(0, len(parts), 65536):
                p = parts[start:start+65536]
                dist = p.square().sum(-1, keepdim=True) + book.square().sum(-1)[None] - 2*p @ book.T
                indices.append(dist.argmin(-1))
            template = book[torch.cat(indices)].reshape(z.shape)
            gain = (norm / template.norm(dim=-1, keepdim=True).clamp_min(1e-20)).half()
            if fit.get('key_head_factors') is not None:
                factor = torch.tensor(fit['key_head_factors'][i], device=device)[None,:,None,None]
                gain = (gain.float()*factor).half()
                if fit['key_gain_mode'] == 'whole':
                    mean = (mean.float()*factor).half()
            khat = (wht((template*gain.float()).half().float()) + mean.float()).half()
            self.packed_keys.append(((template*gain.float()).half(),mean))
            group = v.shape[-2] if cfg.get('v_group_size', 'all') == 'all' else 128
            vr = binary_centroids(state.visual_v.transpose(2,3).contiguous(), group_size=group)
            if fit.get('value_head_factors') is not None:
                vg = torch.tensor(fit['value_head_factors'][i],device=device)[None,:,None,None]
                vc = torch.tensor(fit['value_center_offsets'][i],device=device)[None,:,None,None]
                vr = fold(vr, 1, vg, vc)
            shifts = torch.arange(32, device=device)
            codes = ((vr[0].long()[...,None] >> shifts) & 1).reshape(1,v.shape[1],128,v.shape[-2]).half()
            scales = vr[1].repeat_interleave(group,-1)
            minima = vr[2].repeat_interleave(group,-1)
            vhat = ((codes*scales).half()+minima).half().transpose(2,3).contiguous()
            ek, ev = (khat.float()-k), (vhat.float()-v)
            es, eo = moments(ev)
            self.errors.append((ek, ev))
            wo = self.model_modules[i].o_proj.weight.detach().float()
            h = k.shape[1]
            grouped = wo.reshape(wo.shape[0], h, -1, 128).permute(1,2,0,3)
            # This named proxy omits attention and inter-head cross terms.
            wmetric = torch.einsum('hgod,hgoe->hde', grouped, grouped).cpu()
            self.layers.append(dict(layer=i, k_count=k.shape[-2], k_sum=ks, k_outer_sum=ko,
                v_count=v.shape[-2], v_sum=vs, v_outer_sum=vo,
                v_error_count=v.shape[-2], v_error_sum=es, v_error_outer_sum=eo,
                wo_metric=wmetric,
                k_output_metric_sum=torch.zeros_like(ko),v_output_metric_sum=torch.zeros_like(vo),output_metric_count=0,
                q_prefill_count=0, q_prefill_sum=torch.zeros_like(ks), q_prefill_outer_sum=torch.zeros_like(ko),
                q_decode_count=0, q_decode_sum=torch.zeros_like(ks), q_decode_outer_sum=torch.zeros_like(ko)))

    def read(self, q, i):
        phase = 'prefill' if 'prefill' in self.phase else 'decode'
        slot = (i, self.turn, phase)
        limit = 1 if phase == 'prefill' else type(self).settings.get('decode_samples', 8)
        count = self.sample_counts.get(slot, 0)
        if count < limit:
            self.sample_counts[slot] = count+1
            s = self.states[i]
            h = s.visual_k.shape[1]
            query = q[:,:,-1:].float()
            z = query.reshape(h, -1, q.shape[-1])
            layer = self.layers[i]
            prefix = 'q_'+phase
            layer[prefix+'_count'] += z.shape[1]
            layer[prefix+'_sum'] += z.sum(1).cpu()
            layer[prefix+'_outer_sum'] += (z.transpose(-1,-2)@z).cpu()
            self._local(query, i, phase)
        return super().read(q, i)

    def _local(self, q, i, phase):
        s = self.states[i]
        g = q.shape[1]//s.visual_k.shape[1]
        k, v = s.visual_k.float(), s.visual_v.float()
        ek, ev = self.errors[i]
        tk = s.text_k[:,:,:s.text_length].float().repeat_interleave(g,1)
        tv = s.text_v[:,:,:s.text_length].float().repeat_interleave(g,1)
        key = torch.cat((k.repeat_interleave(g,1), tk),2)
        value = torch.cat((v.repeat_interleave(g,1),tv),2)
        nv = k.shape[-2]
        scores = (q@key.transpose(-1,-2))/math.sqrt(128)
        positions = torch.cat((s.visual_pos,s.text_pos[:s.text_length]))
        scores.masked_fill_(positions[None,None,None,:] >= s.length, -torch.inf)
        weights = scores.softmax(-1)
        rotated_key,mean = self.packed_keys[i]
        packed_score = (wht(q,half=True).float() @ rotated_key.repeat_interleave(g,1).float().transpose(-1,-2))/math.sqrt(128)
        packed_score += (q*mean.repeat_interleave(g,1).float()).sum(-1,keepdim=True)/math.sqrt(128)
        delta_score = packed_score-scores[...,:nv]
        changed = scores.clone()
        changed[...,:nv] += delta_score
        weights_k = changed.softmax(-1)
        original = weights@value
        delta_k = weights_k@value-original
        delta_v = weights[...,:nv]@ev.repeat_interleave(g,1)
        delta_joint = weights_k@value + weights_k[...,:nv]@ev.repeat_interleave(g,1)-original
        wo = self.model_modules[i].o_proj.weight.detach().float()
        project = lambda x: x.transpose(1,2).reshape(1,-1) @ wo.T
        dk, dv, dj = [project(x) for x in (delta_k,delta_v,delta_joint)]
        row = dict(layer=i, phase=phase, turn=self.turn,
            score_delta_squared_sum=float(delta_score.square().sum()), score_elements=delta_score.numel(),
            k_output_error_energy=float(dk.square().sum()), v_output_error_energy=float(dv.square().sum()),
            joint_output_error_energy=float(dj.square().sum()),
            kv_linear_cross_energy=float(2*(dk*dv).sum()),
            nonlinear_remainder_energy=float((dj-dk-dv).square().sum()),
            reference_output_energy=float(project(original).square().sum()),
            visual_mass_mean=float(weights[...,:nv].sum(-1).mean()))
        h=v.shape[1]
        for name,delta in (('k',delta_k),('v',delta_v),('joint',delta_joint)):
            parts=delta.reshape(h,g*128)
            row[name+'_output_error_energy_by_kv_head']=[float((parts[j:j+1]@wo[:,j*g*128:(j+1)*g*128].T).square().sum()) for j in range(h)]
        if type(self).settings.get('measure_sign_refinement_gain',False):
            # Same-query finite interventions, one unit at a time. The current
            # baseline uses quantized K and V, while the teacher q/text is fixed.
            correction=(-ev).sign()*ev.abs().mean(-1,keepdim=True).half().float()
            refined_v=((v+ev)+correction).half().float()
            v_change=weights_k[...,:nv]@(refined_v-v-ev).repeat_interleave(g,1)
            source_rotated=wht(k-k.mean(-2,keepdim=True).half().float())
            rk=source_rotated-rotated_key.float()
            refined_key=(rotated_key.float()+rk.sign()*rk.abs().mean(-1,keepdim=True).half().float()).half()
            refined_score=(wht(q,half=True).float()@refined_key.repeat_interleave(g,1).float().transpose(-1,-2))/math.sqrt(128)
            refined_score+=(q*mean.repeat_interleave(g,1).float()).sum(-1,keepdim=True)/math.sqrt(128)
            scores_r=scores.clone();scores_r[...,:nv]=refined_score
            value_base=torch.cat(((v+ev).repeat_interleave(g,1),tv),2)
            k_change=scores_r.softmax(-1)@value_base-(original+delta_joint)
            energy=float(dj.square().sum())
            for axis,change in (('k',k_change),('v',v_change)):
                parts=change.reshape(h,g*128)
                effects=[parts[j:j+1]@wo[:,j*g*128:(j+1)*g*128].T for j in range(h)]
                row[axis+'_sign_refinement_gain_by_kv_head']=[energy-float((dj+e).square().sum()) for e in effects]
            if type(self).settings.get('interaction_gram',False):
                values_refined=torch.cat((refined_v.repeat_interleave(g,1),tv),2)
                both=scores_r.softmax(-1)@values_refined
                baseline=original+delta_joint
                # Rows h*4+s; states none, K, V, both. Text and query are fixed.
                states=[baseline,baseline+k_change,baseline+v_change,both]
                contributions=[]
                for j in range(h):
                    chunk=torch.cat([(state-original).reshape(h,g*128)[j:j+1] for state in states],0)
                    contributions.append(chunk@wo[:,j*g*128:(j+1)*g*128].T)
                matrix=torch.cat(contributions,0)
                row['interaction_gram']=(matrix@matrix.T).cpu().tolist()
                # Direct all-none/all-both controls for the combination identity.
                row['interaction_none_error_energy']=float(project(baseline-original).square().sum())
                row['interaction_both_error_energy']=float(project(both-original).square().sum())
        self.local_metrics.append(row)
        if type(self).settings.get('common_output_metrics',False) and (i,phase) not in self.output_metric_seen:
            self.output_metric_seen.add((i,phase))
            self._common_output_metrics(q,weights[...,:nv],original,v,wo,i)

    def _common_output_metrics(self,q,attention,original,v,wo,i):
        """Same-unit first-order source-error metrics, retaining GQA cross terms.

        The quadratic sum assumes zero-mean independent token errors with a
        common covariance. K's token-Jacobian sum is stratified-subsampled;
        V's attention overlap is exact. Actual joint errors remain separate.
        """
        heads=v.shape[1];group=q.shape[1]//heads;nv=v.shape[-2]
        sample=min(nv,int(type(self).settings.get('jacobian_token_samples',32)))
        selected=(torch.arange(sample,device=q.device)*nv//sample).long()
        qo=q.reshape(heads,group,128)
        av=attention.reshape(heads,group,nv)
        out=original.reshape(heads,group,128)
        for h in range(heads):
            w=wo[:,h*group*128:(h+1)*group*128].reshape(wo.shape[0],group,128).permute(1,0,2)
            overlap=av[h]@av[h].T
            eigen,u=torch.linalg.eigh((overlap+overlap.T)*.5)
            mix=u*eigen.clamp_min(0).sqrt()[None,:]
            wm=torch.einsum('gb,god->bod',mix,w).reshape(-1,128)
            mv=wm.T@wm
            differences=v[0,h,selected][None,:,:]-out[h,:,None,:]
            effects=torch.einsum('gmd,god->mgo',differences,w)
            effects=effects*av[h,:,selected].T[:,:,None]
            gram=torch.einsum('mgo,mjo->gj',effects,effects)*(nv/sample)
            mk=qo[h].T@gram@qo[h]/128
            self.layers[i]['k_output_metric_sum'][h]+=mk.cpu()
            self.layers[i]['v_output_metric_sum'][h]+=mv.cpu()
        self.layers[i]['output_metric_count']+=1

    def owned(self):
        result = super().owned()
        extra = unique_bytes([t for pair in self.errors+self.packed_keys for t in pair])
        result['collector_reconstruction_residual_bytes'] = extra
        result['total_unique'] += extra
        result['runtime'] = 'full_fp16_calibration_collector'
        return result

    def result_metadata(self):
        if not self.finalized:
            self.finalized = True
            if type(self).sink is not None:
                type(self).sink.append({'layers':self.layers, 'local_metrics':self.local_metrics})
        result = super().result_metadata()
        result.update(task_accuracy_eligible=False, backend='full_fp16_same_query_round5_collector',
            statistics_scope='full-cache queries and same-query frozen codec residuals; no labels',
            key_centering='per_image_visual_mean', collector_samples=len(self.local_metrics),
            local_metrics=self.local_metrics)
        return result
