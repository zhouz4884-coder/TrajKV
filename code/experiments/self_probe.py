"""Online middle-layer residual probe controller; attention stays in the caller.

The caller reads the SAME current query twice at ``layer``: base and
base+stored-residual. Their displacement is a correction signal, not measured
FP16 error. This module never reads a reference cache, future query or answer.

Typical bank integration::

    controller.reset_event(turn, phase, input_start, input_count)
    # Earlier layers: ordinary base reads, with caller-owned cost accounting.
    decision = controller.at_probe(base_y, enhanced_y, {
        'layer': layer, 'same_query': True,
        'probe_read_bytes': actual_two_read_bytes,
        'upgrade_bytes': extra_residual_bytes_for_remaining_layers,
        'cost_scope': 'logical visual payload',
    })
    y = enhanced_y if decision['use_enhanced_probe'] else base_y
    # Later layers: controller.should_refine(layer_idx).
    # Report their actual charges via record_charge(...), or maintain the
    # authoritative ledger in the bank when controller budget_bytes is None.

``budget_bytes`` is a cumulative caller-accounted budget for this controller's
lifetime. reset_event resets the decision, not the budget. at_probe charges a
provided probe_read_bytes once, unless probe_cost_already_charged=True. It does
not charge estimated upgrade bytes; the caller reports actual later reads.
The probe already happened when at_probe is called, so even a rejected upgrade
cannot erase its two-read cost or synchronization.
"""
from __future__ import annotations

import copy
import math
import time
from numbers import Integral, Real

import torch


def _nonnegative_integer(value, name):
    if isinstance(value,bool) or not isinstance(value,Integral) or value<0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return int(value)


def displacement_features(base_output, enhanced_output, epsilon=1e-12):
    """Device-side displacement norms with one scalar-vector host transfer.

    FP16/BF16 inputs accumulate in FP32. FP64 inputs keep FP64. Separate scaled
    norms avoid squaring large finite outputs directly. The scalar transfer
    synchronizes CUDA; feature_wall_ms includes that wait, and may include
    previously queued attention work. The caller must additionally time/charge
    both attention reads and the complete probe, including this synchronization.
    """
    if not isinstance(base_output,torch.Tensor) or not isinstance(enhanced_output,torch.Tensor):
        raise TypeError('Both outputs must be tensors')
    if base_output.shape!=enhanced_output.shape or base_output.numel()==0:
        raise ValueError('Probe outputs must have the same nonempty shape')
    if base_output.device!=enhanced_output.device:
        raise ValueError('Probe outputs must be on the same device')
    if not base_output.is_floating_point() or not enhanced_output.is_floating_point():
        raise TypeError('Probe outputs must be floating point tensors')
    if not isinstance(epsilon,Real) or not math.isfinite(float(epsilon)) or epsilon<=0:
        raise ValueError('epsilon must be positive and finite')
    started=time.perf_counter()
    dtype=torch.float64 if torch.float64 in (base_output.dtype,enhanced_output.dtype) else torch.float32
    with torch.no_grad():
        base=base_output.detach().to(dtype)
        enhanced=enhanced_output.detach().to(dtype)
        difference=enhanced-base
        tiny=torch.finfo(dtype).tiny
        base_scale=base.abs().amax().clamp_min(tiny)
        delta_scale=difference.abs().amax().clamp_min(tiny)
        base_squared=(base/base_scale).square().sum()
        delta_squared=(difference/delta_scale).square().sum()
        finite=(torch.isfinite(base).all() & torch.isfinite(enhanced).all()
                & torch.isfinite(difference).all())
        stats=torch.stack((base_scale,delta_scale,base_squared,delta_squared,finite.to(dtype))).cpu().tolist()
    base_scale,delta_scale,base_squared,delta_squared,finite=stats
    valid=bool(finite) and all(math.isfinite(value) for value in stats)
    count=base_output.numel()
    if valid:
        base_l2=base_scale*math.sqrt(max(base_squared,0.))
        delta_l2=delta_scale*math.sqrt(max(delta_squared,0.))
        denominator=max(base_l2,float(epsilon))
        relative_l2=delta_l2/denominator
        rms=delta_l2/math.sqrt(count)
        base_rms=base_l2/math.sqrt(count)
        valid=all(math.isfinite(value) for value in (base_l2,delta_l2,denominator,relative_l2,rms,base_rms))
    if not valid:
        base_l2=delta_l2=denominator=relative_l2=rms=base_rms=None
    return dict(relative_l2=relative_l2,rms=rms,delta_l2=delta_l2,base_l2=base_l2,
                base_rms=base_rms,relative_denominator=denominator,epsilon=float(epsilon),
                base_norm_below_epsilon=(base_l2<float(epsilon) if base_l2 is not None else None),
                finite_signal=valid,elements=count,shape=list(base_output.shape),
                device=str(base_output.device),input_dtype=str(base_output.dtype),
                enhanced_dtype=str(enhanced_output.dtype),accumulation_dtype=str(dtype),
                feature_wall_ms_including_synchronization=1000*(time.perf_counter()-started),
                scalar_transfer_synchronizes_cuda=bool(base_output.is_cuda),
                timing_scope='feature computation and scalar synchronization, possibly waiting for queued attention; not full two-read probe time',
                feature_availability='online: same current query and stored base/residual only',
                signal_meaning='base+residual output displacement from base; not actual FP16/reference error')


class ProbeController:
    """One middle-layer threshold decision per forward event.

    at_probe returns a JSON-serializable dict. Its ``upgrade`` and
    ``use_enhanced_probe`` flags are identical; only layers strictly AFTER the
    probe layer are enabled by should_refine(). A later reset_event always
    clears that activation. No policy update uses future outcomes.
    """

    def __init__(self, layer: int, threshold: float, budget_bytes=None, *, epsilon=1e-12):
        self.layer=_nonnegative_integer(layer,'layer')
        if isinstance(threshold,bool) or not isinstance(threshold,Real) or not math.isfinite(float(threshold)) or threshold<0:
            raise ValueError('threshold must be finite and nonnegative')
        if not isinstance(epsilon,Real) or not math.isfinite(float(epsilon)) or epsilon<=0:
            raise ValueError('epsilon must be positive and finite')
        self.threshold=float(threshold)
        self.epsilon=float(epsilon)
        self.budget_bytes=None if budget_bytes is None else _nonnegative_integer(budget_bytes,'budget_bytes')
        self.spent_bytes=0
        self.events=[]
        self.current=None
        self._upgrade=False
        self._charge_ids=set()

    @property
    def remaining_bytes(self):
        return None if self.budget_bytes is None else self.budget_bytes-self.spent_bytes

    def reset_event(self, turn, phase, input_start, input_count):
        turn=_nonnegative_integer(turn,'turn')
        input_start=_nonnegative_integer(input_start,'input_start')
        input_count=_nonnegative_integer(input_count,'input_count')
        if input_count==0:
            raise ValueError('An event must contain at least one current input token')
        if not isinstance(phase,str) or not phase:
            raise ValueError('phase must be a nonempty string')
        self._upgrade=False
        self._charge_ids=set()
        self.current=dict(event_id=len(self.events),turn=turn,phase=phase,
                          input_start=input_start,input_count=input_count,
                          probe_layer=self.layer,threshold=self.threshold,
                          spent_bytes_at_start=self.spent_bytes,budget_bytes=self.budget_bytes,
                          charges=[],charged_bytes=0,probe_observed=False,decision=None,
                          feature_availability='online: current query and stored residual',
                          budget_scope='cumulative bytes explicitly reported by caller; no inference for unreported reads')
        self.events.append(self.current)
        return copy.deepcopy(self.current)

    def can_charge(self, amount):
        amount=_nonnegative_integer(amount,'amount')
        return self.budget_bytes is None or self.spent_bytes+amount<=self.budget_bytes

    def record_charge(self, amount, kind='read', *, charge_id=None, metadata=None):
        """Record actual incurred bytes, even if an unavoidable read overran.

        can_charge() is the caller's pre-read check. record_charge() must never
        hide already incurred work. Optional per-event charge_id rejects double
        bookkeeping instead of silently counting the same probe twice.
        """
        if self.current is None:
            raise RuntimeError('reset_event must precede cost accounting')
        amount=_nonnegative_integer(amount,'amount')
        if not isinstance(kind,str) or not kind:
            raise ValueError('kind must be a nonempty string')
        if charge_id is not None:
            if charge_id in self._charge_ids:
                raise ValueError('Duplicate charge_id in the same event')
            self._charge_ids.add(charge_id)
        self.spent_bytes+=amount
        self.current['charged_bytes']+=amount
        entry=dict(kind=kind,bytes=amount,cumulative_spent_bytes=self.spent_bytes,
                   remaining_bytes=self.remaining_bytes,charge_id=charge_id)
        if metadata:
            entry['metadata']={str(k):v for k,v in metadata.items()
                               if isinstance(v,(str,int,float,bool)) or v is None}
        self.current['charges'].append(entry)
        self.current['spent_bytes_now']=self.spent_bytes
        self.current['budget_overrun_bytes']=max(0,-self.remaining_bytes) if self.remaining_bytes is not None else 0
        if self.remaining_bytes is not None and self.remaining_bytes<0:
            self._upgrade=False
            self.current['upgrade_stopped_after_budget_overrun']=True
        return copy.deepcopy(entry)

    def at_probe(self, base_output, enhanced_output, metadata=None):
        if self.current is None:
            raise RuntimeError('reset_event must precede a probe')
        if self.current['probe_observed']:
            raise RuntimeError('Only one probe decision is allowed in each event')
        metadata={} if metadata is None else dict(metadata)
        if 'layer' in metadata and metadata['layer']!=self.layer:
            raise ValueError('Probe callback occurred at the wrong layer')
        if metadata.get('same_query') is False:
            raise ValueError('Probe reads must use the identical current query')
        if metadata.get('uses_fp16_reference') or metadata.get('uses_future_information'):
            raise ValueError('Reference/future information is forbidden in an online residual probe')
        probe_bytes=metadata.get('probe_read_bytes')
        if 'base_read_bytes' in metadata and 'enhanced_read_bytes' in metadata:
            computed=(_nonnegative_integer(metadata['base_read_bytes'],'base_read_bytes')+
                      _nonnegative_integer(metadata['enhanced_read_bytes'],'enhanced_read_bytes'))
            if probe_bytes is not None and computed!=probe_bytes:
                raise ValueError('Two-read probe bytes do not equal the reported component sum')
            probe_bytes=computed
        if probe_bytes is not None:
            probe_bytes=_nonnegative_integer(probe_bytes,'probe_read_bytes')
        planned=metadata.get('upgrade_bytes')
        if planned is not None:
            planned=_nonnegative_integer(planned,'upgrade_bytes')
        if probe_bytes is not None and not metadata.get('probe_cost_already_charged',False):
            self.record_charge(probe_bytes,'probe_two_reads',charge_id='probe_two_reads')
        features=displacement_features(base_output,enhanced_output,self.epsilon)
        above=bool(features['finite_signal'] and features['relative_l2']>=self.threshold)
        if not features['finite_signal']:
            upgrade=False;reason='nonfinite_or_overflowed_signal'
        elif not above:
            upgrade=False;reason='below_relative_l2_threshold'
        elif self.budget_bytes is not None and (probe_bytes is None or planned is None):
            upgrade=False;reason='budget_cost_unknown'
        elif planned is not None and not self.can_charge(planned):
            upgrade=False;reason='insufficient_budget_after_probe'
        else:
            upgrade=True;reason='threshold_met_and_budget_available'
        self._upgrade=upgrade
        decision=dict(upgrade=upgrade,use_enhanced_probe=upgrade,
                      refine_after_layer=self.layer,refine_start_layer=self.layer+1,
                      threshold=self.threshold,threshold_comparison='relative_l2 >= threshold',
                      threshold_met=above,reason=reason,**features,
                      probe_read_bytes=probe_bytes,planned_upgrade_bytes=planned,
                      spent_bytes=self.spent_bytes,remaining_bytes=self.remaining_bytes,
                      budget_bytes=self.budget_bytes,
                      probe_cost_already_incurred=True,
                      upgrade_cost_is_estimate_only=True,
                      caller_must_record_actual_later_read_bytes=True,
                      probe_attention_and_synchronization_costs_must_be_included=True)
        self.current['probe_observed']=True
        self.current['decision']=decision
        allowed={'layer','query_tokens','cost_scope','same_query','policy',
                 'probe_read_bytes','base_read_bytes','enhanced_read_bytes','upgrade_bytes',
                 'probe_cost_already_charged','residuals_stored'}
        self.current['probe_metadata']={k:v for k,v in metadata.items() if k in allowed
                                        and (isinstance(v,(str,int,float,bool)) or v is None)}
        return copy.deepcopy(decision)

    def should_refine(self, layer_idx):
        layer_idx=_nonnegative_integer(layer_idx,'layer_idx')
        return bool(self._upgrade and layer_idx>self.layer)

    def event_rows(self):
        return copy.deepcopy(self.events)

    def summary(self):
        observed=[e for e in self.events if e['probe_observed']]
        return dict(probe_layer=self.layer,threshold=self.threshold,budget_bytes=self.budget_bytes,
                    spent_bytes=self.spent_bytes,remaining_bytes=self.remaining_bytes,
                    events=len(self.events),probes=len(observed),
                    upgrades=sum(e['decision']['upgrade'] for e in observed),
                    feature_availability='online: current query and stored residual only',
                    signal_meaning='correction magnitude, not actual FP16 error',
                    cost_scope='only caller-reported actual byte charges; no implicit attention traffic or latency estimate')
