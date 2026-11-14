"""V2 fidelity probes, forked without editing the frozen mechanism.py v1.

The canonical transcript is generated once with the packed 2-bit visual base.
No reference answer enters this process. Ref/base visual reads independently
re-encode that identical transcript; future reads cross the two legal histories.
All-ref repeat and base free/forced replay provide empirical numerical nulls.
The complete original-FP16-visual history is teacher-forced over the same
canonical token sequence before action probes. Primary action utility is
KL(p_base || p_reference) - KL(p_action || p_reference), not an increase in
the likelihood of a possibly incorrect base-generated answer. Target-token
likelihood remains an auxiliary diagnostic. Fidelity is NOT answer accuracy.
The full reference trace is an offline calibration artifact, never an online
policy feature. Online policies may consume only a table frozen on development
sessions. Bank and generation runtime interfaces remain unchanged.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

from bank import ProgressiveBank, install_legacy
from run_cycle import consume, digest, features


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _snapshot_position(snapshot):
    values = {int(layer['length']) for layer in snapshot}
    if len(values) != 1:
        raise ValueError('Layers disagree on logical sequence length')
    return values.pop()


def _set_read(bank, mode, phase, step=0, action=None):
    bank.phase = phase
    bank.reference_layers = set(range(len(bank.states))) if mode == 'ref' else set()
    bank.refine_layers = set()
    if action is not None:
        group, selected_phase = action
        active = ((selected_phase == 'text_prefill' and phase == 'text_prefill')
                  or (selected_phase == 'early' and phase == 'decode' and step < 2)
                  or (selected_phase == 'late' and phase == 'decode' and step >= 2))
        if active:
            n = len(bank.states)
            bank.refine_layers = set(range(group*n//4, (group+1)*n//4))


def snapshot_difference(left, right, minimum_position=0):
    """Compare text state at identical original positions, excluding padding."""
    if len(left) != len(right):
        raise ValueError('Different number of layers')
    totals = {axis: dict(squared_error=0., squared_reference=0.,
                         max_abs=0., elements=0) for axis in ('k', 'v')}
    per_layer = []
    for layer, (a, b) in enumerate(zip(left, right)):
        if (a['length'], a['text_length']) != (b['length'], b['text_length']):
            raise ValueError('Intervention changed token count or logical position')
        if not torch.equal(a['values'][2], b['values'][2]):
            raise ValueError('Intervention changed text position IDs')
        selected = a['values'][2] >= minimum_position
        detail = {'layer': layer, 'selected_text_positions': int(selected.sum())}
        for offset, axis in enumerate(('k', 'v')):
            x = a['values'][offset][:, :, selected].float()
            y = b['values'][offset][:, :, selected].float()
            difference = x-y
            squared = float(difference.square().sum(dtype=torch.float64))
            reference = float(y.square().sum(dtype=torch.float64))
            largest = float(difference.abs().max()) if difference.numel() else 0.
            detail[axis] = dict(relative_l2=math.sqrt(squared/max(reference, 1e-300)),
                                rms=math.sqrt(squared/max(x.numel(), 1)), max_abs=largest)
            total = totals[axis]
            total['squared_error'] += squared
            total['squared_reference'] += reference
            total['max_abs'] = max(total['max_abs'], largest)
            total['elements'] += x.numel()
        per_layer.append(detail)
    for total in totals.values():
        total['relative_l2'] = math.sqrt(total['squared_error']/max(total['squared_reference'], 1e-300))
        total['rms'] = math.sqrt(total['squared_error']/max(total['elements'], 1))
    return dict(aggregate=totals, per_layer=per_layer,
                minimum_position=minimum_position, token_positions_equal=True)


def logit_comparison(actual, reference, targets, eos_ids):
    """Full-vocabulary KL and fixed-target likelihood, with CPU float64 sums."""
    if len(actual) != len(reference) or len(actual) != len(targets):
        raise ValueError('Logit trajectories must score the same target tokens')
    rows = []
    for index, (logits, baseline, target) in enumerate(zip(actual, reference, targets)):
        logits, baseline = logits.reshape(-1).double(), baseline.reshape(-1).double()
        logp, logq = logits.log_softmax(-1), baseline.log_softmax(-1)
        difference = logits-baseline
        rows.append(dict(index=index, target_token_id=int(target), is_eos=target in eos_ids,
                         target_logprob=float(logp[target]), reference_target_logprob=float(logq[target]),
                         target_logprob_gain=float(logp[target]-logq[target]),
                         kl_reference_to_actual=float((logq.exp()*(logq-logp)).sum()),
                         max_abs_logit_difference=float(difference.abs().max()),
                         rms_logit_difference=float(difference.square().mean().sqrt()),
                         argmax_token_id=int(logits.argmax()), reference_argmax_token_id=int(baseline.argmax()),
                         argmax_changed=bool(logits.argmax() != baseline.argmax())))
    content = [r for r in rows if not r['is_eos']]
    mean = lambda key, chosen: sum(r[key] for r in chosen)/max(len(chosen), 1)
    return dict(per_token=rows, target_logprob_sum=sum(r['target_logprob'] for r in rows),
                reference_target_logprob_sum=sum(r['reference_target_logprob'] for r in rows),
                target_logprob_gain_sum=sum(r['target_logprob_gain'] for r in rows),
                content_logprob_gain_mean=mean('target_logprob_gain', content),
                mean_kl_reference_to_actual=mean('kl_reference_to_actual', rows),
                max_abs_logit_difference=max((r['max_abs_logit_difference'] for r in rows), default=0.),
                argmax_changes=sum(r['argmax_changed'] for r in rows),
                token_count=len(rows), content_token_count=len(content))


def logit_fidelity_gain(base, actual, reference, targets, eos_ids):
    """Distribution fidelity on identical targets/positions, in CPU float64.

    Forward KL means D_KL(p_candidate || p_allref); positive gain means the
    intervention reduced that divergence. Reverse-KL gains are retained as a
    sensitivity diagnostic, not silently mixed with the primary objective.
    """
    if not (len(base) == len(actual) == len(reference) == len(targets)):
        raise ValueError('Fidelity trajectories must share token IDs and positions')
    rows = []
    for index, (low, action, ref, target) in enumerate(zip(base, actual, reference, targets)):
        low, action, ref = (t.detach().reshape(-1).double().cpu() for t in (low, action, ref))
        if not (low.shape == action.shape == ref.shape):
            raise ValueError('Fidelity logits must share the full vocabulary')
        low_logp, action_logp, ref_logp = (t.log_softmax(-1) for t in (low, action, ref))
        base_kl = float((low_logp.exp() * (low_logp-ref_logp)).sum())
        action_kl = float((action_logp.exp() * (action_logp-ref_logp)).sum())
        reverse_base = float((ref_logp.exp() * (ref_logp-low_logp)).sum())
        reverse_action = float((ref_logp.exp() * (ref_logp-action_logp)).sum())
        rows.append(dict(index=index, target_token_id=int(target), is_eos=int(target) in eos_ids,
                         base_kl_to_reference=base_kl, action_kl_to_reference=action_kl,
                         fidelity_gain=base_kl-action_kl,
                         reverse_base_kl=reverse_base, reverse_action_kl=reverse_action,
                         reverse_fidelity_gain=reverse_base-reverse_action))
    content = [row for row in rows if not row['is_eos']]
    def mean(key, selected):
        return sum(row[key] for row in selected)/len(selected) if selected else None
    return dict(per_token=rows,
                mean_base_kl_to_reference=mean('base_kl_to_reference', rows),
                mean_action_kl_to_reference=mean('action_kl_to_reference', rows),
                fidelity_gain_mean=mean('fidelity_gain', rows),
                fidelity_gain_sum=sum(row['fidelity_gain'] for row in rows),
                content_base_kl_to_reference=mean('base_kl_to_reference', content),
                content_action_kl_to_reference=mean('action_kl_to_reference', content),
                content_fidelity_gain_mean=mean('fidelity_gain', content),
                content_fidelity_gain_sum=sum(row['fidelity_gain'] for row in content),
                reverse_fidelity_gain_mean=mean('reverse_fidelity_gain', rows),
                reverse_content_fidelity_gain_mean=mean('reverse_fidelity_gain', content),
                token_count=len(rows), content_token_count=len(content),
                kl_direction='D_KL(p_candidate || p_same_token_allref)',
                positive_means='closer to full-reference distribution on the same tokens; not accuracy')


def _read_cost(bank, pass_kind):
    rows = [r for r in bank.event_rows() if r['pass_kind'] == pass_kind]
    keys = ('calls', 'query_tokens', 'base_visual_bytes', 'residual_bytes', 'text_bytes', 'reference_bytes')
    totals = {key: sum(r[key] for r in rows) for key in keys}
    totals['logical_visual_read_bytes'] = sum(totals[k] for k in ('base_visual_bytes', 'residual_bytes', 'reference_bytes'))
    return dict(totals=totals, by_phase_layer=rows,
                scope='logical tensor payload per fused call; not measured HBM transactions')


def teacher_turn(model, bank, start, turn, number, mode, initial_logits, pass_kind, action=None):
    """Teacher-force one entire answer at the original chunking and positions."""
    from p184_working_set import attention
    bank.restore(start)
    bank.turn, bank.pass_kind = number, pass_kind
    logical = _snapshot_position(start)
    _set_read(bank, mode, 'text_prefill', action=action)
    if turn['suffix_token_ids']:
        ids = torch.tensor([turn['suffix_token_ids']], device=model.device)
        with attention(model):
            out, logical = consume(model, bank.cache, ids, logical)
        logits = out.logits[:, -1].detach()
        del out
    else:
        logits = initial_logits.to(model.device)
    measured = []
    for step, token in enumerate(turn['token_ids']):
        measured.append(logits.float().cpu().clone())
        _set_read(bank, mode, 'decode', step, action)
        incoming = torch.tensor([[token]], device=model.device)
        with attention(model):
            out, logical = consume(model, bank.cache, incoming, logical)
        logits = out.logits[:, -1].detach()
        del out
    if turn['forced_chat_closure']:
        _set_read(bank, mode, 'decode', len(turn['token_ids']), action)
        incoming = torch.tensor([[turn['closure_token_id']]], device=model.device)
        with attention(model):
            out, logical = consume(model, bank.cache, incoming, logical)
        del out
    if logical != turn['logical_end']:
        raise ValueError('Replay changed the original token/position schedule')
    return measured, bank.snapshot(cpu=True)


def generate_canonical(model, processor, bank, row, args, initial_logits):
    """Generate the sole token transcript under base visual reads."""
    from p180_multiturn_core import next_suffix
    from p184_working_set import attention
    import run_p175_modern_replication as oldbase
    eos = set(oldbase.stop_ids(model, processor))
    closure = int(processor.tokenizer.eos_token_id)
    logical = bank.states[0].length
    logits = initial_logits.to(model.device)
    turns, traces, histories = [], [], []
    for number, question in enumerate(row['questions'][:args.turns], 1):
        bank.turn, bank.pass_kind = number, 'canonical_base_generation'
        _set_read(bank, 'base', 'text_prefill')
        suffix = next_suffix(processor, question).to(model.device) if number > 1 else None
        suffix_ids = suffix[0].tolist() if suffix is not None else []
        logical_start = logical
        if suffix is not None:
            with attention(model):
                out, logical = consume(model, bank.cache, suffix, logical)
            logits = out.logits[:, -1].detach()
            del out
        ids, trace = [], []
        for step in range(args.max_tokens):
            trace.append(logits.float().cpu().clone())
            token = int(logits.argmax(-1).item())
            ids.append(token)
            _set_read(bank, 'base', 'decode', step)
            with attention(model):
                out, logical = consume(model, bank.cache, torch.tensor([[token]], device=model.device), logical)
            logits = out.logits[:, -1].detach()
            del out
            if token in eos:
                break
        closed = not ids or ids[-1] != closure
        if closed:
            _set_read(bank, 'base', 'decode', len(ids))
            with attention(model):
                out, logical = consume(model, bank.cache, torch.tensor([[closure]], device=model.device), logical)
            logits = out.logits[:, -1].detach()
            del out
        turn = dict(turn=number, suffix_token_ids=suffix_ids, token_ids=ids,
                    answer=processor.tokenizer.decode(ids, skip_special_tokens=True).strip(),
                    forced_chat_closure=closed, closure_token_id=closure,
                    logical_start=logical_start, logical_end=logical)
        turns.append(turn)
        traces.append(trace)
        histories.append(bank.snapshot(cpu=True))
        print(json.dumps(dict(session=row['id'],stage='canonical_base',turn=number,answer=turn['answer'])), flush=True)
    return turns, traces, histories, eos


def parse_actions(names):
    if 'all' in names:
        return [(group, phase) for group in range(4) for phase in ('text_prefill', 'early', 'late')]
    result = []
    for name in names:
        normalized = name.replace('/', ':')
        if ':' not in normalized:
            normalized = normalized.replace('_', ':', 1)
        group, phase = normalized.split(':', 1)
        group = int(group.removeprefix('layer'))
        if group not in range(4) or phase not in ('text_prefill', 'early', 'late'):
            raise ValueError('Action must be layer0..3:text_prefill|early|late')
        if (group, phase) not in result:
            result.append((group, phase))
    return result


def action_probes(model, processor, bank, before, base_after, turns, traces, reference_traces,
                  number, args, initial_logits, eos):
    results = []
    for action in args.parsed_actions:
        name = f'layer{action[0]}:{action[1]}'
        tag = f'action_{name}_write{number}'
        current, state = teacher_turn(model, bank, before, turns[number-1], number,
                                     'base', initial_logits, tag, action)
        current_metrics = logit_comparison(current, traces[number-1], turns[number-1]['token_ids'], eos)
        current_fidelity = logit_fidelity_gain(traces[number-1], current, reference_traces[number-1],
                                              turns[number-1]['token_ids'], eos)
        current_cost = _read_cost(bank, tag)
        write_state = snapshot_difference(state, base_after, minimum_position=turns[number-1]['logical_start'])
        future = []
        for future_number in range(number+1, min(len(turns), number+args.future_horizon)+1):
            future_tag = f'{tag}_future{future_number}'
            logits, state = teacher_turn(model, bank, state, turns[future_number-1], future_number,
                                        'base', initial_logits, future_tag)
            metrics = logit_comparison(logits, traces[future_number-1], turns[future_number-1]['token_ids'], eos)
            fidelity = logit_fidelity_gain(traces[future_number-1], logits, reference_traces[future_number-1],
                                           turns[future_number-1]['token_ids'], eos)
            future.append(dict(target_turn=future_number, metrics=metrics, fidelity=fidelity,
                               read_cost=_read_cost(bank, future_tag)))
        results.append(dict(action=name,write_turn=number,layer_group=action[0],phase=action[1],
                            current=current_metrics, current_fidelity=current_fidelity,
                            current_read_cost=current_cost,
                            history_state_after_write=write_state,future=future,
                            future_read_mode='base_only',target_source='base_generated_tokens',
                            primary_objective='same_token_allref_logit_KL_fidelity_gain',
                            interpretation='fixed-token distribution fidelity; no ground-truth quality claim'))
        print(json.dumps(dict(stage='action',turn=number,action=name,
                              current_fidelity_gain=current_fidelity['content_fidelity_gain_mean'],
                              future_fidelity_gains=[f['fidelity']['content_fidelity_gain_mean'] for f in future],
                              target_logprob_diagnostic=current_metrics['content_logprob_gain_mean'])),flush=True)
    return results


def run_mechanism(model, processor, row, args):
    from p180_multiturn_core import attention_bridge
    _sync()
    started = time.perf_counter()
    f = {k: v.to(model.device, dtype=model.dtype if v.is_floating_point() else v.dtype)
         for k, v in features(processor, row, args.max_patches).items()}
    visual = torch.nonzero(f['input_ids'][0] == model.config.image_token_id).flatten()
    with attention_bridge(model, native=True):
        out = model(**f, use_cache=True, return_dict=True, logits_to_keep=1)
    initial_logits = out.logits[:, -1].detach().float().cpu().clone()
    cache = out.past_key_values
    del out, f
    bank = ProgressiveBank(model, cache, visual, 'asym', args.legacy_scripts,
                           args.kivi_source, keep_reference=True)
    try:
        initial = bank.snapshot(cpu=True)
        initial_position = _snapshot_position(initial)
        turns, traces, original_states, eos = generate_canonical(model, processor, bank, row, args, initial_logits)
        # Independent complete all-reference history, computed before any action
        # so every action is compared with exactly the same offline reference.
        reference_traces = []
        reference_state = initial
        for number, turn in enumerate(turns, 1):
            logits, reference_state = teacher_turn(model, bank, reference_state, turn, number,
                                                   'ref', initial_logits, 'offline_allref_trace')
            reference_traces.append(logits)
        print(json.dumps(dict(session=row['id'], stage='offline_allref_trace',
                              turns=len(reference_traces), online_feature=False)), flush=True)
        ref_state = base_state = repeat_ref_state = initial
        rows, actions = [], []
        logit_artifact = []
        for number, turn in enumerate(turns, 1):
            starting = {'ref': ref_state, 'base': base_state}
            before_difference = snapshot_difference(base_state, ref_state, initial_position)
            arms, end_states, arm_logits = {}, {}, {}
            for history, current in (('ref','ref'), ('ref','base'), ('base','ref'), ('base','base')):
                name = f'history_{history}__current_{current}'
                logits, after = teacher_turn(model, bank, starting[history], turn, number,
                                             current, initial_logits, name)
                arm_logits[name] = logits
                if history == current:
                    end_states[history] = after
            reference = arm_logits['history_ref__current_ref']
            for name, logits in arm_logits.items():
                arms[name] = logit_comparison(logits, reference, turn['token_ids'], eos)
                arms[name]['read_cost'] = _read_cost(bank, name)
                # Event keys include turn; do not silently cumulate prior turns.
                arms[name]['read_cost']['by_phase_layer'] = [r for r in arms[name]['read_cost']['by_phase_layer'] if r['turn'] == number]
                keys = ('calls','query_tokens','base_visual_bytes','residual_bytes','text_bytes','reference_bytes')
                totals = {k:sum(r[k] for r in arms[name]['read_cost']['by_phase_layer']) for k in keys}
                totals['logical_visual_read_bytes'] = sum(totals[k] for k in ('base_visual_bytes','residual_bytes','reference_bytes'))
                arms[name]['read_cost']['totals'] = totals
            # N1 repeats the complete all-reference teacher-forced trajectory
            # independently, including its text-state construction at every turn.
            repeated, repeat_ref_state = teacher_turn(model, bank, repeat_ref_state, turn,
                                                      number, 'ref', initial_logits, 'N1_allref_repeat')
            null_ref = logit_comparison(repeated, reference, turn['token_ids'], eos)
            null_ref['history_state'] = snapshot_difference(repeat_ref_state, end_states['ref'], initial_position)
            null_precomputed_ref = logit_comparison(reference, reference_traces[number-1],
                                                    turn['token_ids'], eos)
            # N2 compares the originally free-selected base token sequence with
            # exact same-token replay; targets are never re-selected here.
            base_logits = arm_logits['history_base__current_base']
            null_base = logit_comparison(base_logits, traces[number-1], turn['token_ids'], eos)
            null_base['history_state'] = snapshot_difference(end_states['base'], original_states[number-1], initial_position)
            sums = {name:value['target_logprob_sum'] for name,value in arms.items()}
            rr = sums['history_ref__current_ref']
            rb = sums['history_ref__current_base']
            br = sums['history_base__current_ref']
            bb = sums['history_base__current_base']
            item = dict(turn=number,target=turn,arms=arms,history_state_before=before_difference,
                        contrasts=dict(current_base_effect_given_ref_history=rb-rr,
                                       base_history_effect_given_ref_read=br-rr,
                                       joint_base_effect=bb-rr,interaction=bb-br-rb+rr,
                                       reference_history_rescue_given_base_read=rb-bb,
                                       reference_read_rescue_given_base_history=br-bb),
                        numerical_nulls=dict(N1_allref_reencoding_repeat=null_ref,
                                             N2_base_free_vs_teacher_forced=null_base,
                                             N3_precomputed_allref_vs_four_arm=null_precomputed_ref))
            if number >= args.min_target_turn:
                rows.append(item)
            if (args.parsed_actions and number < len(turns)
                    and (not args.action_turns or number in args.action_turns)):
                actions.extend(action_probes(model,processor,bank,base_state,end_states['base'],
                                             turns,traces,reference_traces,number,args,initial_logits,eos))
            if args.save_logits:
                logit_artifact.append(dict(turn=number,arms=arm_logits,canonical=traces[number-1],
                                           allref_repeat=repeated,offline_allref=reference_traces[number-1]))
            ref_state, base_state = end_states['ref'], end_states['base']
            print(json.dumps(dict(session=row['id'],stage='four_arm',turn=number,
                                  contrasts=item['contrasts'],
                                  N1_max=null_ref['max_abs_logit_difference'],
                                  N2_max=null_base['max_abs_logit_difference'])),flush=True)
        _sync()
        current_table = [dict(session_id=row['id'],action=a['action'],write_turn=a['write_turn'],
                              layer_group=a['layer_group'], phase=a['phase'],
                              gain=a['current_fidelity']['content_fidelity_gain_mean'],
                              gain_sum=a['current_fidelity']['content_fidelity_gain_sum'],
                              metric='content_mean_KL_fidelity_gain',
                              residual_read_bytes=a['current_read_cost']['totals']['residual_bytes'],
                              target_logprob_diagnostic=a['current']['content_logprob_gain_mean'],
                              read_cost=a['current_read_cost']['totals']) for a in actions]
        future_table = [dict(session_id=row['id'],action=a['action'],write_turn=a['write_turn'],
                             layer_group=a['layer_group'], phase=a['phase'],
                             target_turn=future['target_turn'],
                             horizon=future['target_turn']-a['write_turn'],
                             gain=future['fidelity']['content_fidelity_gain_mean'],
                             gain_sum=future['fidelity']['content_fidelity_gain_sum'],
                             metric='content_mean_KL_fidelity_gain',
                             residual_read_bytes=a['current_read_cost']['totals']['residual_bytes'],
                             target_logprob_diagnostic=future['metrics']['content_logprob_gain_mean'],
                             action_read_cost=a['current_read_cost']['totals'],
                             future_read_cost=future['read_cost']['totals']) for a in actions for future in a['future']]
        scatter = [dict(intervention_id=f'{row["id"]}:t{a["write_turn"]}:{a["action"]}:future{future["target_turn"]}',
                        session_id=row['id'],image_id=row.get('source_image_id',row.get('image_id')),
                        delta_current=a['current_fidelity']['content_fidelity_gain_mean'],
                        delta_future=future['fidelity']['content_fidelity_gain_mean'],
                        current_metric='content_mean_KL_fidelity_gain_to_same_token_allref',
                        future_metric='content_mean_KL_fidelity_gain_to_same_token_allref',
                        current_content_tokens=a['current']['content_token_count'],
                        future_content_tokens=future['metrics']['content_token_count'],
                        horizon=future['target_turn']-a['write_turn'],
                        policy=a['action'],action=a['action'],write_turn=a['write_turn'],
                        target_turn=future['target_turn'],
                        current_read_cost=a['current_read_cost'],future_read_cost=future['read_cost'],
                        target_logprob_current_diagnostic=a['current']['content_logprob_gain_mean'],
                        target_logprob_future_diagnostic=future['metrics']['content_logprob_gain_mean'],
                        reference_availability='offline_reference_only',
                        kl_direction='D_KL(p_candidate || p_same_token_allref)',
                        positive_means='reduced divergence from same-token full-reference distribution; not accuracy')
                   for a in actions for future in a['future']]
        null_rows = [dict(turn=r['turn'],control=name,**{k:v for k,v in null.items() if k!='history_state'})
                     for r in rows for name,null in r['numerical_nulls'].items()]
        result = dict(schema='trajkv.mechanism_fidelity.v2',session_id=row['id'],
                      image_id=row.get('source_image_id',row.get('image_id')),split=row.get('split'),
                      protocol='same_base_generated_tokens_same_positions_history_visual_crossover',
                      canonical_turns=turns,results=rows,actions=actions,
                      current_gain_table=current_table,future_gain_table=future_table,
                      action_scatter=scatter,
                      primary_action_metric='content_mean_KL_fidelity_gain_to_same_token_allref',
                      primary_kl_direction='D_KL(p_base || p_ref)-D_KL(p_action || p_ref)',
                      online_policy_scope='only frozen development-fit tables may be used online; no reference trace access',
                      numerical_null_distribution=null_rows,
                      numerical_null_policy='report empirical errors; no hard-coded numerical rejection threshold',
                      first_turn_scope='initial visual prefix and first answer-token logits are common native FP16; compression acts after prefill',
                      history_scope='all prior text states re-encoded under ref/base visual reads; identical token IDs, positions and chunk schedule',
                      reference_scope='allref is original FP16 visual read through the same fused backend; not separate-native-backend equivalence',
                      target_scope='canonical compressed-model generated tokens; primary KL excludes EOS; likelihood retained as diagnostic; no ground-truth answers',
                      state_controls='legal complete ref/base histories; no arbitrary tensor interpolation',
                      causal_contrast_scope='fixed-token interventional contrasts; no claim of natural mediation or downstream task quality',
                      owned_diagnostic=bank.owned(),read_events=bank.event_rows(),
                      visual_tokens=int(visual.numel()),session_ms=1000*(time.perf_counter()-started),
                      paid_calls=0,reference_answers_visible_to_runner=False)
        if args.save_logits:
            torch.save(logit_artifact,args.output/f'{row["id"]}_fixed_token_logits.pt')
        return result
    finally:
        bank.close()
        del bank, cache
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model','samples','legacy-scripts','kivi-source','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--count',type=int,default=8)
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--turns',type=int,default=10)
    parser.add_argument('--max-tokens',type=int,default=24)
    parser.add_argument('--max-patches',type=int,default=4)
    parser.add_argument('--min-target-turn',type=int,default=2)
    parser.add_argument('--actions',nargs='*',default=[],help='all, or layer0:text_prefill layer1:early ...')
    parser.add_argument('--action-turns',nargs='*',type=int,default=[])
    parser.add_argument('--future-horizon',type=int,default=1)
    parser.add_argument('--save-logits',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    args.parsed_actions = parse_actions(args.actions)
    if min(args.count,args.turns,args.max_tokens,args.max_patches,args.future_horizon) < 1:
        parser.error('Counts, turns, token cap, patch cap and future horizon must be positive')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    install_legacy(args.legacy_scripts)
    from transformers import AutoModelForImageTextToText, AutoProcessor
    torch.manual_seed(1212)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    rows = json.loads(args.samples.read_text(encoding='utf-8'))[args.start:args.start+args.count]
    manifest = dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                    selected_ids=[r['id'] for r in rows],torch=torch.__version__,
                    gpu=torch.cuda.get_device_name(),
                    sources={str(p):digest(p) for p in (Path(__file__),Path(__file__).with_name('bank.py'),
                                                       Path(__file__).with_name('run_cycle.py'),args.samples)},
                    schema='trajkv.mechanism_fidelity.v2',
                    frozen_v1_lineage_sha256=digest(Path(__file__).with_name('mechanism.py')),
                    primary_objective='same-token full-reference logit KL fidelity gain, not answer accuracy',
                    canonical_generation='packed base2, all visual support',
                    reference_answers_visible_to_runner=False,paid_calls=0)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    processor=AutoProcessor.from_pretrained(args.model,local_files_only=True)
    model=AutoModelForImageTextToText.from_pretrained(args.model,local_files_only=True,
                dtype=torch.float16,device_map='cuda:0',attn_implementation='sdpa').eval()
    logpath=args.output/'records.jsonl'
    seen=set()
    if args.resume and logpath.exists():
        seen={json.loads(line)['session_id'] for line in logpath.read_text(encoding='utf-8').splitlines() if line.strip()}
    count=0
    with torch.inference_mode(),logpath.open('a',encoding='utf-8') as log, \
            (args.output/'action_scatter.jsonl').open('a',encoding='utf-8') as scatter_log:
        for row in rows:
            if row['id'] in seen:
                continue
            result=run_mechanism(model,processor,row,args)
            log.write(json.dumps(result)+'\n')
            log.flush()
            for point in result['action_scatter']:
                scatter_log.write(json.dumps(point)+'\n')
            scatter_log.flush()
            count+=1
    (args.output/'complete.json').write_text(json.dumps(dict(records_added=count,paid_calls=0)),encoding='utf-8')


if __name__=='__main__':
    main()
