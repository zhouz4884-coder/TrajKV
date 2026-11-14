"""Image-boundary protocol: compress before re-reading first question suffix.

Native causal visual-prefix KV is extracted from an initial pass. The entire
post-image text suffix is then discarded and recomputed under the selected
codec. Initial extraction/replay work remains in wall time and read ledgers.
"""
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=(HERE/'session.py').read_text(encoding='utf-8')
s=s.replace("initial_logits=logits.float().cpu();del out,f", "initial_logits=logits.float().cpu()\n    boundary_start=int(visual[-1].item())+1\n    boundary_suffix=f['input_ids'][:,boundary_start:].clone()\n    del out,f")
needle="    bank.gain_table=getattr(args,'gain_table_data',None)"
insert="""    # Causality: all source visual positions precede boundary_start. Remove
    # every later native text state before computing the question with the codec.
    for s in bank.states:
        s.text_length=int((s.text_pos[:s.text_length]<boundary_start).sum().item())
        s.length=boundary_start
    bank.phase='question_prefill';bank.pass_kind='committed';bank.turn=1
    bank.refine_layers=layer_schedule(policy,bank,'text_prefill',0,1)
    with attention(model):out,logical=consume(model,cache,boundary_suffix,boundary_start)
    logits=out.logits[:,-1];initial_logits=logits.float().cpu();del out
"""
assert needle in s;s=s.replace(needle,insert+needle)
s=s.replace("protocol='fixed_tokens' if forced else 'free_running'", "protocol='image_boundary_fixed_tokens' if forced else 'image_boundary_free_running'")
s=s.replace("      result['read_ledger_exact']=True", "      result['read_ledger_exact']=True\n      result['compression_point']='after last visual token, before complete first-question suffix'\n      result['initial_logits_shared_fp16']=False\n      result['first_question_suffix_replay_tokens']=int(boundary_suffix.shape[-1])\n      result['prefill_work_scope']='native causal prefix extracted via full prefill, later native suffix discarded and charged replayed; no extraction-speed claim'")
(HERE/'session_boundary.py').write_text(s,encoding='utf-8')
print('Created image-boundary executor; frozen shared-first executor unchanged')
