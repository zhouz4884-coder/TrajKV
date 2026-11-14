"""Derive separate-cycle readers/executor; never mutate frozen originals."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
source=(ROOT/'code/experiments/fused_progressive.py').read_text(encoding='utf-8')
s=source.replace('NORM,\n','NORM, MASK,\n').replace('SELECTED: tl.constexpr, NI: tl.constexpr,','SELECTED: tl.constexpr, NI: tl.constexpr, MASKED: tl.constexpr, PAGE: tl.constexpr,')
start=s.index('                if REFINE_K:');end=s.index('\n            else:\n                kp = VK',start)
old=s[start:end]
new='                use_residual = True\n                if MASKED:\n                    use_residual = tl.load(MASK+owner*tl.cdiv(NVFULL,PAGE)+(block*BN)//PAGE) != 0\n                if use_residual:\n'+''.join('    '+line+'\n' for line in old.splitlines()).rstrip()
s=s[:start]+new+s[end:]
s=s.replace('refine_k=True, refine_v=True):','refine_k=True, refine_v=True, refinement_mask=None, page_size=256):')
at=s.index('    vk=state.k_code')
s=s[:at]+'''    if refinement_mask is not None:
        if selected is not None or page_size not in (128,256) or state.visual_pos.numel()%page_size:
            raise ValueError('Masked refinement preserves all support and requires aligned 128/256-token pages')
        expected=(b,state.text_k.shape[1],state.visual_pos.numel()//page_size)
        if tuple(refinement_mask.shape)!=expected or not refinement_mask.is_contiguous():
            raise ValueError('Expected contiguous [batch,kv_head,page] mask')
        if refinement_mask.dtype not in (torch.bool,torch.uint8):raise ValueError('Byte mask required')
''' +s[at:]
s=s.replace('[query]*6), norm,','[query]*6), norm, refinement_mask if refinement_mask is not None else query,')
s=s.replace('observe, selected is not None, nv, 16, 32,','observe, selected is not None, nv, refinement_mask is not None, page_size, 16, 32,')
s='"""A2 all-support page residual reader; derived from frozen fused_progressive.py.\nMask selects residual payload only, never visual softmax support.\n"""\n'+s
(HERE/'fused_masked.py').write_text(s,encoding='utf-8')
old=(ROOT/'code/experiments/run_cycle.py').read_text(encoding='utf-8').split('\ndef main():')[0]
v=old.replace('from bank import ProgressiveBank,install_legacy','from masked_bank import A2Bank as ProgressiveBank\nfrom bank import install_legacy')
v=v.replace("bank.turn=number;bank.phase='text_prefill'", "bank.on_turn_begin(number,logical);bank.turn=number;bank.phase='text_prefill'")
v=v.replace('        turns.append(turn)','        bank.on_turn_end(number,logical)\n        turns.append(turn)')
v=v.replace("raw_extra=sum(e['residual_bytes']+", "raw_extra=sum(e.get('selection_bytes',0)+e['residual_bytes']+")
v=v.replace("      result['read_ledger_exact']=True", "      result['read_ledger_exact']=True\n      result.update(bank.result_metadata())")
v=v.replace("      result['kernel_variant']='local_1bit_axis_extension' if bank.extended_kernel else 'legacy_p202_fused'", "      result['kernel_variant']='a2_all_support_masked_refinement'")
v=v.replace("kernel=Path(__file__).with_name('fused_progressive.py') if bank.extended_kernel else args.legacy_scripts/'p202_fused.py'", "kernel=Path(__file__).with_name('fused_masked.py')")
(HERE/'session.py').write_text(v,encoding='utf-8')
record={'derived_from':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'code/experiments/fused_progressive.py',ROOT/'code/experiments/run_cycle.py']},
        'derived':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'fused_masked.py',HERE/'session.py']}}
(ROOT/'audit/a2/derived_sources.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
