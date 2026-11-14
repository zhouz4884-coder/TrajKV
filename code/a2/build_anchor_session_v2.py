"""Derive charged-prefix-only session; preserve aborted v1 source snapshots."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
 source=Path(__file__).with_name('session_boundary.py');text=source.read_text(encoding='utf-8')
 start=text.index("    f={k:v.to(model.device,dtype=model.dtype if v.is_floating_point() else v.dtype)",text.index('def run_session('));finish=text.index('    bank.gain_table=',start)
 replacement='''    from anchor_protocol_v2 import build_inputs,prefix_inputs
    original_settings=ProgressiveBank.settings
    raw,visual,count,anchor_input_info=build_inputs(model,processor,row,args.max_patches,features,original_settings)
    f={k:v.to(model.device,dtype=model.dtype if v.is_floating_point() else v.dtype) for k,v in raw.items()}
    visual=visual.to(model.device);visual_end=int(visual[-1])+1;boundary_start=visual_end+count
    native_end=boundary_start if original_settings.get('anchor_mode')=='native' else visual_end
    native=prefix_inputs(f,native_end)
    with attention_bridge(model,native=True):out=model(**native,use_cache=True,return_dict=True,logits_to_keep=1)
    cache=out.past_key_values;logical=native_end;native_history=False
    suffix=f['input_ids'][:,boundary_start:].clone();anchor_ids=f['input_ids'][:,visual_end:boundary_start].clone()
    del out,native,f,raw
    ProgressiveBank.settings=dict(original_settings,_anchor_count=count)
    bank=ProgressiveBank(model,cache,visual,method,args.legacy_scripts,args.kivi_source,keep_reference=False)
    if original_settings.get('anchor_mode')=='compressed' and count:
        bank.phase='anchor_write';bank.pass_kind='committed';bank.turn=1
        with attention(model):out,logical=consume(model,cache,anchor_ids,visual_end)
        del out
        bank.capture_anchors()
    boundary_suffix=suffix
    bank.phase='question_prefill';bank.pass_kind='committed';bank.turn=1
    with attention(model):out,logical=consume(model,cache,boundary_suffix,boundary_start)
    logits=out.logits[:,-1];initial_logits=logits.float().cpu();del out
'''
 text=text[:start]+replacement+text[finish:]
 text=text.replace("result['question_state_scope']='native_fp16_prefix_except_last' if native_history else 'recomputed_under_codec'", "result['question_state_scope']='only charged fixed-prefix anchors precede actual question recomputation'")
 text=text.replace("kernel=Path(__file__).with_name('fused_masked.py')","kernel=Path(__file__).with_name('fused_direction.py' if method=='anchor_direction' else 'fused_rates.py' if method=='anchor_affine' else 'fused_masked.py')")
 text=text.replace("finally:bank.close();del bank,cache;gc.collect()","finally:bank.close();ProgressiveBank.settings=original_settings;del bank,cache;gc.collect()")
 text=text.replace("result.update(bank.result_metadata())","result.update(bank.result_metadata());result.update(anchor_input_info)")
 text=text.replace("result['compression_point']=('after native question prefix, replay final prefix token only' if native_history else\n                                   'after last visual token, before complete first-question suffix')", "result['compression_point']='native image plus only charged fixed-prefix tokens; actual question unseen before packing'")
 dest=source.with_name('session_anchor_v2.py');compile(text,str(dest),'exec')
 if dest.exists():raise RuntimeError('Anchor v2 session already exists')
 dest.write_text('# Charged-prefix-only anchor protocol.\n'+text,encoding='utf-8')
 (ROOT/'audit/a2/anchor_protocol_v2_build.json').write_text(json.dumps({'base_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'derived_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
  'difference':'no fixed-prompt tail; every introduced token is in the charged compact states; actual Q is absent from native forward'},indent=2))
 print(json.dumps({'status':'PREPARED','session':dest.name}))
if __name__=='__main__':main()
