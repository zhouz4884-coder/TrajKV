"""Derive isolated prefix-anchor protocol, preserving previous session evidence."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
 source=Path(__file__).with_name('session_boundary.py');text=source.read_text(encoding='utf-8')
 start=text.index("    f={k:v.to(model.device,dtype=model.dtype if v.is_floating_point() else v.dtype)",text.index('def run_session('))
 finish=text.index('    bank.gain_table=',start)
 replacement='''    from anchor_protocol import prepare_row,prefix_inputs,verify_fixed_prefix
    from anchor_bank import anchor_count
    original_settings=ProgressiveBank.settings
    row=prepare_row(row,original_settings)
    f={k:v.to(model.device,dtype=model.dtype if v.is_floating_point() else v.dtype)
       for k,v in features(processor,row,args.max_patches).items()}
    visual=torch.nonzero(f['input_ids'][0]==model.config.image_token_id).flatten()
    visual_end=int(visual[-1])+1;count=anchor_count(model,visual.numel(),original_settings)
    boundary_start=visual_end+count
    verify_fixed_prefix(processor,row,args.max_patches,features,f,boundary_start,original_settings)
    native_end=boundary_start if original_settings.get('anchor_mode')=='native' else visual_end
    native=prefix_inputs(f,native_end)
    with attention_bridge(model,native=True):out=model(**native,use_cache=True,return_dict=True,logits_to_keep=1)
    cache=out.past_key_values;logical=native_end;native_history=False
    suffix=f['input_ids'][:,boundary_start:].clone();anchor_ids=f['input_ids'][:,visual_end:boundary_start].clone()
    del out,native,f
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
 text=text.replace("'question_state_scope']='native_fp16_prefix_except_last' if native_history else 'recomputed_under_codec'", "'question_state_scope']='fixed-prefix anchors then actual question recomputed under codec'")
 old="kernel=Path(__file__).with_name('fused_masked.py')"
 assert text.count(old)==1
 text=text.replace(old,"kernel=Path(__file__).with_name('fused_direction.py' if method=='anchor_direction' else 'fused_rates.py' if method=='anchor_affine' else 'fused_masked.py')")
 text=text.replace("finally:bank.close();del bank,cache;gc.collect()","finally:bank.close();ProgressiveBank.settings=original_settings;del bank,cache;gc.collect()")
 text=text.replace("result['compression_point']=('after native question prefix, replay final prefix token only' if native_history else\n                                   'after last visual token, before complete first-question suffix')", "result['compression_point']='native image plus fixed prefix only; actual question unseen before packing'")
 dest=source.with_name('session_anchor.py');compile(text,str(dest),'exec')
 if dest.exists():raise RuntimeError('Anchor session exists')
 dest.write_text('# Isolated image-write-only prefix-anchor protocol.\n'+text,encoding='utf-8')
 runner=source.with_name('run_variants.py').read_text(encoding='utf-8')
 assert runner.count('import session_boundary\n')==1
 runner=runner.replace('import session_boundary\n','import session_anchor as session_boundary\n')
 runner=runner.replace("if v['method']=='direction':","if v['method'] in ('direction','anchor_direction'):")
 runner=runner.replace("elif v['method'] in ('rate','affine_rate'):","elif v['method'] in ('rate','affine_rate','anchor_affine'):")
 runpath=source.with_name('run_anchor_variants.py');compile(runner,str(runpath),'exec');runpath.write_text(runner,encoding='utf-8')
 report={'session_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'session_anchor_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
  'runner_sha256':hashlib.sha256(runpath.read_bytes()).hexdigest(),'status':'prepared, GPU checks and task evaluation pending'}
 (ROOT/'audit/a2/anchor_protocol_build.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
