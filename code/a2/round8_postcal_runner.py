"""Deploy336FP32 post-residual head parameters over unchanged packed R42 fields."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

from round8_postcal_owner import (EXTRA_BYTES,identity_table,priority_bytes,
    table_values_sha256,validate_variant,verify_factor_artifact)

SOURCE_FILES=('round8_postcal_runner.py','round8_postcal_fields.py','round8_postcal_owner.py',
    'round8_runner.py','round8_bank.py','round8_codec.py','round8_attention.py','round8_joint_gain.py',
    'round6_depth_codec.py','round7_key_layout.py','quantizers.py','affine_rate_bank.py','fast_wht.py',
    'run_representation_variants.py','factory.py')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def source_fingerprints():
    result={name:sha(Path(__file__).with_name(name)) for name in SOURCE_FILES}
    result['experiments/fused_progressive.py']=sha(Path(__file__).resolve().parents[1]/'experiments/fused_progressive.py')
    return result
def require(value,message):
    if not value:raise ValueError(message)
def validate_config(config):
    require(config.get('backend_family','postcal')=='postcal' and config.get('variants'),'Explicit postcal candidates required')
    declarations={};common=None
    for variant in config['variants']:
        name=variant['name'];require(name not in declarations,'Duplicate variant')
        options=validate_variant(variant);declarations[name]=options
        if options is None:continue
        signature={key:variant.get(key) for key in ('direction_book_path','metadata_factor_path','metadata_reference_book_path',
            'selected_stages','res_stages_limit','cap','budget_bits','round8_factor_policy')}
        require(common is None or signature==common,'Postcal candidates must share originalbook/f0/ranking/mask/cap/budget')
        common=signature
    require(common is not None,'At least one actual postcal R42 arm required')
    return declarations

def postcal_class(base_class):
    from round8_postcal_fields import postcal_fields
    class PostcalBank(base_class):
        def _initialize(self,*args,**kwargs):
            import torch
            cfg=type(self).settings;options=validate_variant(cfg);require(options is not None,'Postcal class requires declared R42 arm')
            # The original implementation owns every initial J/sign/mean and
            # actual FP16 f0 fold and signMAD. No postcal code re-encodes them.
            result=super()._initialize(*args,**kwargs)
            require(len(self.states)==28 and self.states[0].direction_gain.shape[1]==4,'Frozen28-layer/four-KV-head geometry required')
            require(tuple(self.book.shape)==(16,4) and sha(cfg['direction_book_path'])==sha(cfg['metadata_reference_book_path']),
                'Original global Gaussian4_16 and old f0 reference required')
            require(self.vbook is None and self.vgroup==self.states[0].visual_pos.numel(),'Original all-frame binary V required')
            require(self.requested_residual_stages==[tuple(x) for x in cfg['selected_stages']]
                and self.selected_residual_stages==[tuple(x) for x in cfg['selected_stages'][:42]]
                and self.ranking_constant_bytes==448,'Original full224/42-stage mask changed')
            require(sum(x[0]=='k' for x in self.selected_residual_stages)==23 and sum(x[0]=='v' for x in self.selected_residual_stages)==19,'Original23K/19V stages required')
            self.postcal_options=options
            trained=verify_factor_artifact(cfg['postcal_factor_path'],cfg['postcal_train_complete_path'],cfg) if options['mode']=='learned' else None
            values=trained['values'] if trained else identity_table()
            table=torch.tensor(values,device=self.book.device,dtype=torch.float32).contiguous()
            require(tuple(table.shape)==(3,28,4) and table.numel()*table.element_size()==EXTRA_BYTES,'Actual retained post table geometry/bytes differs')
            self.postcal_training=trained;self.postcal_table_hash=table_values_sha256(values)
            self.postcal_original_R42_complete_bytes=self.predicted_budget_bytes
            nv=self.states[0].visual_pos.numel();self.postcal_old43_complete_bytes=self.predicted_budget_bytes+18*nv
            with torch.no_grad():
                for layer,state in enumerate(self.states):
                    new=postcal_fields(state,table[:,layer])
                    require(all(getattr(new,k) is getattr(state,k) for k in ('direction_code','direction_mean','v_code','depth_residual_lookup')),
                        'Pure field transform changed integer/mean/lookup ownership')
                    if hasattr(state,'depth_residual_code'):require(new.depth_residual_code is state.depth_residual_code,'Postcal changed residual signs')
                    for name in ('direction_gain','v_scale','v_min','depth_residual_alpha'):
                        if not hasattr(state,name):continue
                        old,candidate=getattr(state,name),getattr(new,name)
                        if options['mode']=='identity':require(torch.equal(old.view(torch.int16),candidate.view(torch.int16)),'Identity table changed actual field bits: '+name)
                        # Commit into original storage so refinements/read-ledger
                        # pointers still own the same compact alpha allocation.
                        old.copy_(candidate)
            self.postcal_factors=table
            self.encoder_parameter_bytes+=EXTRA_BYTES;self.predicted_budget_bytes+=EXTRA_BYTES
            self.depth_plan=dict(self.depth_plan)
            self.depth_plan.update(fixed_bytes=self.depth_plan['fixed_bytes']+EXTRA_BYTES,
                predicted_bytes=self.predicted_budget_bytes,actual_budget_bits=16*self.predicted_budget_bytes/self.visual_full_bytes)
            require(self.predicted_budget_bytes<=self.postcal_old43_complete_bytes and self.predicted_budget_bytes<=self.depth_plan['allowed_bytes'],
                'Complete R42 plus336FP32 table exceeds originalR43 or declared budget')
            self.actual_budget_bits=16*self.predicted_budget_bytes/self.visual_full_bytes
            require(self.owned()['complete_visual_owner_bytes']==self.predicted_budget_bytes,'Actual retained post table owner mismatch')
            for entry in self.depth_distortions:entry['scope']='Pre-postcal originalR42 coordinate distortion; final postcal source distortion not measured by this entry'
            return result

        def result_metadata(self):
            result=super().result_metadata();trained=self.postcal_training
            result.update(stage='round8 fixed-J/sign post-residual teacher-KL head calibration',backend_family='postcal',
                postcal_mode=self.postcal_options['mode'],postcal_parameter_shape=[3,28,4],
                postcal_added_encoder_constant_bytes=EXTRA_BYTES,postcal_added_retained_tensor_bytes=EXTRA_BYTES,
                postcal_table_values_sha256=self.postcal_table_hash,
                postcal_factor_file_sha256=trained['factors_sha256'] if trained else None,
                postcal_train_complete_sha256=trained['train_complete_sha256'] if trained else None,
                postcal_J_sign_mean_Vcode_unchanged=True,postcal_reader_unchanged=True,
                postcal_original_R42_complete_bytes=self.postcal_original_R42_complete_bytes,
                postcal_old43_complete_bytes=self.postcal_old43_complete_bytes,postcal_complete_le_old43=True,
                postcal_scope='Original R42 packed fields first; only existing g/S/m/alpha changed once by frozen absolute head factors; no J/sign re-encoding',
                postcal_V_min_formula='F16(old_m + [0.5*(1-fV)+o*fV]*old_S), using original unscaled storedS',
                postcal_K_formula='F16(old_g*fK), F16(all existing K alpha*fK); original mean retained',
                postcal_V_formula='F16(old_S*fV), centered min shift from oldS, F16(all existing V alpha*fV)',
                postcal_old_f0_retained_and_charged=True,postcal_full_priority_retained_and_charged=True,
                postcal_source_sha256=source_fingerprints(),
                depth_residual_distortions_scope='Pre-postcal initialization only; not the final transformed reconstruction error',
                key_gain_rule='post-residual absolute head scale after originalnorm/f0/signMAD',
                key_gain_rounding='original norm/f0 FP16 and signMAD, then one FP32 head factor into existingFP16 g and all K alphas')
            return result
    return PostcalBank

@contextmanager
def backend_registration(family='postcal'):
    require(family=='postcal','This context only implements postcal')
    import round8_runner
    import round8_bank
    import factory
    import run_representation_variants as executor
    import session
    import session_boundary
    previous=(factory.bank_class,executor.bank_class,session.ProgressiveBank,session_boundary.ProgressiveBank)
    cls=postcal_class(round8_bank.Round8Bank)
    def choose(method):return cls if method=='round8_joint' else previous[0](method)
    factory.bank_class=executor.bank_class=choose
    session.ProgressiveBank=session_boundary.ProgressiveBank=cls
    try:yield cls
    finally:
        factory.bank_class,executor.bank_class,session.ProgressiveBank,session_boundary.ProgressiveBank=previous

def finalize_run(output,external,sources,config_sha,declarations,training):
    output=Path(output)
    require(not (output/'complete_round8.json').exists() and not (output/'complete_postcal.json').exists(),'Use a new immutable output')
    require(source_fingerprints()==sources and all(sha(p)==v for p,v in external.items()),'Postcal source/input changed; preserve run')
    engine=output/'complete.json';mp=output/'manifest.json';complete=json.loads(engine.read_text(encoding='utf-8-sig'))
    require(complete.get('status')=='COMPLETE' and complete.get('records')==complete.get('expected') and complete.get('expected',0)>0,'Incomplete producer coverage')
    manifest=json.loads(mp.read_text(encoding='utf-8-sig'));require(manifest.get('config_sha256')==config_sha,'Producer consumed different config')
    records=[json.loads(x) for x in (output/'records.jsonl').read_text().splitlines() if x.strip()]
    names=[v['name'] for v in manifest['config']['variants']];ids=manifest['selected_ids']
    require(len(records)==complete['records'] and len(records)==len(ids)*len(names)
        and {(x['session_id'],x['variant']) for x in records}=={(sid,v) for sid in ids for v in names},'Incomplete actual source x arm records')
    manifest.setdefault('external_codec_inputs',{}).update(external)
    manifest.update(stage='round8 post-residual head calibration',backend_family='postcal',candidate_subset=True,factorial_complete=False,
        postcal_source_sha256=sources,postcal_declarations=declarations,postcal_training_artifacts=training,
        executed_candidate_variants=names,round8_engine_complete_requires_postcheck=True)
    mp.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    marker=dict(status='COMPLETE',backend_family='postcal',candidate_subset=True,factorial_complete=False,
        engine_complete_sha256=sha(engine),manifest_sha256=sha(mp),records_sha256=sha(output/'records.jsonl'),
        encoder_inputs_unchanged=True,source_inputs_unchanged=True,postcal_source_sha256=sources)
    with (output/'complete_round8.json').open('x',encoding='utf-8') as f:json.dump(marker,f,indent=2)
    final=dict(status='COMPLETE',backend_family='postcal',candidate_subset=True,factorial_complete=False,accepted_only_after_this_marker=True,
        complete_round8_sha256=sha(output/'complete_round8.json'),manifest_sha256=sha(mp),records_sha256=sha(output/'records.jsonl'),
        config_sha256=config_sha,runner_sha256=sha(__file__),source_sha256=sources,postcal_training_artifacts=training)
    with (output/'complete_postcal.json').open('x',encoding='utf-8') as f:json.dump(final,f,indent=2)
    return final

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--plan-only',action='store_true');args=parser.parse_args()
    config_sha=sha(args.config);config=json.loads(args.config.read_text(encoding='utf-8-sig'));declarations=validate_config(config);sources=source_fingerprints()
    if args.plan_only:
        print(json.dumps(dict(status='PLAN_ONLY',backend_family='postcal',declarations=declarations,source_sha256=sources,extra_retained_bytes=EXTRA_BYTES,GPU_executed=False)),flush=True);return
    import round8_runner
    external={v[k]:sha(v[k]) for v in config['variants'] for k in round8_runner.EXTRA_KEYS if v.get(k)};training={}
    for v in config['variants']:
        if v.get('postcal_mode')=='learned':
            proof=verify_factor_artifact(v['postcal_factor_path'],v['postcal_train_complete_path'],v)
            training[v['name']]={k:x for k,x in proof.items() if k!='values'}
            external[v['postcal_factor_path']]=proof['factors_sha256'];external[v['postcal_train_complete_path']]=proof['train_complete_sha256']
            root=Path(v['postcal_train_complete_path']).parent
            external.update({str(root/name):digest for name,digest in proof['bound_training_files_sha256'].items()})
    with backend_registration('postcal'):
        import run_representation_variants as executor
        executor.main()
    require(sha(args.config)==config_sha,'Postcal config changed; preserve run')
    finalize_run(args.output,external,sources,config_sha,declarations,training)
if __name__=='__main__':main()
