"""Selected final-state common-ray K controls; no existing source edits."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

from round8_postres_runner import validate_variant as original_format_validation

MODES=('off','mse','norm','projection')
SOURCE_FILES=('round8_final_ray_runner.py','round8_final_ray.py','round8_postres_runner.py',
    'round8_field_joint_runner.py','round8_bank.py','round8_codec.py','round8_attention.py',
    'round8_runner.py','round6_depth_codec.py','round7_key_layout.py','quantizers.py','affine_rate_bank.py','fast_wht.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fingerprints():
    result={name:sha(Path(__file__).with_name(name)) for name in SOURCE_FILES}
    result['experiments/fused_progressive.py']=sha(Path(__file__).resolve().parents[1]/'experiments/fused_progressive.py')
    return result


def validate_variant(variant):
    mode=variant.get('final_ray_mode','off')
    if mode not in MODES:raise ValueError('final_ray_mode must be off/mse/norm/projection')
    if variant.get('postres_direction',False):raise ValueError('Final ray keeps J fixed; no postres direction recoding')
    original=original_format_validation(variant)
    if original is None:
        if mode!='off':raise ValueError('A control method cannot silently ignore final-state ray scaling')
        return None
    return dict(mode=mode,ranking_bytes=448,requested_stages=original['requested_stages'],selected_stages=original['selected_stages'])


def validate_config(config):
    if config.get('backend_family','final_ray')!='final_ray':raise ValueError('No mixed final-ray backend family')
    if not config.get('variants'):raise ValueError('At least one explicit candidate/control required')
    declarations,common={},None
    for variant in config['variants']:
        if variant['name'] in declarations:raise ValueError('Duplicate variant name')
        options=validate_variant(variant);declarations[variant['name']]=options
        if options is None:continue
        signature=dict(stages=options['requested_stages'],inputs={key:variant.get(key) for key in
            ('direction_book_path','metadata_reference_book_path','metadata_factor_path','round8_factor_policy','budget_bits','depth_policy','cap')})
        if common is not None and common!=signature:raise ValueError('Final-ray candidates must share full ranking/book/factors/cap/price')
        common=signature
    return declarations


def final_ray_class(base_class):
    from round8_final_ray import refit_key_final_ray
    class FinalRayBank(base_class):
        def _initialize(self,*args,**kwargs):
            cfg=type(self).settings;self.final_ray_options=validate_variant(cfg);self.final_ray_statistics=[]
            if self.final_ray_options is not None:
                book,reference=Path(cfg['direction_book_path']),Path(cfg['metadata_reference_book_path'])
                if sha(book)!=sha(reference):raise ValueError('Unchanged old factor-reference K book required')
                centers=json.loads(book.read_text(encoding='utf-8-sig'))['centers']
                if len(centers)!=16 or any(len(row)!=4 for row in centers):raise ValueError('Old global16x4 book required')
            result=super()._initialize(*args,**kwargs);options=self.final_ray_options
            if options is not None:
                if self.joint_gain_enabled:raise RuntimeError('Ordinary NNLS must not precede ray scaling')
                if list(map(tuple,self.selected_residual_stages))!=options['selected_stages'] or list(map(tuple,self.requested_residual_stages))!=options['requested_stages'] or self.ranking_constant_bytes!=448:
                    raise RuntimeError('Inherited bank changed complete224 ranking/actual43 mask')
                expected=sum(axis=='k' for axis,_,_,_ in options['selected_stages']) if options['mode']!='off' else 0
                if len(self.final_ray_statistics)!=expected:raise RuntimeError('Selected K head ray coverage mismatch')
            return result

        def _encode_axis(self,layer,state,axis,selected,source,base):
            result=super()._encode_axis(layer,state,axis,selected,source,base)
            options=self.final_ray_options
            if axis!='k' or options is None or options['mode']=='off':return result
            if self.joint_gain_enabled:raise RuntimeError('Original norm/fold+MAD is required')
            for head in sorted({head for a,head,_ in selected if a=='k'}):
                if sum(a=='k' and h==head for a,h,_ in selected)!=1 or ('k',head,0) not in selected:
                    raise ValueError('Final ray requires exactly one existing first K residual')
                slot=selected.index(('k',head,0));factors=self.encoder_factors.get('key_head_factors')
                factor=None if factors is None else factors[layer,head]
                gain,alpha,stats=refit_key_final_ray(source[head].float(),self.book,state.direction_code[0,head],state.direction_gain[0,head],
                    state.depth_residual_code[slot],state.depth_residual_alpha[slot],mode=options['mode'],gain_factor=factor)
                state.direction_gain[0,head].copy_(gain);state.depth_residual_alpha[slot].copy_(alpha)
                entries=[e for e in self.depth_distortions if e['axis']=='k' and e['layer']==layer and e['head']==head]
                if len(entries)!=1 or entries[0]['depth']!=1:raise RuntimeError('Expected one original first-K distortion entry')
                entry=entries[0];entry['pre_final_ray_squared_error_by_depth']=list(entry['squared_error_by_depth'])
                entry['squared_error_by_depth']=[stats['source_base_sse_after'],stats['source_sse_after']]
                entry['final_ray_mode']=options['mode'];entry['scope']='actual complete-state common ray after original signMAD; supplied centered-WHT error; own objective may permit SSE increase'
                self.final_ray_statistics.append(dict(layer=layer,head=head,frozen_head_factor_supplied=factor is not None,**stats))
            return result

        def result_metadata(self):
            result=super().result_metadata();options=self.final_ray_options
            if options is None:return result
            mode=options['mode']
            result.update(stage='round8 selected complete-residual-state common-ray controls; unchanged packed reader/owner',
                backend_family='final_ray',final_ray_mode=mode,final_ray_statistics=self.final_ray_statistics,
                final_ray_J_sign_unchanged=True,final_ray_added_retained_tensor_bytes=0,final_ray_reader_unchanged=True,
                final_ray_scope='selected existing first-stage K heads only; original g:alpha ratio scaled before independent FP16 field rounding',
                final_ray_acceptance={'off':'original fields','mse':'strict actual SSE decrease','norm':'strict actual absolute norm-error decrease; no SSE gate',
                    'projection':'strict actual source-self-projection-error decrease; no SSE gate'}[mode],
                final_ray_projection_scope='only projection on supplied source; no arbitrary-query/unbiased claim',
                final_ray_direction_scope='common-ray proposal before rounding; actual direction drift measured, no exact FP16 angle invariance',
                final_ray_metric_scope='supplied centered-WHT coordinates; not mean/native bridge or task quality',final_ray_source_sha256=source_fingerprints())
            if mode!='off':
                result.update(key_gain_rule='final-state-common-ray-'+mode,
                    key_gain_formula='selected first K stage: Q16(c*old_effective_g),Q16(c*old_alpha) with c from original final nested-half state; original J/sign; unselected K keeps original norm/fold',
                    key_gain_rounding='original given-J norm->FP16->frozen factor->FP16 and signMAD, then common c->FP16 g/alpha; no second factor fold',
                    encoding_assignment='original frozen l2 J unchanged; complete-state radial coefficient proposal only')
            return result
    return FinalRayBank


@contextmanager
def backend_registration(family='final_ray'):
    if family!='final_ray':raise ValueError('This registration only implements final_ray')
    import round8_runner  # noqa: F401
    import round8_bank
    import factory
    import run_representation_variants as executor
    old_factory,old_executor=factory.bank_class,executor.bank_class;cls=final_ray_class(round8_bank.Round8Bank)
    def choose(method):return cls if method=='round8_joint' else old_factory(method)
    factory.bank_class=executor.bank_class=choose
    try:yield cls
    finally:factory.bank_class,executor.bank_class=old_factory,old_executor


def finalize_run(output,external,sources,config_sha,declarations):
    output=Path(output)
    if (output/'complete_round8.json').exists() or (output/'complete_final_ray.json').exists():raise ValueError('Use new immutable output')
    if source_fingerprints()!=sources or any(sha(path)!=value for path,value in external.items()):raise RuntimeError('Source/input changed; preserve run')
    engine,manifest_path=output/'complete.json',output/'manifest.json';complete=json.loads(engine.read_text(encoding='utf-8-sig'))
    if complete.get('status')!='COMPLETE' or complete.get('records')!=complete.get('expected') or not complete.get('expected',0):raise RuntimeError('Incomplete producer coverage')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if manifest.get('config_sha256')!=config_sha:raise RuntimeError('Producer consumed different config')
    manifest.setdefault('external_codec_inputs',{}).update(external)
    manifest.update(stage='round8 final-state common-ray candidates',backend_family='final_ray',candidate_subset=True,factorial_complete=False,
        final_ray_declarations=declarations,final_ray_source_sha256=sources,round8_engine_complete_requires_postcheck=True,
        executed_candidate_variants=[v['name'] for v in manifest['config']['variants']])
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8');records_sha=sha(output/'records.jsonl')
    marker=dict(status='COMPLETE',engine_complete_sha256=sha(engine),manifest_sha256=sha(manifest_path),records_sha256=records_sha,
        encoder_inputs_unchanged=True,source_inputs_unchanged=True,backend_family='final_ray',candidate_subset=True,factorial_complete=False,final_ray_source_sha256=sources)
    with (output/'complete_round8.json').open('x',encoding='utf-8') as stream:json.dump(marker,stream,indent=2)
    final=dict(status='COMPLETE',complete_round8_sha256=sha(output/'complete_round8.json'),manifest_sha256=sha(manifest_path),records_sha256=records_sha,
        config_sha256=config_sha,runner_sha256=sha(__file__),source_sha256=sources,backend_family='final_ray',candidate_subset=True,factorial_complete=False,accepted_only_after_this_marker=True)
    with (output/'complete_final_ray.json').open('x',encoding='utf-8') as stream:json.dump(final,stream,indent=2)
    return final


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args();config_sha=sha(args.config);config=json.loads(args.config.read_text(encoding='utf-8-sig'))
    declarations,sources=validate_config(config),source_fingerprints()
    if args.plan_only:
        print(json.dumps(dict(status='PLAN_ONLY',backend_family='final_ray',candidate_subset=True,factorial_complete=False,
            variants=declarations,source_sha256=sources,GPU_executed=False,added_retained_tensor_bytes=0)),flush=True);return
    import round8_runner as original_runner
    external={v[key]:sha(v[key]) for v in config['variants'] for key in original_runner.EXTRA_KEYS if v.get(key)}
    with backend_registration('final_ray'):
        import run_representation_variants as executor
        executor.main()
    if sha(args.config)!=config_sha:raise RuntimeError('Config changed; preserve run')
    finalize_run(args.output,external,sources,config_sha,declarations)


if __name__=='__main__':main()
