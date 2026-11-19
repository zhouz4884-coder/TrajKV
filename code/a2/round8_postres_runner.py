"""Selected post-residual fixed-original-norm K candidates, unchanged reader.

The frozen Round8Bank first encodes original given-J norm/fold/signMAD, with
ordinary joint_gain_refit disabled. Only selected first K heads then recode
existing J/g/alpha through the independent v2 direction helper. No signs,
means, V fields, tensor layout, book or persistent GPU/cache field is added.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

from round8_field_joint_runner import validate_variant as original_format_validation


SOURCE_FILES=('round8_postres_runner.py','round8_postres_direction_v2.py',
    'round8_field_joint_runner.py','round8_bank.py','round8_codec.py','round8_attention.py',
    'round8_runner.py','round6_depth_codec.py','round7_key_layout.py','quantizers.py',
    'affine_rate_bank.py','fast_wht.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fingerprints():
    sources={name:sha(Path(__file__).with_name(name)) for name in SOURCE_FILES}
    sources['experiments/fused_progressive.py']=sha(Path(__file__).resolve().parents[1]/'experiments/fused_progressive.py')
    return sources


def validate_variant(variant):
    enabled=variant.get('postres_direction',False)
    if not isinstance(enabled,bool):
        raise ValueError('postres_direction must be boolean')
    if variant['method']!='round8_joint':
        if enabled:raise ValueError('A control method cannot silently ignore postres_direction')
        if variant.get('joint_gain_refit',False) or variant.get('value_joint_refit',False) or variant.get('key_assignment')=='row_cosine_cd':
            raise ValueError('A control method cannot silently ignore another backend intervention')
        return None
    if variant.get('joint_gain_refit',False) or variant.get('value_joint_refit',False):
        raise ValueError('No ordinary K NNLS or V refit before fixed-original-norm postres search')
    original=original_format_validation(variant)
    chunk=variant.get('postres_row_chunk',4096)
    if not isinstance(chunk,int) or isinstance(chunk,bool) or chunk<1:
        raise ValueError('Positive integer postres_row_chunk required')
    return dict(enabled=enabled,row_chunk=chunk,ranking_bytes=448,
        requested_stages=original['requested_stages'],selected_stages=original['selected_stages'])


def validate_config(config):
    declarations,common={},None
    if config.get('backend_family','postres')!='postres':raise ValueError('No cross-family postres config')
    if not config.get('variants'):raise ValueError('At least one named candidate/control required')
    for variant in config['variants']:
        if variant['name'] in declarations:raise ValueError('Duplicate variant name')
        options=validate_variant(variant);declarations[variant['name']]=options
        if options is None:continue
        signature=dict(stages=options['requested_stages'],inputs={key:variant.get(key) for key in
            ('direction_book_path','metadata_reference_book_path','metadata_factor_path',
             'round8_factor_policy','budget_bits','depth_policy','cap')})
        if common is not None and common!=signature:
            raise ValueError('Postres candidates must share full ranking, book/factors, cap and price')
        common=signature
    return declarations


def postres_class(base_class):
    from round8_postres_direction_v2 import recode_key_postres_direction
    class PostresBank(base_class):
        def _initialize(self,*args,**kwargs):
            cfg=type(self).settings
            self.postres_options=validate_variant(cfg)
            self.postres_statistics=[]
            if self.postres_options is not None:
                book,reference=Path(cfg['direction_book_path']),Path(cfg['metadata_reference_book_path'])
                if sha(book)!=sha(reference):raise ValueError('Unchanged old factor-reference K book required')
                centers=json.loads(book.read_text(encoding='utf-8-sig'))['centers']
                if len(centers)!=16 or any(len(row)!=4 for row in centers):raise ValueError('Old global16x4 book required')
            result=super()._initialize(*args,**kwargs)
            options=self.postres_options
            if options is not None:
                if self.joint_gain_enabled:raise RuntimeError('Ordinary joint gain ran before postres anchor')
                if list(map(tuple,self.selected_residual_stages))!=options['selected_stages'] or list(map(tuple,self.requested_residual_stages))!=options['requested_stages'] or self.ranking_constant_bytes!=448:
                    raise RuntimeError('Inherited bank changed complete224 ranking/actual43 mask')
                expected=sum(axis=='k' for axis,_,_,_ in options['selected_stages']) if options['enabled'] else 0
                if len(self.postres_statistics)!=expected:raise RuntimeError('Selected K head postres coverage mismatch')
            return result

        def _encode_axis(self,layer,state,axis,selected,source,base):
            result=super()._encode_axis(layer,state,axis,selected,source,base)
            options=self.postres_options
            if axis!='k' or options is None or not options['enabled']:return result
            if self.joint_gain_enabled:raise RuntimeError('Do not define r0 after ordinary NNLS')
            heads=sorted({head for item_axis,head,_ in selected if item_axis=='k'})
            for head in heads:
                if sum(a=='k' and h==head for a,h,_ in selected)!=1 or ('k',head,0) not in selected:
                    raise ValueError('Postres requires exactly one existing first K residual')
                slot=selected.index(('k',head,0))
                factors=self.encoder_factors.get('key_head_factors')
                factor=None if factors is None else factors[layer,head]
                code,gain,alpha,stats=recode_key_postres_direction(source[head].float(),self.book,
                    state.direction_code[0,head],state.direction_gain[0,head],
                    state.depth_residual_code[slot],state.depth_residual_alpha[slot],
                    gain_factor=factor,row_chunk=options['row_chunk'])
                state.direction_code[0,head].copy_(code)
                state.direction_gain[0,head].copy_(gain)
                state.depth_residual_alpha[slot].copy_(alpha)
                entries=[entry for entry in self.depth_distortions if entry['axis']=='k' and entry['layer']==layer and entry['head']==head]
                if len(entries)!=1 or entries[0]['depth']!=1:raise RuntimeError('Expected one existing first-K distortion entry')
                entry=entries[0]
                entry['pre_postres_squared_error_by_depth']=list(entry['squared_error_by_depth'])
                entry['squared_error_by_depth']=[stats['source_base_sse_after'],stats['source_sse_after']]
                entry['postres_direction']=True
                entry['scope']='actual final postres physical J/g/alpha, fixed sign; supplied centered-WHT coordinate error'
                self.postres_statistics.append(dict(layer=layer,head=head,frozen_head_factor_supplied=factor is not None,**stats))
            return result

        def result_metadata(self):
            result=super().result_metadata();options=self.postres_options
            if options is None:return result
            result.update(stage='round8 selected postres direction candidate; unchanged packed reader/owner',
                backend_family='postres',postres_direction=options['enabled'],postres_row_chunk=options['row_chunk'],
                postres_statistics=self.postres_statistics,postres_added_retained_tensor_bytes=0,
                postres_sign_recode=False,postres_mean_unchanged=True,postres_reader_unchanged=True,
                postres_scope='selected existing first-stage K heads only; no missing/depth2 stage and no V intervention',
                postres_norm_relative_tolerance=2**-9,
                postres_metric_scope='actual final centered-WHT reconstruction; no mean/native-bridge or attention/task guarantee',
                postres_source_sha256=source_fingerprints(),key_final_assignment='postres_direction' if options['enabled'] else 'l2')
            if options['enabled']:
                result.update(encoding_assignment='given original l2 J then one full-final-vector constrained cosine sweep with real NNLS direction proposals',
                    key_assignment_semantics='key_assignment=l2 describes initialization; final recoding declared by postres_direction',
                    key_gain_rule='postres-fixed-original-final-norm-effective-coefficients',
                    key_gain_formula='selected first-stage K: existing effective g/alpha jointly scaled to original final norm, stored FP16; Q16(Q16(g*t_J)+alpha*fixed_sign); norm is initialization only; unselected K keeps original norm/fold gain',
                    key_gain_rounding='initial given-J norm->FP16->frozen factor->FP16, original signMAD; subsequent real common scale then FP16 g/alpha, no factor reapplied',
                    joint_gain_scope='ordinary K joint NNLS disabled; real NNLS directions are norm-constrained proposals before actual final-vector guards')
            return result
    return PostresBank


@contextmanager
def scoped_registration(factory,executor,choose):
    original_factory,original_executor=factory.bank_class,executor.bank_class
    factory.bank_class=executor.bank_class=choose
    try:yield
    finally:factory.bank_class,executor.bank_class=original_factory,original_executor


@contextmanager
def backend_registration(family='postres'):
    if family!='postres':raise ValueError('This registration only implements postres')
    import round8_runner  # noqa: F401; preserve original baseline registration.
    import round8_bank
    import factory
    import run_representation_variants as executor
    previous=factory.bank_class;cls=postres_class(round8_bank.Round8Bank)
    def choose(method):return cls if method=='round8_joint' else previous(method)
    with scoped_registration(factory,executor,choose):yield cls


def finalize_run(output,external,sources,config_sha256,declarations):
    output=Path(output)
    if (output/'complete_round8.json').exists() or (output/'complete_postres.json').exists():raise ValueError('Use a new immutable output')
    if source_fingerprints()!=sources or any(sha(path)!=expected for path,expected in external.items()):
        raise RuntimeError('Source/input changed; preserve run and use new output')
    engine,manifest_path=output/'complete.json',output/'manifest.json'
    complete=json.loads(engine.read_text(encoding='utf-8-sig'))
    if complete.get('status')!='COMPLETE' or complete.get('records')!=complete.get('expected') or not complete.get('expected',0):raise RuntimeError('Incomplete producer record coverage')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if manifest.get('config_sha256')!=config_sha256:raise RuntimeError('Producer config differs')
    manifest.setdefault('external_codec_inputs',{}).update(external)
    manifest.update(stage='round8 selected postres direction candidates',backend_family='postres',candidate_subset=True,factorial_complete=False,
        postres_declarations=declarations,postres_source_sha256=sources,round8_engine_complete_requires_postcheck=True,
        executed_candidate_variants=[v['name'] for v in manifest['config']['variants']])
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    records_sha=sha(output/'records.jsonl')
    marker=dict(status='COMPLETE',engine_complete_sha256=sha(engine),manifest_sha256=sha(manifest_path),records_sha256=records_sha,
        encoder_inputs_unchanged=True,source_inputs_unchanged=True,backend_family='postres',candidate_subset=True,factorial_complete=False,postres_source_sha256=sources)
    with (output/'complete_round8.json').open('x',encoding='utf-8') as stream:json.dump(marker,stream,indent=2)
    final=dict(status='COMPLETE',complete_round8_sha256=sha(output/'complete_round8.json'),manifest_sha256=sha(manifest_path),
        records_sha256=records_sha,config_sha256=config_sha256,runner_sha256=sha(__file__),source_sha256=sources,
        backend_family='postres',candidate_subset=True,factorial_complete=False,accepted_only_after_this_marker=True)
    with (output/'complete_postres.json').open('x',encoding='utf-8') as stream:json.dump(final,stream,indent=2)
    return final


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args();config_sha=sha(args.config);config=json.loads(args.config.read_text(encoding='utf-8-sig'))
    declarations,sources=validate_config(config),source_fingerprints()
    if args.plan_only:
        print(json.dumps(dict(status='PLAN_ONLY',backend_family='postres',candidate_subset=True,factorial_complete=False,variants=declarations,
            source_sha256=sources,GPU_executed=False,added_retained_tensor_bytes=0)),flush=True);return
    import round8_runner as original_runner
    external={v[key]:sha(v[key]) for v in config['variants'] for key in original_runner.EXTRA_KEYS if v.get(key)}
    with backend_registration('postres'):
        import run_representation_variants as executor
        executor.main()
    if sha(args.config)!=config_sha:raise RuntimeError('Config changed; preserve run')
    finalize_run(args.output,external,sources,config_sha,declarations)


if __name__=='__main__':main()
