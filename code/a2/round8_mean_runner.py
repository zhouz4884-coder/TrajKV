"""One final-decoded visual K mean-header correction; frozen reader/fields.

This selected-candidate runner deliberately fixes the old Gaussian16 L2/norm,
binary/all-V, full224 ranking/43-stage layout. It accepts arbitrary subsets of
explicit off/on candidates and unchanged Full/R7 controls. No old file changes.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

from round8_field_joint_runner import validate_variant as original_format_validation

SOURCE_FILES = ('round8_mean_runner.py','round8_mean_correction.py','round8_field_joint_runner.py',
    'round8_bank.py','round8_codec.py','round8_attention.py','round8_joint_gain.py','round8_runner.py',
    'round7_runner.py','round6_runner.py','round6_depth_codec.py','round6_depth_bank.py',
    'round7_key_bank.py','round7_key_codec.py','round7_key_attention.py','round7_key_layout.py',
    'round7_key_artifacts.py','metadata_direction_bank.py','masked_bank.py','factory.py',
    'run_representation_variants.py','session.py','session_boundary.py','quantizers.py',
    'affine_rate_bank.py','fast_wht.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fingerprints():
    result = {name:sha(Path(__file__).with_name(name)) for name in SOURCE_FILES}
    result['experiments/fused_progressive.py'] = sha(Path(__file__).resolve().parents[1]/'experiments/fused_progressive.py')
    return result


def validate_variant(variant):
    enabled = variant.get('visual_mean_correction',False)
    if not isinstance(enabled,bool):raise ValueError('visual_mean_correction must be an explicit boolean')
    if variant.get('postres_direction',False) or variant.get('final_ray_mode','off')!='off':
        raise ValueError('First mean control excludes postres/ray interventions')
    if variant.get('joint_gain_refit',False) or variant.get('value_joint_refit',False) or variant.get('key_assignment')=='row_cosine_cd':
        raise ValueError('First mean control excludes K/V joint refit and CD')
    if variant['method']!='round8_joint':
        if enabled:raise ValueError('A Full/R7 control cannot silently ignore mean correction')
        return None
    if 'visual_mean_correction' not in variant:raise ValueError('Every R8 mean candidate needs an explicit on/off flag')
    chunk=variant.get('mean_decode_chunk',4096)
    if not isinstance(chunk,int) or isinstance(chunk,bool) or chunk<1:raise ValueError('Positive integer mean_decode_chunk required')
    original = original_format_validation(variant)
    return dict(enabled=enabled,ranking_bytes=448,requested_stages=original['requested_stages'],
        selected_stages=original['selected_stages'],mean_decode_chunk=chunk,added_persistent_tensor_bytes=0)


def validate_config(config):
    if config.get('backend_family','mean')!='mean':raise ValueError('No mixed mean backend family')
    if not config.get('variants'):raise ValueError('At least one explicit candidate/control required')
    declarations,common = {},None
    for variant in config['variants']:
        if variant['name'] in declarations:raise ValueError('Duplicate variant name')
        options = validate_variant(variant);declarations[variant['name']] = options
        if options is None:continue
        signature = dict(stages=options['requested_stages'],inputs={key:variant.get(key) for key in
            ('direction_book_path','metadata_reference_book_path','metadata_factor_path','round8_factor_policy',
             'budget_bits','depth_policy','cap')})
        if common is not None and common!=signature:raise ValueError('Mean candidates must share full ranking/book/factors/cap/price')
        common = signature
    return declarations


def tensor_inventory(states,book):
    """Encoder-only fingerprints/pointers; no tensor or source rows retained."""
    import torch
    def item(tensor):
        return dict(sha256=hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest(),
            shape=list(tensor.shape),dtype=str(tensor.dtype),data_ptr=tensor.data_ptr(),
            storage_ptr=tensor.untyped_storage().data_ptr(),storage_bytes=tensor.untyped_storage().nbytes())
    rows = [{name:item(value) for name,value in vars(state).items()
        if isinstance(value,torch.Tensor) and name!='direction_mean'} for state in states]
    return dict(book=item(book),non_mean_layers=rows,
        mean_pointers=[state.direction_mean.data_ptr() for state in states],
        mean_shapes=[list(state.direction_mean.shape) for state in states])


def mean_class(base_class):
    from round8_mean_correction import correct_state_mean_inplace
    class AuditedBase(base_class):
        def _initialize(self,*args,**kwargs):
            cfg=type(self).settings;self.mean_options=validate_variant(cfg)
            if self.mean_options is not None:
                book,reference=Path(cfg['direction_book_path']),Path(cfg['metadata_reference_book_path'])
                if sha(book)!=sha(reference):raise ValueError('Unchanged old factor-reference K book required')
                centers=json.loads(book.read_text(encoding='utf-8-sig'))['centers']
                if len(centers)!=16 or any(len(row)!=4 for row in centers):raise ValueError('Old global16x4 book required')
            result=super()._initialize(*args,**kwargs)
            if self.mean_options is not None:
                if self.joint_gain_enabled:raise RuntimeError('Ordinary K joint refit must stay off')
                if (list(map(tuple,self.requested_residual_stages))!=self.mean_options['requested_stages']
                    or list(map(tuple,self.selected_residual_stages))!=self.mean_options['selected_stages']
                    or self.ranking_constant_bytes!=448):raise RuntimeError('Inherited bank changed full224/actual43 layout')
            self.mean_header_owner_before=self.owned()
            self.mean_header_fields_before=tensor_inventory(self.states,self.book)
            return result
    class Corrected(AuditedBase):
        # Same one-pass contract as mean_corrected_bank_class, explicitly
        # forwarding chunk here so a config cannot claim an ignored value.
        def _initialize(self,*args,**kwargs):
            result=super()._initialize(*args,**kwargs)
            self.mean_correction_enabled=self.mean_options['enabled']
            self.mean_decode_chunk=self.mean_options['mean_decode_chunk']
            self.mean_correction_statistics=[]
            if self.mean_correction_enabled:
                for layer,state in enumerate(self.states):
                    self.mean_correction_statistics.append(dict(layer=layer,apply_count=1,actual_decode_chunk=self.mean_decode_chunk,
                        **correct_state_mean_inplace(state,self.book,chunk=self.mean_decode_chunk)))
            return result
        def result_metadata(self):
            result=super().result_metadata()
            result.update(visual_mean_correction=self.mean_correction_enabled,mean_actual_decode_chunk=self.mean_decode_chunk,
                mean_correction_added_persistent_tensor_bytes=0,mean_correction_source_FP16_used=False,
                mean_correction_statistics=self.mean_correction_statistics,mean_correction_serving_FP16_materialization=False,
                mean_correction_scope='one encoder-only correction of existing mean slots; J/g/sign/alpha/V/text/book/reader unchanged')
            if self.mean_correction_enabled:
                result['depth_residual_K_reference']='Original centered-WHT residual fields unchanged; stored mean corrected once after final hard decode'
            return result
    class AuditedMeanBank(Corrected):
        def _initialize(self,*args,**kwargs):
            result=super()._initialize(*args,**kwargs)
            after=tensor_inventory(self.states,self.book);owner=self.owned()
            if after!=self.mean_header_fields_before or owner!=self.mean_header_owner_before:
                raise RuntimeError('Mean correction changed non-mean fingerprint/pointer or actual owner')
            expected=len(self.states) if self.mean_correction_enabled else 0
            if len(self.mean_correction_statistics)!=expected:raise RuntimeError('Mean correction layer coverage differs')
            if any(row['visual_tokens']!=int(self.states[row['layer']].visual_pos.numel())
                or not row['full_visual_mean'] or row['source_FP16_used'] for row in self.mean_correction_statistics):
                raise RuntimeError('Mean correction did not consume all real final packed tokens')
            self.mean_header_integrity=dict(layer_count=len(self.states),
                non_mean_tensor_fingerprints_and_pointers_unchanged=True,mean_pointers_preserved=True,
                actual_owner_before=self.mean_header_owner_before,actual_owner_after=owner,
                actual_owner_unchanged=True,fields_before=self.mean_header_fields_before,fields_after=after)
            return result
        def result_metadata(self):
            result=super().result_metadata()
            result.update(backend_family='mean',mean_header_integrity=self.mean_header_integrity,
                mean_source_sha256=source_fingerprints(),mean_reader_unchanged=True,
                mean_encoding_audit_scope='encoder-only CPU fingerprints/transfers included; no retained FP16 visual cache',
                mean_metric_scope='stored FP16 source-mean target only; rounded query transform and FP32 accumulation remain; task benefit empirical')
            return result
    AuditedMeanBank.__name__='AuditedMeanCorrectedRound8Bank'
    return AuditedMeanBank


@contextmanager
def backend_registration(family='mean'):
    if family!='mean':raise ValueError('This registration only implements mean')
    import round8_runner  # noqa: F401; complete original registration before patch.
    import round8_bank
    import factory
    import run_representation_variants as executor
    import session_boundary
    pf,pe=factory.bank_class,executor.bank_class
    cls=mean_class(round8_bank.Round8Bank)
    classes={cls,round8_bank.Round8Bank,pf('full'),pf('round7_key')}
    settings=[(item,'settings' in item.__dict__,getattr(item,'settings',None)) for item in classes]
    previous_bank=session_boundary.ProgressiveBank
    def choose(method):return cls if method=='round8_joint' else pf(method)
    factory.bank_class=executor.bank_class=choose
    try:yield cls
    finally:
        factory.bank_class,executor.bank_class=pf,pe
        session_boundary.ProgressiveBank=previous_bank
        for item,owned,value in settings:
            if owned:item.settings=value
            elif 'settings' in item.__dict__:delattr(item,'settings')


def external_fingerprints(config):
    paths={config['arguments']['samples']}
    for variant in config['variants']:
        for key,value in variant.items():
            if key.endswith('_path') and isinstance(value,str) and value:paths.add(value)
            if key.endswith('_paths') and isinstance(value,list):paths.update(value)
    model=config['arguments'].get('model')
    if model:paths.add(str(Path(model)/'config.json'))
    return {path:sha(path) for path in sorted(paths)}


def finalize_run(output,external,sources,config_sha,declarations):
    output=Path(output)
    if (output/'complete_round8.json').exists() or (output/'complete_mean.json').exists():raise ValueError('Use new immutable output')
    if source_fingerprints()!=sources or any(sha(path)!=value for path,value in external.items()):raise RuntimeError('Source/input changed; preserve run')
    engine,mp=output/'complete.json',output/'manifest.json'
    complete=json.loads(engine.read_text(encoding='utf-8-sig'));manifest=json.loads(mp.read_text(encoding='utf-8-sig'))
    if complete.get('status')!='COMPLETE' or complete.get('records')!=complete.get('expected') or not complete.get('expected',0):raise RuntimeError('Incomplete producer coverage')
    if manifest.get('config_sha256')!=config_sha:raise RuntimeError('Producer consumed different config')
    config=manifest['config']
    if validate_config(config)!=declarations:raise RuntimeError('Producer declaration differs')
    if manifest.get('samples_sha256')!=external[config['arguments']['samples']]:raise RuntimeError('Producer consumed different samples')
    rows=[json.loads(line) for line in (output/'records.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    expected={(sid,name) for sid in manifest['selected_ids'] for name in declarations}
    actual=[(row['id'],row['variant']) for row in rows]
    if len(rows)!=complete['records'] or len(actual)!=len(set(actual)) or set(actual)!=expected:raise RuntimeError('Incomplete/duplicate selected ID x candidate records')
    by_name={v['name']:v for v in config['variants']};coverage=[]
    for row in rows:
        variant=by_name[row['variant']];options=declarations[row['variant']]
        if row.get('variant_config')!=variant or row.get('method')!=variant['method']:raise RuntimeError('Record consumed different variant')
        if options is None:continue
        integrity=row.get('mean_header_integrity',{})
        stats=row.get('mean_correction_statistics',[])
        if (row.get('visual_mean_correction') is not options['enabled'] or row.get('mean_source_sha256')!=sources
            or row.get('mean_correction_added_persistent_tensor_bytes')!=0
            or row.get('mean_correction_source_FP16_used') is not False or row.get('mean_reader_unchanged') is not True
            or integrity.get('layer_count')!=28 or not integrity.get('non_mean_tensor_fingerprints_and_pointers_unchanged')
            or not integrity.get('mean_pointers_preserved') or not integrity.get('actual_owner_unchanged')
            or integrity.get('actual_owner_before')!=integrity.get('actual_owner_after')
            or integrity.get('fields_before')!=integrity.get('fields_after')
            or row.get('mean_actual_decode_chunk')!=options['mean_decode_chunk']
            or len(stats)!=(28 if options['enabled'] else 0)):
            raise RuntimeError('Mean record lacks 28-layer unchanged-field/owner evidence')
        if options['enabled'] and (sorted(r['layer'] for r in stats)!=list(range(28)) or any(
            r['visual_tokens']!=row['visual_tokens'] or not r['full_visual_mean'] or r['source_FP16_used']
            or not r['mean_storage_pointer_preserved'] or r['added_persistent_tensor_bytes']!=0
            or r.get('apply_count')!=1 or r.get('actual_decode_chunk')!=options['mean_decode_chunk'] for r in stats)):
            raise RuntimeError('Mean record lacks complete all-real-token coverage')
        coverage.append(dict(id=row['id'],variant=row['variant'],enabled=options['enabled'],layers=28,real_visual_tokens=row['visual_tokens']))
    manifest.setdefault('external_codec_inputs',{}).update(external)
    manifest.update(stage='round8 final-decoded visual mean-header candidates',backend_family='mean',candidate_subset=True,
        factorial_complete=False,mean_declarations=declarations,mean_source_sha256=sources,
        round8_engine_complete_requires_postcheck=True,executed_candidate_variants=list(declarations))
    mp.write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
    marker=dict(status='COMPLETE',engine_complete_sha256=sha(engine),manifest_sha256=sha(mp),records_sha256=sha(output/'records.jsonl'),
        encoder_inputs_unchanged=True,source_inputs_unchanged=True,backend_family='mean',candidate_subset=True,
        factorial_complete=False,mean_source_sha256=sources)
    with (output/'complete_round8.json').open('x',encoding='utf-8') as stream:json.dump(marker,stream,indent=2)
    final=dict(status='COMPLETE',complete_round8_sha256=sha(output/'complete_round8.json'),manifest_sha256=sha(mp),
        records_sha256=sha(output/'records.jsonl'),config_sha256=config_sha,runner_sha256=sha(__file__),source_sha256=sources,
        external_inputs_sha256=external,backend_family='mean',candidate_subset=True,factorial_complete=False,
        mean_record_coverage=coverage,added_persistent_tensor_bytes=0,accepted_only_after_this_marker=True)
    with (output/'complete_mean.json').open('x',encoding='utf-8') as stream:json.dump(final,stream,indent=2,allow_nan=False)
    return final


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args();config_sha=sha(args.config);config=json.loads(args.config.read_text(encoding='utf-8-sig'))
    declarations,sources=validate_config(config),source_fingerprints()
    if args.plan_only:
        print(json.dumps(dict(status='PLAN_ONLY',backend_family='mean',candidate_subset=True,factorial_complete=False,
            variants=declarations,source_sha256=sources,GPU_executed=False,added_persistent_tensor_bytes=0)),flush=True);return
    external=external_fingerprints(config)
    with backend_registration('mean'):
        import run_representation_variants as executor
        executor.main()
    if sha(args.config)!=config_sha:raise RuntimeError('Config changed; preserve run')
    finalize_run(args.output,external,sources,config_sha,declarations)


if __name__=='__main__':main()
