"""Independent postcal artifact and complete-owner gates; imports no GPU code."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import struct
from pathlib import PurePosixPath

EXTRA_BYTES=3*28*4*4
def require(value,message):
    if not value:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def local_path(value):
    # Archive validators may supply sealed virtual members with these methods.
    return value if hasattr(value,'read_text') and hasattr(value,'parent') else Path(value)
def table_values_sha256(values):
    flat=[float(v) for axis in values for layer in axis for v in layer]
    require(len(flat)==336,'Exactly336 postcal FP32 parameters required')
    return hashlib.sha256(struct.pack('<336f',*flat)).hexdigest()
def identity_table():return [[[1.]*4 for _ in range(28)],[[1.]*4 for _ in range(28)],[[0.]*4 for _ in range(28)]]
def priority_bytes(variant):
    stages=variant.get('selected_stages');require(isinstance(stages,list) and len(stages)==224,'Full224 priority required')
    tuples=[tuple(x) for x in stages]
    require(len(set(tuples))==224 and all(len(x)==4 and x[0] in ('k','v') and 0<=x[1]<28 and 0<=x[2]<4 and x[3]==0 for x in tuples),'Full first-stage K/V priority required')
    require(stages[42]==['v',27,3,0] and sum(x[0]=='k' for x in stages[:42])==23 and sum(x[0]=='v' for x in stages[:42])==19,'Frozen42 prefix must preserve23K/19V and drop only V27h3')
    return struct.pack('<224H',*(((int(a=='v')*28+l)*4+h)*2+s for a,l,h,s in tuples))
def validate_variant(v):
    mode=v.get('postcal_mode','off')
    if v.get('method')!='round8_joint':
        require(mode=='off','An external control cannot silently apply postcal');return None
    require(mode in ('identity','learned'),'Postcal R8 arm requires explicit identity/learned mode')
    for key,want in [('key_assignment','l2'),('key_amplitude','norm'),('key_encoding','reencode'),('value_encoding','binary'),('v_group_size','all'),('round8_factor_policy','inherit_from_reference_book'),('residual_layout_reference','actual')]:
        require(v.get(key)==want,'Unchanged original format required: '+key)
    require(v.get('res_stages_limit')==42 and not v.get('joint_gain_refit',False) and not v.get('value_joint_refit',False)
        and not v.get('postres_direction',False) and v.get('final_ray_mode','off')=='off','Postcal requires plain original R42 initialization')
    require(not any(v.get(k) for k in ('key_metric_path','functional_v_factor_path','encoder_book_path','value_joint_book_path')),'No extra encoder, metric or V dictionary')
    packed=priority_bytes(v)
    if mode=='learned':require(v.get('postcal_factor_path') and v.get('postcal_train_complete_path'),'Learned mode requires final training artifacts')
    else:require(not v.get('postcal_factor_path') and not v.get('postcal_train_complete_path'),'Identity arm must not ignore trained artifacts')
    return dict(mode=mode,extra_bytes=EXTRA_BYTES,priority_sha256=hashlib.sha256(packed).hexdigest(),stages=42,K_stages=23,V_stages=19)

def verify_factor_artifact(factor_path,marker_path,variant,*,resolver=None,reader=None,hasher=None,shared_fields_sha256=None):
    """Require the FINAL training graph before a learned table can be consumed."""
    resolve,load,hashfile=resolver or local_path,reader or read,hasher or sha
    factor_path,marker_path=resolve(factor_path),resolve(marker_path);data,marker=load(factor_path),load(marker_path)
    require(marker.get('status')=='COMPLETE' and marker.get('steps')==200,'Final200-step training marker required')
    require(marker.get('factors_sha256')==hashfile(factor_path),'Final training factor SHA mismatch')
    root=marker_path.parent;bound={}
    def member(name):
        parsed=PurePosixPath(str(name).replace('\\','/'))
        require(not parsed.is_absolute() and '..' not in parsed.parts,'Training canonical file escaped its run')
        return root/str(parsed)
    for key,default in (('manifest','manifest.json'),('schedule','schedule.json')):
        name=marker.get(key+'_file',default);path=member(name)
        require(marker.get(key+'_sha256')==hashfile(path),'Training '+key+' SHA mismatch')
        bound[name]=hashfile(path)
    for mapping_key in ('files_sha256','canonical_sha256'):
        mapping=marker.get(mapping_key,{})
        require(isinstance(mapping,dict),'Training file hash graph must be a mapping')
        for name,digest in mapping.items():
            path=member(name)
            require((hasher or sha)(path)==digest,'Training ledger SHA mismatch: '+name);bound[name]=digest
    require(data.get('schema')=='trajkv.round8.postcal-factors.v1' and data.get('encoder_constant_bytes')==EXTRA_BYTES
        and data.get('fixed_stage_count')==42,'Wrong learned postcal artifact schema/price/stage count')
    values=data.get('post_factors');require(isinstance(values,list) and len(values)==3 and all(len(x)==28 and all(len(row)==4 for row in x) for x in values),'FP32[3,28,4] factor table required')
    require(all(math.isfinite(float(v)) and (a==2 or float(v)>0) for a,axis in enumerate(values) for row in axis for v in row),'Finite positive K/V factors and finite shift required')
    require(data.get('old_f0_sha256')==(hasher or sha)(resolve(variant['metadata_factor_path'])) and data.get('encoder_book_sha256')==(hasher or sha)(resolve(variant['direction_book_path'])),'Training old initialization inputs differ')
    require(data.get('full_priority_sha256')==hashlib.sha256(priority_bytes(variant)).hexdigest(),'Training full ranking differs')
    shared=shared_fields_sha256 or (hasher or sha)(resolve(Path(__file__).with_name('round8_postcal_fields.py')))
    sources=marker.get('source_sha256',{});require(isinstance(sources,dict) and sources.get('round8_postcal_fields.py')==shared,'Training/deployment shared physical transform source differs')
    if 'source_sha256' in data:require(data['source_sha256']==sources,'Factor and final training source inventories differ')
    if 'steps' in data:require(data['steps']==200,'Factor table training step count differs')
    return dict(values=values,factors_sha256=(hasher or sha)(factor_path),train_complete_sha256=(hasher or sha)(marker_path),
        table_values_sha256=table_values_sha256(values),bound_training_files_sha256=bound,training_source_sha256=sources)

def audit_owner(record,variant,external,cache):
    """Recount actual table bytes on top of the frozen R8 field-shape oracle."""
    import round8_quality_owner as base
    if variant.get('postcal_mode','off') not in ('identity','learned'):
        return base.audit_owner(record,variant,external,cache)
    options=validate_variant(variant)
    if options is None:return base.audit_owner(record,variant,external,cache)
    require(record.get('backend_family')=='postcal' and record.get('postcal_mode')==options['mode'],'Postcal producer mode differs')
    require(record.get('postcal_parameter_shape')==[3,28,4] and record.get('postcal_added_encoder_constant_bytes')==EXTRA_BYTES
        and record.get('postcal_added_retained_tensor_bytes')==EXTRA_BYTES,'Postcal actual FP32 table omitted/doubled')
    require(record.get('postcal_J_sign_mean_Vcode_unchanged') is True and record.get('postcal_reader_unchanged') is True,'Postcal format/reader contract missing')
    proof=verify_factor_artifact(base.local_path(variant['postcal_factor_path']),base.local_path(variant['postcal_train_complete_path']),
        {**variant,'metadata_factor_path':base.local_path(variant['metadata_factor_path']),'direction_book_path':base.local_path(variant['direction_book_path'])}) if options['mode']=='learned' else None
    expected=proof['values'] if proof else identity_table()
    require(record.get('postcal_factor_file_sha256')==(proof['factors_sha256'] if proof else None)
        and record.get('postcal_train_complete_sha256')==(proof['train_complete_sha256'] if proof else None),'Record learned artifact/final-marker binding differs')
    require(record.get('postcal_table_values_sha256')==table_values_sha256(expected),'Recorded post table numerical identity differs')
    clone=deepcopy(record);nv=int(record['visual_tokens']);denominator=record['owned_initial']['visual_fp16_denominator']
    for owner_key in ('owned_initial','owned_final'):
        if owner_key in clone:
            for key in ('encoder_constant_bytes','complete_visual_owner_bytes','total_unique'):
                if key in clone[owner_key]:clone[owner_key][key]-=EXTRA_BYTES
    for key in ('encoder_calibration_parameters_bytes','depth_residual_fixed_complete_bytes','depth_residual_complete_bytes'):
        clone[key]-=EXTRA_BYTES
    original=clone['depth_residual_complete_bytes']
    clone['depth_residual_actual_budget_bits']=clone['sparse_residual_actual_budget_bits']=16*original/denominator
    result=base.audit_owner(clone,variant,external,cache)
    total=original+EXTRA_BYTES;old43=original+18*nv
    require(record['owned_initial']['complete_visual_owner_bytes']==record['depth_residual_complete_bytes']==total
        and record.get('postcal_original_R42_complete_bytes')==original and record.get('postcal_old43_complete_bytes')==old43
        and record.get('postcal_complete_le_old43') is True and total<=old43,'Postcal current complete owner exceeds originalR43')
    require(math.isclose(record['depth_residual_actual_budget_bits'],16*total/denominator,rel_tol=1e-11,abs_tol=1e-10),'Postcal actual bits differ')
    result.update(encoder_parameters_bytes=result['encoder_parameters_bytes']+EXTRA_BYTES,complete_visual_owner_bytes=total,
        actual_visual_bits=16*total/denominator,postcal_added_bytes=EXTRA_BYTES,original_R42_bytes=original,old_R43_bytes=old43,
        postcal_complete_le_old43=True,owner_scope='Frozen actualR8 shape recount plus one persistent336FP32 postcal table; old f0/full224 ranking/policy retained')
    return result
