"""One explicit visual-owner account for second-cycle evidence tables."""
VISUAL_KEYS=('visual_base','visual_residual','selector_metadata','hadamard_constant','visual_position','anchor_visual_auxiliary',
             'axis_packed_reference','calib_packed_reference','hybrid_packed_reference','kvarn_packed_reference','rotation_packed_reference')
def visual_owner(record):
 owned=record['owned_initial'];base=sum(owned.get(k,0) for k in VISUAL_KEYS)
 # Historical KVarN mechanism ports cache a global float32 Hadamard matrix in
 # the reference module, outside bank.owned(). Charge it once explicitly.
 shared_hadamard=65536 if record['method'].startswith('kvarn') and not owned.get('hadamard_constant') else 0
 return base+shared_hadamard,shared_hadamard
def role(record):
 if record['method']=='calib_packed':return 'actual packed runtime'
 if record.get('oracle'):return 'FP16 oracle diagnostic'
 if record.get('task_accuracy_eligible') is False:return 'local diagnostic only'
 if record.get('materialized_quality_port') or record['method'].startswith(('calib','kvarn')) or record['method'] in ('axisdiag','hybrid','rotk_port'):return 'materialized quality prototype'
 return 'FP16 runtime' if record['method'] in ('full','anchor_full') else 'actual packed runtime'
