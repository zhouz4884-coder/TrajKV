"""Portable cold primary plus explicitly sealed follow-up verification.

Only archived text/source/scalar files are opened. Private native/PT/W_O/model
payloads are intentionally absent. No network, GPU inference or Git operations.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path,PurePosixPath
import sys
import tempfile
import zipfile
sys.dont_write_bytecode=True
R7_ROLE_PINS={'a2/round7_key_bank.py': '38b6e0ea2486a900e404fa6347c91972198795f8972dc7ab2bf90527fbc1cdc5', 'a2/round7_key_attention.py': 'b054e8629083b7685ad9e02289b500008b45fbeadbd2bc48455f91d0b6a728a0', 'a2/round7_key_codec.py': '80c2c16220097242b25d3f843a3b57569590131c02618e766e35f0f0791ab651', 'a2/round7_key_layout.py': '7b9773f65103062dd51c8ccc64d9b5047f88112d951b6eb10e7938f8fc3f5a04', 'a2/round7_key_artifacts.py': 'f429aa502dae27b7616e8dfca63ac6d498d16f18d8bf2db4ad674b5039760485', 'a2/round6_depth_codec.py': '333ef93165fd786bc40015f6f8c775ce3de20bb98da26ffa6555c38cba5f12d4', 'a2/fast_wht.py': '9dc73998f66d5b7bafc96a5aa3db57974d1ca8e604e432e7539625f104d90ee3', 'a2/quantizers.py': '314bc458c9c2d84579b2a02cb8a45908e0099b613368be40a07669ca5d273e38', 'a2/affine_rate_bank.py': '8090240cd1ee5edde07d1252c41f4dfcaa8aebf167a2b7d890d37ae57edcd2c0', 'a2/masked_bank.py': '7c313fddfdadc81939235b0a4fd843540961f8841c69d7400101b33a66737d9d', 'a2/metadata_direction_bank.py': 'bea958a75c8df84702b20ba32836ae6f4f7ffaae8f6a1024ea6de3c18659ff61', 'experiments/bank.py': 'f1ab22bf04f592263246bac82c415fd9a2b8c6adb4d813a5b46d056f08987ee5', 'experiments/fused_progressive.py': 'd0130e31d055f2c2ced6de1cc26a5cbfa99eea0642eb60ce5f4b6c359cd6520c', 'experiments/check_fused_progressive.py': 'b7578afafcb4bc522bc0742b99e4ee5ab49307fb120b3475dc553900076dac55', 'code/bytes.py': '00625b902a29027075c13d64ca9019753e4aa153215aea6f349d25902a848115', 'scripts/p185_model_state.py': '437dbb5e9bd34e699ddc1f589834d207248074e214dfa821143781cd4fe3c4af', 'scripts/p185_single_state.py': 'ca4bce75c1e80c0a99eb18451323f2d0c2a07a8c707a3238f1a4e8d5d9114b3c', 'scripts/p185_segmented_kernel.py': 'bf09b996f098aea5027f25de54b50aef78f305225de105d9b8c9343751eb3f64', 'scripts/p176_kivi_codec.py': '1c278ebd1f648170696208ffed89a01e96960c18670610948d742262ad6b7cbb', 'scripts/p201_refinement.py': 'a1b8b84b9dbc655c21ead0bffa032be1ae4af97b5da4768112fe2a6106e58afb', 'scripts/p202_refinement.py': '292aca3d0f793c164b1d4204af21573cd4092597eb3f25055ded6d879afa04aa', 'kivi_source': '3f1140b0faff2cb524a3eb684167e403dffd488477b301826497462fd36fe540', 'a2/round6_depth_bank.py': '1d423bf03cdc76c0650776230d6a1093710e9c9fb8ac2cd84974d791aa65c6f7', 'a2/round6_depth_attention.py': 'f96277c726c39f2880e8618525b4bdcbac0049e68e7c4d3e2eb699fa24dfd945'}

DTYPE_BYTES={'torch.float16': 2, 'torch.bfloat16': 2, 'torch.float32': 4, 'torch.float64': 8, 'torch.int64': 8, 'torch.int32': 4, 'torch.int16': 2, 'torch.int8': 1, 'torch.uint8': 1, 'torch.bool': 1}


def require(value,message):
    if not value:raise ValueError(message)


def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def sha_stream(stream):
    h=hashlib.sha256()
    for part in iter(lambda:stream.read(1024**2),b''):h.update(part)
    return h.hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:return sha_stream(stream)


def io_path(value):
    path=Path(value).resolve();prefix=chr(92)*2+'?'+chr(92)
    return Path(prefix+str(path)) if os.name=='nt' and not str(path).startswith(prefix) else path


def r7_role_path(cfg,role):
    if role=='kivi_source':return cfg['kivi_source']
    if role.startswith('a2/'):return cfg['legacy_a2']+'/'+role[3:]
    if role.startswith('experiments/'):return cfg['legacy_experiments']+'/'+role[len('experiments/'):]
    if role.startswith('scripts/'):return cfg['legacy_scripts']+'/'+role[len('scripts/'):]
    if role=='code/bytes.py':return cfg['legacy_a2'].rsplit('/',1)[0]+'/bytes.py'
    raise ValueError('Unknown frozen role')


def r7_verify_executed_roles(cfg,actual):
    for role,digest in R7_ROLE_PINS.items():
        path=r7_role_path(cfg,role)
        if path in actual:require(actual[path]==digest,'Actual executed legacy role/source differs: '+role)
    required=('a2/round7_key_bank.py','a2/round7_key_attention.py') if cfg['variant']['name']=='r7_joint_k1_gt_cond' else ('a2/round6_depth_bank.py','a2/round6_depth_attention.py')
    require(all(actual.get(r7_role_path(cfg,role))==R7_ROLE_PINS[role] for role in required),'Actual frozen bank/reader roles missing')


def r7_verify_record_role_pins(cfg,row):
    value=row['r7_bridge']['verified_legacy_source_pins']
    allowed=dict(R7_ROLE_PINS)
    if cfg['variant']['name']=='r7_joint_k1_gt_cond':
        allowed.pop('a2/round6_depth_bank.py');allowed.pop('a2/round6_depth_attention.py')
    expected={r7_role_path(cfg,role):{'role':role,'sha256':digest} for role,digest in allowed.items()}
    require(value==expected,'Per-record frozen source role/actual-path admission differs')


def physical_complete_bytes(raw, spec):
    owner = raw['owner_before_suffix']; tensors = owner['tensors']
    require(tensors, 'Missing physical owner table')
    for name, item in tensors.items():
        require(math.prod(item['shape'])*DTYPE_BYTES[item['dtype']] == item['logical_bytes'] <= item['storage_bytes'], 'Physical capacity/shape differs')
        require(item['charged_bytes'] in (0, item['storage_bytes']), 'Invalid unique physical charge')
        if item['shared_with'] is None:
            require(item['charged_bytes'] == item['storage_bytes'], 'Unshared physical owner was not fully charged')
        else:
            first = tensors[item['shared_with']]
            require(first['shared_with'] is None and first['charged_bytes'] == first['storage_bytes'] and
                    item['charged_bytes'] == 0 and item['device'] == first['device'] and item['storage_bytes'] == first['storage_bytes'], 'Unbound, cyclic or indirect shared physical owner')
    require(sum(item['charged_bytes'] for item in tensors.values()) == owner['total_bytes'], 'Physical owner charge sum differs')
    denominator = spec['layers']*2*spec['kv_heads']*raw['actual_visual_token_count']*spec['head_dim']*2
    require(raw['full_kv_byte_denominator'] == denominator, 'Native geometry denominator differs')
    if owner['accounting_scope'] == 'native_full_allocation_equals_disjoint_visual_plus_text_regions':
        require(all(item['charged_bytes'] == item['storage_bytes'] == item['logical_bytes'] for name, item in tensors.items() if name.startswith('layers.')), 'Native KV aliases or hides backing capacity')
        require(owner['stored_visual_payload_bytes'] == denominator, 'Native disjoint visual charge differs')
        actual = denominator+sum(item['charged_bytes'] for name, item in tensors.items() if name.startswith('common.'))
    else:
        require(owner['accounting_scope'] == 'global_unique_storages_across_all_layers_and_shared_constants', 'Unknown owner accounting scope')
        actual = sum(item['charged_bytes'] for name, item in tensors.items() if '.visual.' in name or name == 'original_visual_positions' or name.startswith('common.'))
    require(actual == raw['complete_visual_owner_bytes'] == raw['stored_visual_KV_bytes'] == owner['complete_visual_owner_bytes'] > 0, 'Complete visual bytes differ from physical owners')
    require(owner['total_bytes'] == actual+owner['text_kv_bytes']+owner['text_positions_bytes']+owner.get('other_persistent_metadata_bytes', 0), 'Physical owner component balance differs')
    return actual


def verify_files(root,archive=None):
    root=io_path(root);index=load(root/'PUBLIC_CONTENT_REFS.json');entries=index['files']
    require(index['schema'] in ('baseline_matrix.public_content_refs.v2','baseline_matrix.public_content_refs.v3'),
            'Expected source/public identity schema')
    require(index['completion_scope']=='root_sealed_research_stage_only' and index['whole_Round10_goal_complete_claimed'] is False,'Unsupported whole-library completion claim')
    total=0;line_count=0
    for name,entry in entries.items():
        path=root/name;require(path.resolve().is_relative_to(root),'Indexed path escapes public namespace')
        require(path.stat().st_size==entry['public_bytes'] and sha(path)==entry['public_sha256'],'Public file identity mismatch: '+name)
        lf=hashlib.sha256();blob=hashlib.sha1(f"blob {entry['git_lf']['bytes']}\0".encode());size=0
        with path.open('rb') as stream:
            pending=b''
            for piece in iter(lambda:stream.read(1024**2),b''):
                piece=pending+piece;pending=b''
                if path.suffix!='.npy' and piece.endswith(b'\r'):piece,pending=piece[:-1],b'\r'
                if path.suffix!='.npy':piece=piece.replace(b'\r\n',b'\n')
                lf.update(piece);blob.update(piece);size+=len(piece)
            if pending:lf.update(pending);blob.update(pending);size+=len(pending)
        require(size==entry['git_lf']['bytes'] and lf.hexdigest()==entry['git_lf']['sha256'] and blob.hexdigest()==entry['git_lf']['blob_oid_sha1'],'Git LF/blob identity mismatch: '+name)
        if entry['transformation']=='EXACT':require(entry['source_sha256']==entry['public_sha256'],'False exact-source assertion')
        if entry.get('line_identities_file'):
            side=entry['line_identities_file'];require(side in entries,'Missing line identity sidecar')
            count=0
            with path.open('rb') as data,(root/side).open(encoding='utf-8') as identities:
                for count,line in enumerate(data,1):
                    identity=json.loads(identities.readline());obj=json.loads(line)
                    require(identity['line']==count and identity['id']==obj.get('id') and identity['public_sha256']==hashlib.sha256(line).hexdigest(),'Line identity mismatch: '+name)
                require(not identities.readline(),'Extra line identities')
            require(count==entry['line_count'],'Line count mismatch');line_count+=count
        total+=entry['public_bytes']
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(actual==set(entries)|{'PUBLIC_CONTENT_REFS.json'},'Unindexed or missing public files')
    for group in index.get('supporting_evidence_groups',[]):
        require(group['not_part_of_primary_or_auxiliary_quality_QA'] is True,'Supporting controls mixed into quality QA counts')
        for row in group['files']:
            require(row['file'] in entries and entries[row['file']]['source_sha256']==row['source_sha256'],
                    'Supporting original evidence identity differs')
    if archive:
        prefix=index['namespace']+'/'
        with zipfile.ZipFile(archive) as zipped:
            require(set(zipped.namelist())=={prefix+p for p in entries}|{prefix+'PUBLIC_CONTENT_REFS.json'},'Archive member coverage mismatch')
            require(zipped.read(prefix+'PUBLIC_CONTENT_REFS.json')==(root/'PUBLIC_CONTENT_REFS.json').read_bytes(),'Archive content index mismatch')
            for name,entry in entries.items():
                with zipped.open(prefix+name) as stream:require(sha_stream(stream)==entry['public_sha256'],'Archive body mismatch: '+name)
    return root,index,entries,total,line_count

R10_DRIVER='22b5e45152ea9be982ae7ba77950ed09bd065ae574b58b80725beb0983c54e96'
R10_FACTORY='4c25f1bdd2174e07edac37f522e54005c0139e5e00103d42975bfaf3210de05c'
R10_CALIBRATOR='abbc45f07e5899ff6018e0a22c5bba94c8764b03cd518e764de1949d9df87a8a'
DOMAINS=('textvqa','infovqa','docvqa','chartqa')
MODES=('FP16','R7low','R7high','LS-FZ','LS-RE','FB0','FB-W')
COUNTS={'textvqa':64,'infovqa':64,'docvqa':16,'chartqa':16}
MIDDLE_MODES=('V128-R43','AddOldOrder')
MIDDLE_DRIVER='6dcf3fa5865915271b7c7074bc987aee74625ce41cc9d70ac4c56e268e7f86ac'
MIDDLE_FACTORY='8928db801032c4d104debf1d361fb1275fd12ce957c61df541440c92361c8bbb'
EXTENSION_DRIVER='83b8c5003f891b92cd33ed7d3387852c50090709c7f5a0bfd0196f02d01007f1'
KRA_DRIVER='0980c601a8aa343877880163169a76acbc0b147920eafef28f68d6d6386df982'
KRA_FACTORY='ede14d949ff905a15b25d83a928e8a79352065bc9850cc9eaf5759acff75503d'
KRA_CALIBRATOR='4d411ff7275cc891b539d4310ab2933ed0bcced1cf841151cc0d97f7f9d90235'
KRA_METRIC='cb65703f1df7e0f8b0b88340ef3086834d6bdb36498d07d4a2acbb1572d82ee7'
KRA_EXTENSION_DRIVER=None
KRA_EXTENSION_HARVEST=None
HARVEST_PINS={'middle_development':('ops/round10_middle_harvest_v2.py','a26076dcaa0d1d9e6073042a11cc8ccb7629990b5b1e8a99432a75aa9fc80aac'),
              'extension_retest':('ops/round10_extension_harvest_v1.py','2d5e77935b36dc529e4da337046556cba0eb20618ee7ef6fd9eeef2d84dda5ac'),
              'kra_development':('ops/round10_kra_harvest_v1.py','4014a59966f773d848cfa9e00550b02cf286e89123cc98542cfa9e522c5e1096')}
EXTENSION_COUNTS={'textvqa':64,'infovqa':64,'docvqa':48,'chartqa':48}


def middle_physical_guard(row, mode, actual):
    n=row['actual_visual_token_count'];require(n>0 and n%128==0,'middle actual group geometry differs')
    b0=4590*n+89733;v128=5038*n+32389;q=(448*n-57344)//(18*n)
    expected=v128 if mode=='V128-R43' else b0+18*n*q
    require(q>=0 and actual==expected and 0<=v128-(b0+18*n*q)<18*n,'middle actual complete owner formula differs')
    bridge=row['r7_bridge'];d=bridge['round10_middle_candidate'];stages=bridge['selected_stages']
    ranking=bridge['variant']['selected_stages'];old=ranking[:43]
    require(len(ranking)==224 and all(s[3]==0 for s in ranking) and
            bridge['original_selected_stages']==old and len(old)==43 and
            sum(s[0]=='k' for s in old)==23 and sum(s[0]=='v' for s in old)==20,'middle original43/ranking differs')
    selected=old if mode=='V128-R43' else ranking[:43+q]
    require(stages==selected and bridge['selected_stage_count']==len(selected) and
            len({tuple(s) for s in stages})==len(stages),'middle actual compact selection differs')
    require(d['mode']==mode and d['source_sha256']==MIDDLE_FACTORY and
            d['after_unique_capacity_bytes']==actual and d['before_unique_capacity_bytes']==b0 and
            d['actual_extra_persistent_bytes']==actual-b0 and d['new_persistent_constant_bytes']==0 and
            d['global_lookup_bytes']==1792 and d['original_full_ranking_bytes_reused']==448 and
            d['policy_descriptor']['bytes']==5 and d['actual_compact_per_layer_residual_pools'] is True and
            d['mask_over_large_pool'] is False and d['native_FP16_source_or_workspace_retained'] is False,
            'middle source/persistent storage evidence differs')
    require(d['V_group_size']==(128 if mode=='V128-R43' else n) and
            d['original_K_base_fields_unchanged'] is True and d['original_K23_code_alpha_unchanged'] is True,
            'middle base grouping/K preservation differs')
    if mode=='V128-R43':
        require(d['original_V20_reencoded'] is True and d['original_V0_factors_folded_exactly_once'] is True and
                d['reader_dispatch_changed'] is True,'V128 actual fold/reencode/dispatch contract differs')
    else:
        require(d['original43_code_alpha_bitwise_unchanged'] is True and d['original_V_base_fields_unchanged'] is True,
                'AddOldOrder original code/alpha/base preservation differs')
    recorded=row['round10_budget_vs_original_R7_low']
    require(recorded['candidate_complete_visual_bytes']==expected and recorded['actual_complete_visual_bytes']==actual and
            recorded['extra_old_order_stages']==q and recorded['V128_minus_AddOldOrder_bytes']==v128-(b0+18*n*q) and
            recorded['same_complete_bytes_pair']==(v128==b0+18*n*q),'stored middle physical guard differs')


def kra_physical_guard(row,actual):
    n=row['actual_visual_token_count'];b0=4590*n+89733
    b=row['r7_bridge'];d=b['round10_kra_candidate'];stages=b['selected_stages']
    require(actual==b0+5980 and stages==b['variant']['selected_stages'][:43] and b['selected_stage_count']==43 and
            sum(s[0]=='k' for s in stages)==23 and sum(s[0]=='v' for s in stages)==20,'KRA actual original43/byte formula differs')
    require(d['mode']=='KRA1' and d['source_sha256']==KRA_FACTORY and d['before_unique_capacity_bytes']==b0 and
            d['after_unique_capacity_bytes']==actual and d['new_persistent_constant_bytes']==5980 and
            d['actual_extra_persistent_bytes']==5980 and d['unit_order23']==[s for s in stages if s[0]=='k'],
            'KRA metric source/order/storage differs')
    require(all(d[k] is True for k in ('original_K_base_fields_unchanged','original_V_base_fields_unchanged',
            'all_residual_sign_codes_unchanged','original_V20_alpha_unchanged','original_lookup_ranking_policy_unchanged',
            'reader_unchanged','zero_gamma_is_original_alpha_bitwise_identity')) and
            d['native_FP16_source_or_workspace_retained'] is False and d['alpha_write_dtype']=='float16',
            'KRA changed a fixed field or retained native/workspace')
    require(d['actual_H_after_worse_tokens']==0,'KRA selected deployed alpha worsened the declared coordinate metric')


def verify_kra_calibration(root,supplement,entries,source,original,under,recompute):
    import numpy as np
    c=supplement['kra_calibration'];out={'disposition':c['disposition'],
        'private_fit48_qrot_PCA_replayed':False,'private_teacher_logits_or_GPU_replayed':False}
    if c['disposition']!='CLOSED_COMPLETE':
        require(c['disposition'] in ('STOPPED','NOT_RUN') and c.get('reason'),'invalid unrun KRA calibration')
        e=c['disposition_evidence'];original(source(e['path']),e['sha256']);return out
    require(all(p is not None for p in (KRA_CALIBRATOR,KRA_METRIC,KRA_FACTORY)),'KRA calibrated source epochs not sealed')
    original('ops/round10_kra_fetch_calibration_v1.py','4aff5ffc3f81172faab35226f46c5d7a95f589da06e65cc724408c6e6010f566')
    folder=source(c['raw']+'/manifest.json').rsplit('/',1)[0]
    doc=lambda n:load(root/under(folder,n))
    m,done,selection,fit,selected,identity=map(doc,('manifest.json','complete.json','selection.json','metric_fit48.json','selected_metric.json','identity.json'))
    proof=load(root/source(c['independent_source_proof']['path']))
    original(source(c['independent_source_proof']['path']),c['independent_source_proof']['sha256'])
    require(proof['remote_hashes_computed_independently'] is True and isinstance(proof['files'],list),
            'KRA calibration actual independent source/raw proof missing')
    require(proof['schema']=='trajkv.round10.kra_calibration_retrieval.v1' and
            proof['status']=='PASS_REMOTE_KRA_CALIBRATION_LAUNCH_SOURCE_AND_SCALAR_SHA' and proof['identity_only'] is False and
            proof['single_worker_has_no_sequence_parent_START_CELL_or_exit0_receipt'] is True and
            proof['OS_process_exit0_independently_observed'] is False,'KRA calibration source proof scope differs')
    for e in proof['files']:
        rel=under(folder,e['file']);original(rel,e['sha256']);require(entries[rel]['source_bytes']==e['bytes'],'KRA calibration proof source bytes differ')
    require(done['status']=='PASS_KRA_FIT48_REUSED_AND_ACTUAL_MODEL_SELECT16' and done['actual_GPU'] is True and
            done['fit_sources_reused']==48 and done['new_fit_source_captures']==0 and done['select_sources']==16 and
            done['references_read'] is False and done['new_free_generation_task_answers']==0,
            'KRA actual select16 closure differs')
    require(m['source_hashes']==done['source_hashes'] and m['source_hashes']['code/round10_kra_calibrate_v1.py']==KRA_CALIBRATOR and
            m['source_hashes']['code/round10_kra_metric_v1.py']==KRA_METRIC and
            m['source_hashes']['code/round10_kra_bank_v1.py']==KRA_FACTORY,'KRA calibration source epoch differs')
    for n,digest in {'records.jsonl':done['records_sha256'],'selected_metric.json':done['metric_sha256'],
                     'selection.json':done['selection_sha256'],'identity.json':done['identity_sha256']}.items():original(under(folder,n),digest)
    cfg=doc('source_config.json');launch=doc('launch_receipt.json');job=doc('source_job.json');inventory=doc('source_inventory.json')
    original(under(folder,'source_config.json'),m['config']['calibration_config_sha256'])
    require(m['config']==dict(cfg,calibration_config_sha256=m['config']['calibration_config_sha256']) and
            type(launch['pid']) is int and launch['pid']>0 and launch['pid']==m['runtime_pid']==proof['actual_launch_pid'] and
            m['runtime_source_root']==launch['source_snapshot'],'KRA calibration config/actual parent differs')
    original(under(folder,'source_job.json'),launch['job_sha256']);original(under(folder,'source_inventory.json'),launch['source_inventory_sha256'])
    executed=[launch['source_snapshot']+'/code/'+a.rsplit('/code/',1)[1] if '/code/' in a and a.endswith('.py') else a for a in job['argv']]
    require(executed==launch['argv_executed'] and executed[2]==launch['source_snapshot']+'/code/round10_kra_calibrate_v1.py',
            'KRA calibration actual driver invocation differs')
    require(executed==proof['actual_argv_executed'] and m['runtime_executable']==executed[0] and
            m['runtime_argv']==[executed[0],*executed[2:]],'KRA actual process argv differs')
    original(under(folder,'source_driver.py'),KRA_CALIBRATOR)
    event=proof['actual_complete_stdout_event']
    require(json.loads(event['exact_line_utf8'])==done,'KRA actual complete stdout event differs')
    for rel,digest in m['source_hashes'].items():
        require(inventory[rel]==digest,'KRA actual snapshot graph differs');original(under(folder,'source_files/'+rel),digest)
    require(identity['status']=='PASS_T0_FIELDS_AND_ACTUAL_MODEL_LOGITS_IDENTICAL' and identity['actual_GPU'] is True and
            identity['KRA1_t0_complete_bytes']==identity['original_complete_bytes']+5980 and
            identity['KRA1_t0_actual_TF']['actual_raw_logits_sha256']==identity['original_R7_actual_TF']['actual_raw_logits_sha256'] and
            identity['KRA1_t0_actual_TF']['position_KL']==identity['original_R7_actual_TF']['position_KL'],'KRA t0 actual fields/logits identity differs')
    require(selection['strength_grid']==[0,.25,.5,1.] and selection['selected_strength'] in (.25,.5,1.) and
            selection['selected_on_calSelect16_only'] is True and selected['selection_provenance']==selection and
            selected['strength']==selection['selected_strength']==done['selected_strength'],'KRA selected nonzero strength lineage differs')
    u=np.asarray(selected['u'],dtype=np.float16);gamma=np.asarray(selected['gamma'],dtype=np.float32)
    require(u.shape==(23,128) and gamma.shape==(23,) and np.isfinite(u).all() and np.isfinite(gamma).all() and
            ((gamma>=0)&(gamma<=32)).all() and u.nbytes+gamma.nbytes==5980 and
            selected['unit_order23']==fit['unit_order23'] and selected['u']==fit['u'],'KRA actual Half/F32 metric format differs')
    basegamma=np.asarray(fit['gamma'],dtype=np.float32)
    require(np.array_equal(gamma,(basegamma*np.float32(selected['strength'])).astype(np.float32)), 'KRA strength was not folded into actual F32 gamma')
    losses={t:{} for t in (0.,.25,.5,1.)};teacher_ids=set();teacher_domains=Counter();immutable={};branches=0
    for r in map(json.loads,(root/under(folder,'records.jsonl')).read_text(encoding='utf-8').splitlines()):
        key=r.get('dataset','')+':'+r.get('source_id','')
        if r['phase']=='select_source_teacher':
            require(key not in teacher_ids,'duplicate KRA teacher source');teacher_ids.add(key);teacher_domains[r['dataset']]+=1
        elif r['phase']=='select_branch':
            t=r['strength'];require(t in losses and key not in losses[t],'duplicate/unknown KRA selection branch')
            owner=r['owner_before_question_suffix'];n=r['context']['visual_tokens'];amount=owner['stored_visual_KV_bytes']
            physical={'owner_before_suffix':owner,'actual_visual_token_count':n,'full_kv_byte_denominator':28*2*4*n*128*2,
                      'complete_visual_owner_bytes':amount,'stored_visual_KV_bytes':amount,'r7_bridge':r['owner_metadata']}
            require(physical_complete_bytes(physical,{'layers':28,'kv_heads':4,'head_dim':128})==amount,'KRA calibration physical owner differs')
            kra_physical_guard(physical,amount)
            immutable.setdefault(key,set()).add(r['immutable_fields_sha256'])
            require(r['loss']['actual_model_TF_source_evaluation'] is True and
                    r['loss']['actual_packed_owner_calls']==28*r['loss']['positions'] and
                    r['loss']['question_suffix_recomputed'] is True and r['loss']['question_native_KV_reused'] is False,
                    'KRA actual model/current packed suffix TF scope differs')
            require(r['loss']['positions']==len(r['loss']['position_KL']) and
                    abs(sum(r['loss']['position_KL'])/r['loss']['positions']-r['loss']['source_mean_KL'])<1e-11,
                    'KRA per-source KL scalar arithmetic differs')
            losses[t][key]=r['loss']['source_mean_KL'];branches+=1
    require(len(teacher_ids)==16 and teacher_domains==Counter({d:4 for d in DOMAINS}) and branches==64 and
            all(set(v)==teacher_ids for v in losses.values()) and all(len(v)==1 for v in immutable.values()),'KRA all16/all4strength coverage differs')
    means={t:sum(v.values())/16 for t,v in losses.items()}
    require(all(abs(means[t]-selection['source_balanced_mean_KL'][str(t)])<1e-11 for t in means) and
            min((.25,.5,1.),key=lambda t:(means[t],t))==selection['selected_strength'],'KRA source-balanced selected strength differs')
    out.update(status='PASS_KRA_PUBLIC_SCALAR_SELECTION_AND_SOURCE_CLOSURE',selection_sources=16,branches=64,
               selected_strength=selection['selected_strength'],actual_metric_bytes=5980,
               source_balanced_KL_recomputed=True,metric_tensor_format_verified=True,
               OS_process_exit0_independently_observed=False)
    return out


def public_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def scoped_scalar_sha(input_entries,generated_output):
    """Original identity for explicit archived inputs; real SHA for new outputs.

    The caller supplies only verified archive entries. No unindexed file can
    acquire an input identity. The only output exception is this invocation's
    exact generated folder, including resolved-path containment checks.
    """
    allowed={Path(p).resolve():entry for p,entry in input_entries.items()}
    output=Path(generated_output).resolve()
    def resolve(path):
        actual=Path(path).resolve()
        if actual in allowed:
            entry=allowed[actual]
            require(entry.get('source_sha256') is not None,'scalar input has no original source identity')
            require(sha(actual)==entry['public_sha256'],'archived scalar input body changed')
            return entry['source_sha256']
        require(actual.is_relative_to(output) and actual.is_file(),
                'scalar verifier opened unarchived input or unrelated output')
        return sha(actual)
    return resolve


def replay_legacy_scalar_helper(module,run,input_entries,output):
    previous=module.sha
    module.sha=scoped_scalar_sha(input_entries,output)
    try:return module.summarize(run,output)
    finally:module.sha=previous


def input_identity(r):
    return {k:r[k] for k in ('dataset','dataset_revision','split','source_id','source_cluster_id','question_id',
        'original_image_sha256','question_sha256','full_prompt_hash','input_ids_sha256','question_suffix_sha256',
        'native_position_ids_sha256','native_position_ids_shape','visual_positions','actual_visual_token_count',
        'input_tokens','compression_boundary','question_suffix_tokens')}


def native_context(m):
    names=('torch_version','transformers_version','CUDA_version','python_version','hardware','runtime_executable',
        'processor_id','processor_version','processor_file_hashes','generation_native','generation_overrides',
        'weight_dtype','tf32','seed','attention_implementation','tokenizer_class','tokenizer_is_fast','tokenizers_version','decode_overrides')
    return {'model_spec':m['model_spec'],'model_files':[{k:x[k] for k in ('file','bytes','sha256')} for x in m['model_identity']['files']],
        **{k:m[k] for k in names}}


def bootstrap(values,repetitions,seed):
    import numpy as np
    groups=defaultdict(list)
    for cluster,value in values:groups[cluster].append(value)
    ordered=sorted(groups)
    if len(ordered)<2:return None
    sums=np.asarray([sum(groups[k]) for k in ordered]);sizes=np.asarray([len(groups[k]) for k in ordered])
    draws=np.random.default_rng(seed).integers(0,len(ordered),size=(repetitions,len(ordered)))
    return np.quantile(sums[draws].sum(1)/sizes[draws].sum(1),[.025,.975]).tolist()


def verify(root,archive=None,recompute_scores=False,scratch=None):
    root,index,entries,total,line_count=verify_files(root,archive)
    require(index['custom_driver_provenance_schema']=='trajkv.public_round10.v1' and
            index['custom_SourceRun_admission_claimed'] is False and index['private_tensor_payloads_archived'] is False,
            'Round10 custom/scalar privacy contract differs')
    require(index.get('supplement_provenance_schema')=='trajkv.public_round10.v3','explicit v3 follow-up provenance required')
    aliases=index['source_aliases']
    def normalize(value):
        value=str(value).replace('\\','/')
        while '//' in value:value=value.replace('//','/')
        return value
    def source(value):
        value=normalize(value)
        if value in aliases:
            require(aliases[value] in entries,'source alias target absent');return aliases[value]
        require(value in entries,'exact full source path/alias is missing: '+value);return value
    def original(path,digest):
        require(path in entries and entries[path]['source_sha256']==digest,'original source identity differs: '+path)
    def under(folder,name):
        rel=folder+'/'+name;require(rel in entries,'source closure missing: '+rel);return rel
    contract=load(root/'evidence/ROUND10_STAGE_CONTRACT.json');seal=load(root/'evidence/ROUND10_STAGE_SEAL.json')
    require(contract['schema']=='trajkv.round10.public_stage_contract.v1' and contract['stage_state']=='ROOT_SEALED' and
            contract['SourceRun_admission'] is False and contract['whole_Round10_goal_complete_claimed'] is False,'unsealed or falsely admitted stage')
    binding=index['sealed_scope_bindings']
    require(seal['schema']=='trajkv.round10.public_stage_seal.v1' and seal['status']=='SEALED_FOR_LOCAL_PUBLIC_PACKAGE' and
            seal['phase_label']==contract['phase_label'] and seal['cells_disposition_sha256']==binding['source_cells_sha256'] and
            seal['calibration_disposition_sha256']==binding['source_calibration_sha256'] and
            canonical(contract['cells'])==binding['public_cells_sha256'] and
            canonical(contract['calibration'])==binding['public_calibration_sha256'],'source/public stage seal semantic binding differs')
    require(len(contract['cells'])==28 and {(r['dataset'],r['mode']) for r in contract['cells']}=={(d,m) for d in DOMAINS for m in MODES},'original28 cell scope was changed')
    require(all(r['expected_QA']==COUNTS[r['dataset']] and r['disposition'] in ('CLOSED_COMPLETE','CLOSED_WITH_FAILURES','STOPPED','NOT_RUN') for r in contract['cells']),'pending/partial disposition cannot be exported')
    supplement=load(root/'evidence/ROUND10_SUPPLEMENT_CONTRACT.json')
    supplement_seal=load(root/'evidence/ROUND10_SUPPLEMENT_SEAL.json');sb=index['supplement_sealed_scope_bindings']
    require(supplement['schema']=='trajkv.round10.public_stage_contract.v3' and supplement['stage_state']=='ROOT_SEALED' and
            supplement['whole_Round10_goal_complete_claimed'] is False and supplement['SourceRun_admission'] is False and
            supplement['history_and_all_previous_namespaces_releases_must_be_preserved'] is True,'unsealed/inadmissible v3 scope')
    require(supplement_seal['schema']=='trajkv.round10.public_stage_seal.v3' and
            supplement_seal['status']=='SEALED_FOR_LOCAL_PUBLIC_PACKAGE' and supplement_seal['phase_label']==supplement['phase_label'] and
            supplement_seal['groups_sha256']==sb['source_groups_sha256'] and
            supplement_seal['files_sha256']==sb['source_files_sha256'] and
            canonical(supplement['groups'])==sb['public_groups_sha256'] and
            canonical(supplement['files'])==sb['public_files_sha256'] and
            supplement_seal['kra_calibration_sha256']==sb['source_kra_calibration_sha256'] and
            canonical(supplement['kra_calibration'])==sb['public_kra_calibration_sha256'] and
            supplement_seal['primary_contract_sha256']==sb['primary_contract_source_sha256']==supplement['primary_contract']['sha256'],
            'v3 source/public Root seal semantic binding differs')
    original('evidence/ROUND10_STAGE_CONTRACT.json',supplement['primary_contract']['sha256'])
    require(len(supplement['groups'])==3 and {g['kind'] for g in supplement['groups']}=={'middle_development','kra_development','extension_retest'},'missing actual followup disposition')
    grouped_rows=[{**r,'_group_id':'primary','_kind':'primary','_refs_root':contract['refs_root']} for r in contract['cells']]
    group_results=[]
    for g in supplement['groups']:
        require(g['execution_disposition'] in ('EXECUTED','NOT_RUN'),'pending followup scope cannot export')
        summary_group=next((x for x in index['supplement_scope'] if x['group_id']==g['group_id']),None)
        require(summary_group is not None and summary_group['kind']==g['kind'] and
                summary_group['execution_disposition']==g['execution_disposition'],'followup summary group differs')
        if g['execution_disposition']=='NOT_RUN':
            require(not g['cells'] and not g['plans'] and g.get('reason') and summary_group['new_answer_events']==0 and
                    summary_group['reused_answer_events']==0,'unrun followup cannot invent cells/answers')
            e=g['disposition_evidence'];original(source(e['path']),e['sha256'])
            group_results.append({'group_id':g['group_id'],'kind':g['kind'],'execution_disposition':'NOT_RUN','new_answer_events':0,'reused_answer_events':0})
            continue
        counts=EXTENSION_COUNTS if g['kind']=='extension_retest' else COUNTS;modes=tuple(g['candidate_modes'])
        helper,digest=HARVEST_PINS[g['kind']]
        if g['kind']=='extension_retest' and 'KRA1' in modes:
            helper,digest='ops/round10_extension_harvest_v2.py',KRA_EXTENSION_HARVEST
        require(digest is not None,'followup approved source epoch remains unsealed')
        require(summary_group['harvest_adapter']=={'file':helper,'source_sha256':digest},'actual followup harvest adapter epoch differs')
        original(helper,digest)
        original('ops/round10_middle_harvest_v1.py','4bd78e263c23680f2162efdb1a0277b3029bdb70a53e2bd60038c2b10b191ebe')
        require(g['counts']==counts and (modes==MIDDLE_MODES if g['kind']=='middle_development' else modes==('KRA1',) if g['kind']=='kra_development' else
                len(modes) in (1,2) and len(set(modes))==len(modes) and set(modes)<=set((*MODES[3:],*MIDDLE_MODES,'KRA1'))),'followup cohort/modes differ')
        require(len(g['cells'])==4*(3+len(modes)) and {(r['dataset'],r['mode']) for r in g['cells']}==
                {(d,m) for d in DOMAINS for m in ('FP16','R7low','R7high',*modes)},'followup exact matrix differs')
        for r in g['cells']:
            require(r['expected_QA']==counts[r['dataset']] and r['disposition'] in ('CLOSED_COMPLETE','CLOSED_WITH_FAILURES','STOPPED','NOT_RUN'),'invalid followup row')
            if g['kind'] in ('middle_development','kra_development') and r['mode'] in ('FP16','R7low','R7high'):
                require(r.get('reuse_from')=={'group_id':'primary','dataset':r['dataset'],'mode':r['mode']},'middle reference not explicit primary reuse')
                old=next(x for x in contract['cells'] if (x['dataset'],x['mode'])==(r['dataset'],r['mode']))
                require(old['disposition']==r['disposition'] and old['expected_QA']==r['expected_QA'],'reused reference disposition differs')
            else:require(r.get('reuse_from') is None,'fresh followup falsely reused')
            grouped_rows.append({**r,'_group_id':g['group_id'],'_kind':g['kind'],'_refs_root':g['refs_root']})
        require(len(summary_group['cells'])==len(g['cells']),'followup summary cell count differs')
        group_results.append({'group_id':g['group_id'],'kind':g['kind'],'execution_disposition':'EXECUTED',
                              'new_answer_events':0,'reused_answer_events':0})
    for item in contract['files']:
        rel=source(item['path']);original(rel,item['sha256'])
    for item in supplement['files']:original(source(item['path']),item['sha256'])
    force=set(index['git_stage_plan']['force_paths'])
    require(force=={index['namespace']+'/'+p for p,e in entries.items() if e['expected_in_git']} and
            index['git_stage_plan']['verify_actual_tree_and_blob_bodies'] is True and
            index['git_stage_plan']['configs_data_ignore_rule_must_not_hide_required_files'] is True,'explicit expected Git tree differs')
    for rel in index['required_reference_files']:
        require(rel in entries and entries[rel]['expected_in_git'] and index['namespace']+'/'+rel in force,'reference/input swallowed by ignore planning')
    safe_suffix={'.py','.json','.jsonl','.csv','.md','.txt','.toml','.yaml','.yml'}
    require(all(Path(p).suffix.lower() in safe_suffix or Path(p).name=='.gitattributes' or
                Path(p).name.lower().startswith(('license','copying','notice')) for p in entries),
            'unreviewed binary/tensor/model/image/parquet payload in public namespace')
    for rel,e in entries.items():
        ref=e['content_ref'];require(ref['member']==index['namespace']+'/'+rel and ref['public_body_sha256']==e['public_sha256'] and
            ref['archive_asset'].startswith('https://github.com/zhouz4884-coder/TrajKV/releases/download/'+index['release_tag']+'/'),'fixed content_ref differs')
    locks=[p for p,e in entries.items() if e.get('evidence_role')=='source_lock']
    require(len(locks)==1,'one exact primary Round10 reference source lock required')
    sys.path.insert(0,str(root/'code'))
    prior_vendor=sys.modules.get('vendor_textvqa_metric')
    public_module('vendor_textvqa_metric',root/'code/vendor_textvqa_metric.py')
    metric=public_module('_round10_verified_metrics',root/'code/metrics.py')
    if prior_vendor is not None:sys.modules['vendor_textvqa_metric']=prior_vendor
    bundles={};quality=[];physical_count=0
    for declared in grouped_rows:
        ds,mode=declared['dataset'],declared['mode'];state=declared['disposition'];n=declared['expected_QA']
        group,kind=declared['_group_id'],declared['_kind']
        if state in ('STOPPED','NOT_RUN'):
            require(declared.get('primary_mean') is None and declared.get('quality_QA') in (None,0),'stopped/unrun has an invented primary')
            original(source(declared['disposition_evidence']['path']),declared['disposition_evidence']['sha256'])
            quality.append({'group_id':group,'dataset':ds,'mode':mode,'disposition':state,'primary_mean':None,'scored_QA':0});continue
        if kind=='primary': refs_scope=index['phase']['closed']+index['phase']['failures']
        else:refs_scope=next(x for x in index['supplement_scope'] if x['group_id']==group)['cells']
        ref=next(x for x in refs_scope if (x['dataset'],x['mode'])==(ds,mode))
        if declared.get('reuse_from'):
            primary_ref=next(x for x in index['phase']['closed']+index['phase']['failures'] if (x['dataset'],x['mode'])==(ds,mode))
            require(ref['raw']==primary_ref['raw'] and ref['scored']==primary_ref['scored'],'middle reused view is not the same actual primary artifact')
        folder=source(ref['raw']+'/manifest.json').rsplit('/',1)[0]
        scored=source(ref['scored']+'/summary.json').rsplit('/',1)[0]
        m=load(root/under(folder,'manifest.json'));c=load(root/under(folder,'complete.json'));cfg=m['config']
        require(m['schema']=='baseline_matrix.raw_inference.v1' and m['runner_schema']=='trajkv.round10_research.raw.v1' and
                m['references_read'] is False and m['prediction_cache_used'] is False and
                cfg['candidate']['mode']==('original' if mode in ('R7low','R7high') else mode) and
                cfg['candidate'].get('technical_only',False) is False and
                cfg['input_role']==('reserved_unused_extension' if kind=='extension_retest' else 'development') and
                cfg['cohort_kind']==('replication' if kind=='extension_retest' else 'development') and
                cfg['start']==0 and cfg['count']==cfg['input_count']==n,'wrong custom driver/cohort/candidate')
        require(m['expected_QA']==c['expected_QA']==c['recorded_QA']==n and c['coverage_complete'] is True and c['quality_scored'] is False,'whole-domain closure differs')
        original(under(folder,'manifest.json'),c['manifest_sha256']);original(under(folder,'records.jsonl'),c['records_jsonl_sha256'])
        proof=load(root/under(folder,'source_retrieval_receipt_round10_v1.json'))
        require(proof['status']=='PASS_ROUND10_SEQUENCE_CHILD_EXECUTED_SOURCE_SHA' and proof['remote_hashes_computed_independently'] is True and
                proof['SourceRun_admission_established'] is False,'missing independent actual child source proof')
        require(len(proof['files'])==len({x['file'] for x in proof['files']}),'duplicate proof files')
        if state=='CLOSED_COMPLETE':require(proof['parent_cell_success'] is True,'successful closed cell lacks parent exit0 commitment')
        for e in proof['files']:
            rel=under(folder,e['file']);original(rel,e['sha256']);require(entries[rel]['source_bytes']==e['bytes'],'source proof original length differs')
        inventory=load(root/under(folder,'source_inventory.json'));launch=load(root/under(folder,'launch_receipt.json'))
        job=load(root/under(folder,'source_parent_job.json'));seq=load(root/under(folder,'source_sequence_config.json'))
        original(under(folder,'source_parent_job.json'),launch['job_sha256']);original(under(folder,'source_inventory.json'),launch['source_inventory_sha256'])
        require(type(launch['pid']) is int and launch['pid']>0 and launch['name']==proof['parent_producer'] and
                m['runtime_source_root']==launch['source_snapshot'],'actual parent producer differs')
        executed=[launch['source_snapshot']+'/code/'+a.rsplit('/code/',1)[1] if '/code/' in a and a.endswith('.py') else a for a in job['argv']]
        require(executed==launch['argv_executed'] and executed[2].endswith('/code/round10_sequence_v1.py'),'parent actual argv differs')
        event=proof['actual_child_start'];start=event['event']
        require(json.loads(event['exact_line_utf8'])==start and start['name']==proof['cell_name'] and type(start['pid']) is int and start['pid']>0,'child start evidence differs')
        cells=[x for x in seq['entries'] if x['name']==start['name']]
        require(len(cells)==1,'actual child missing from sequence');cell=cells[0]
        require(start['argv']==[executed[0],'-B',launch['source_snapshot']+'/'+cell['script'],'--config',cell['config'],'--output',cell['output']] and
                start['config_sha256']==cell['config_sha256']==m['config_sha256'],'actual child argv/config differs')
        extension_kra=kind=='extension_retest' and 'KRA1' in next(g['candidate_modes'] for g in supplement['groups'] if g['group_id']==group)
        driver=('code/round10_extension_runner_v2.py' if extension_kra else 'code/round10_extension_runner_v1.py' if kind=='extension_retest' else
                'code/round10_kra_runner_v1.py' if mode=='KRA1' else
                'code/round10_middle_runner_v1.py' if mode in MIDDLE_MODES else 'code/round10_runner_v1.py')
        require(cell['script']==proof['actual_driver']==driver,'executed driver is wrong for followup scope')
        original(under(folder,'source_driver.py'),cell['script_sha256']);original(under(folder,'source_config.json'),m['config_sha256'])
        require(m['source_hashes'][driver]==cell['script_sha256'],'executed child source differs')
        if driver=='code/round10_runner_v1.py':require(cell['script_sha256']==R10_DRIVER,'primary driver epoch differs')
        if kind=='extension_retest':require(cell['script_sha256']==(KRA_EXTENSION_DRIVER if extension_kra else EXTENSION_DRIVER),'extension driver epoch differs')
        if mode=='KRA1':
            require(KRA_DRIVER is not None and KRA_FACTORY is not None and KRA_METRIC is not None and
                    m['source_hashes']['code/round10_kra_runner_v1.py']==KRA_DRIVER and
                    m['source_hashes']['code/round10_kra_bank_v1.py']==KRA_FACTORY and
                    m['source_hashes']['code/round10_kra_metric_v1.py']==KRA_METRIC,'KRA actual approved source epoch differs')
        if mode in MIDDLE_MODES:
            require(m['source_hashes']['code/round10_middle_runner_v1.py']==MIDDLE_DRIVER and
                    m['source_hashes']['code/round10_middle_bank_v1.py']==MIDDLE_FACTORY,'middle source epoch differs')
            require(m['frozen_low43_profile_is_current_deployment_proof'] is False and
                    (m['extension_deployment_contract']['numerical_parent']=='middle' if kind=='extension_retest' else
                     m['middle_deployment_contract']['base_variant_is_encoding_lineage_only'] is True),
                    'middle actual deployment conflated with original43 profile')
        require(m['source_hashes']==cfg['expected_source_hashes'] and m['source_hashes']['code/round10_runner_v1.py']==R10_DRIVER and
                m['source_hashes']['code/round10_value_bank_v2.py']==R10_FACTORY,'unapproved driver/factory epoch')
        for rel,digest in m['source_hashes'].items():
            require(inventory[rel]==digest,'snapshot source declaration differs');original(under(folder,'source_files/'+rel),digest)
        snap=proof['parent_state_snapshot'];parent_state=json.loads(snap['body_utf8'])
        original(under(folder,'source_sequence_config.json'),parent_state['plan_sha256'])
        require(parent_state['source_root']==launch['source_snapshot'],'parent snapshot source differs')
        if state=='CLOSED_COMPLETE':
            closes=[x for x in parent_state['completed_cells'] if x['name']==cell['name']]
            require(len(closes)==1 and closes[0]['exit_code']==0 and closes[0]['output']==cell['output'] and
                    closes[0]['recorded_QA']==n and closes[0]['complete_sha256']==entries[under(folder,'complete.json')]['source_sha256'],
                    'parent committed child/raw3 binding differs')
        if kind=='extension_retest':
            lineage=cfg['extension_lineage'];parentrel=source(lineage['parent_config_path']);original(parentrel,lineage['parent_config_sha256'])
            parent=load(root/parentrel);allowed={'inputs','inputs_sha256','input_count','start','count','input_role','cohort_kind',
                'scope','expected_source_hashes','extension_lineage','input_manifest_sha256'}
            same=lambda x:{k:v for k,v in x.items() if k not in allowed}
            require(same(parent)==same(cfg) and parent['input_role']=='development' and parent['cohort_kind']=='development',
                    'extension reselected parameters or changed original scientific method')
        actual=c.get('executed_legacy_source_sha256',{})
        for e in proof['files']:
            if e['file'].startswith('legacy_sources/') and e['remote_file'] in actual:require(e['sha256']==actual[e['remote_file']],'actual legacy/source proof differs')
        if mode!='FP16':r7_verify_executed_roles(cfg,actual)
        for key,digest in cfg['expected_asset_sha256'].items():original(under(folder,'assets/base_'+key+'.json'),digest)
        if mode=='KRA1':
            require(supplement['kra_calibration']['disposition']=='CLOSED_COMPLETE','KRA free answers lack completed select16 lineage')
            selected_complete=load(root/source(supplement['kra_calibration']['raw']+'/complete.json'))
            require(cfg['candidate']['metric_sha256']==selected_complete['metric_sha256'],'KRA free answers use a different metric than selected cal16')
            metricfile=under(folder,'assets/candidate_metric.json');original(metricfile,cfg['candidate']['metric_sha256'])
            metricdoc=load(root/metricfile)
            require(metricdoc['schema']=='trajkv.round10.kra_metric.v1' and metricdoc['strength']==selected_complete['selected_strength'] and
                    metricdoc['metric_storage_bytes']==5980,'KRA actual selected metric JSON differs')
        original(under(folder,'assets/original_profile.json'),cfg['variant_source']['sha256'])
        original_doc=load(root/under(folder,'assets/original_profile.json'));variants=[v for v in original_doc['variants'] if v['name']==cfg['variant_source']['name']]
        strip=lambda v:{k:x for k,x in v.items() if k not in cfg['expected_asset_sha256']}
        require(len(variants)==1 and strip(variants[0])==strip(cfg['variant']),'original base profile numeric fields changed')
        inputname=PurePosixPath(cfg['inputs'].replace('\\','/')).name;refname=inputname.replace('_inputs.jsonl','_refs.jsonl')
        lockpath=source(declared['_refs_root']+'/source_lock.json');lock=load(root/lockpath)
        inputs=source(cfg['inputs']);original(inputs,m['inputs_sha256']);require(entries[inputs]['source_sha256']==lock['files'][inputname]['sha256'],'frozen input/source lock differs')
        refs=[p for p,e in entries.items() if e.get('evidence_role') in ('reference_answers','followup_reference_answers') and Path(p).name==refname]
        require(len(refs)==1,'exact cohort reference file missing/ambiguous');refs=refs[0];original(refs,lock['files'][refname]['sha256'])
        inrows=[json.loads(line) for line in (root/inputs).read_text(encoding='utf-8').splitlines()]
        references={r['id']:r for r in map(json.loads,(root/refs).read_text(encoding='utf-8').splitlines())}
        require(len(inrows)==len(references)==n and [r['id'] for r in inrows]==m['expected_ids'],'exact full-domain inputs/references differ')
        s=load(root/under(scored,'summary.json'));original(under(scored,'scored_records.jsonl'),s['derived_file_hashes']['scored_records.jsonl'])
        require(s['expected_run_questions']==n and s['integrity']['inputs_sha256']==m['inputs_sha256'] and
                s['integrity']['refs_sha256']==entries[refs]['source_sha256'] and
                s['integrity']['dataset_source_lock_sha256']==entries[lockpath]['source_sha256'],'score source/reference bindings differ')
        for name,digest in s['integrity']['raw_files'].items():original(under(folder,name),digest)
        for rel,digest in s['integrity']['metric_code_hashes'].items():original(rel,digest)
        original('code/score.py',s['integrity']['scorer_sha256'])
        values=[];paired={};ids={};statuses=Counter();failed=0
        rawpath=under(folder,'records.jsonl');scorepath=under(scored,'scored_records.jsonl');side=entries[rawpath]['line_identities_file']
        with (root/rawpath).open(encoding='utf-8') as raw,(root/scorepath).open(encoding='utf-8') as scores,(root/side).open(encoding='utf-8') as lines:
            for number,frozen in enumerate(inrows,1):
                r=json.loads(raw.readline());v=json.loads(scores.readline());line=json.loads(lines.readline())
                require(r['id']==v['id']==frozen['id'] and v['raw_jsonl_line_number']==number and v['raw_jsonl_exact_line_sha256']==line['source_sha256'],'raw/score original row binding differs')
                statuses[r['status']]+=1
                for key in ('dataset','dataset_revision','split','source_cluster_id','source_id','question_id','original_image_sha256','question_sha256'):
                    require(r[key]==frozen[key],'raw/frozen source input differs')
                if r['status']!='PASS':require(v['score'] is None,'raw failure got a fake score');failed+=1;continue
                amount=physical_complete_bytes(r,m['model_spec']);physical_count+=1;nv=r['actual_visual_token_count']
                require(abs(r['actual_effective_visual_bits']-16*amount/r['full_kv_byte_denominator'])<1e-12,'actual bytes/bits differ')
                require(r['original_full_prefix_released'] is True and r['original_FP16_visual_alias_retained'] is False and
                        r['owner_after_generation']['stored_visual_KV_bytes']==amount,'native prefix retained or visual owner mutated')
                if mode!='FP16':r7_verify_record_role_pins(cfg,r)
                if mode=='KRA1':kra_physical_guard(r,amount)
                elif mode in MIDDLE_MODES:middle_physical_guard(r,mode,amount)
                elif mode not in ('FP16','R7high'):
                    delta=896 if mode in ('LS-FZ','LS-RE') else 0
                    require(amount==4590*nv+89733+delta and r['r7_bridge']['selected_stage_count']==43,'low43 or real theta bytes differ')
                    if mode not in ('R7low',):
                        vc=r['r7_bridge']['round10_value_candidate']
                        require(vc['mode']==cfg['candidate']['mode'] and vc['source_sha256']==R10_FACTORY and
                                vc['actual_extra_persistent_bytes']==delta and vc['new_theta_asset_bytes']==delta and
                                vc['fixed_K'] is True and vc['native_FP16_source_or_feedback_workspace_retained'] is False,'value candidate/owner contract differs')
                if recompute_scores:
                    score=metric.score_question(ds,r['raw_prediction'],references[r['id']])
                    require(v['score_status']=='SCORED_PASS' and abs(score['score']-v['score'])<1e-12 and
                            score['metric_id']==v['metric_id'] and score['reference_texts']==v['reference_texts'] and
                            score['group']==v.get('group'),'primary metric/reference score differs')
                if v['score_status']!='SCORED_PASS':failed+=1;continue
                values.append((v['source_cluster_id'],v['score']));paired[r['id']]=(v['source_cluster_id'],v['score']);ids[r['id']]=input_identity(r)
            require(not raw.readline() and not scores.readline() and not lines.readline(),'extra rows beyond whole-domain cohort')
        require(dict(statuses)==c['counts'],'actual raw status count differs')
        if failed:
            require(state=='CLOSED_WITH_FAILURES' and s['primary_mean'] is None,'failures retain a false primary quality')
        else:
            require(state=='CLOSED_COMPLETE' and s['status']=='PASS_SCORED_COMPLETE' and c['status']=='PASS_RAW_COMPLETE' and
                    abs(sum(v for _,v in values)/n-s['primary_mean'])<1e-12,'complete full-domain mean differs')
            if recompute_scores:
                ci=bootstrap(values,s['bootstrap']['repetitions'],s['bootstrap']['seed'])
                require(ci is None and s['primary_ci95'] is None or ci is not None and max(abs(a-b) for a,b in zip(ci,s['primary_ci95']))<1e-12,'source-cluster CI differs')
            bundles[group,ds,mode]={'context':native_context(m),'identity':ids,'scores':paired}
        if kind!='primary':
            output=next(x for x in group_results if x['group_id']==group)
            output['reused_answer_events' if declared.get('reuse_from') else 'new_answer_events']+=n
        quality.append({'group_id':group,'dataset':ds,'mode':mode,'disposition':state,'primary_mean':s['primary_mean'],'scored_QA':len(values),
            'primary_recomputed':bool(recompute_scores),'failure_QA':failed})
    calibration=contract['calibration'];calibration_result={'disposition':calibration['disposition'],'private_tensor_payloads_replayed':False}
    if calibration['disposition']=='CLOSED_COMPLETE':
        cal=source(calibration['raw']+'/manifest.json').rsplit('/',1)[0]
        complete=load(root/under(cal,'complete.json'));manifest=load(root/under(cal,'manifest.json'))
        require(complete['status']=='PASS_LS48_AND_ACTUAL_MODEL_SELECT16' and complete['actual_GPU'] is True and
                manifest['source_hashes']==complete['source_hashes'] and complete['source_hashes']['code/round10_calibrate_v1.py']==R10_CALIBRATOR,'selector producer epoch differs')
        for name,digest in {'records.jsonl':complete['records_sha256'],'value_theta_selected.json':complete['theta_sha256'],'window_selected.json':complete['window_sha256']}.items():original(under(cal,name),digest)
        if recompute_scores:
            helper='ops/round10_summarize_calibration_v1.py';require(helper in entries,'frozen scalar selector verifier required')
            mod=public_module('_round10_public_scalar_calibration',root/helper)
            # Original inputs and this call's newly generated outputs have
            # distinct identities; arbitrary unarchived files stay rejected.
            input_entries={(root/rel).resolve():entry for rel,entry in entries.items()}
            scratch=Path(scratch) if scratch else root.parent/'round10_cold_scratch'
            scratch.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='selector_',dir=scratch) as temporary:
                require(Path(temporary).resolve().is_relative_to(scratch.resolve()),'selector scratch escaped requested directory')
                calibration_result.update(replay_legacy_scalar_helper(mod,root/cal,input_entries,Path(temporary)/'scalar_recompute'))
            require(calibration_result['status']=='PASS_LOCAL_SELECTOR_AND_COUNTER_RECOMPUTATION','scalar selection does not independently recompute')
    else:require(calibration['disposition']=='STOPPED' and calibration['selected_parameters_claimed'] is False,'unclosed calibration pretends to select parameters')
    pairs=[];pair_checks=[]
    scopes=[('primary',('LS-FZ','LS-RE','FB0','FB-W'))]+[(g['group_id'],tuple(g['candidate_modes']))
            for g in supplement['groups'] if g['execution_disposition']=='EXECUTED']
    for group,modes in scopes:
        contrasts=[(c,a) for c in modes for a in ('FP16','R7low','R7high')]
        if group=='primary':contrasts += [('LS-RE','LS-FZ'),('FB-W','FB0')]
        elif len(modes)==2:contrasts.append((modes[1],modes[0]))
        for ds in DOMAINS:
            for candidate,anchor in contrasts:
                if (group,ds,candidate) not in bundles or (group,ds,anchor) not in bundles:continue
                left,right=bundles[group,ds,anchor],bundles[group,ds,candidate]
                require(left['context']==right['context'] and left['identity']==right['identity'],'paired actual runtime/position/input identity differs')
                pair_checks.append({'group_id':group,'dataset':ds,'candidate':candidate,'anchor':anchor,'QA':len(left['scores'])})
                if not recompute_scores:continue
                import numpy as np
                grouped={}
                for identity,(cluster,value) in left['scores'].items():
                    require(right['scores'][identity][0]==cluster,'paired source cluster differs')
                    entry=grouped.setdefault(cluster,[0.,0]);entry[0]+=right['scores'][identity][1]-value;entry[1]+=1
                values=np.asarray(list(grouped.values()));sums,counts=values[:,0],values[:,1]
                rng=np.random.default_rng(20261119);draws=rng.integers(0,len(values),size=(2000,len(values)))
                ci=np.percentile(sums[draws].sum(1)/counts[draws].sum(1),[2.5,97.5]).tolist()
                pairs.append({'group_id':group,'dataset':ds,'candidate':candidate,'anchor':anchor,'QA':len(left['scores']),
                    'difference_pp':100*float(sums.sum()/counts.sum()),'ci95_pp':[100*x for x in ci],
                    'unadjusted_descriptive_not_equivalence_or_algorithm_winner':True})
    actualclosed=sum(r['disposition']=='CLOSED_COMPLETE' for r in quality if r['group_id']=='primary')
    require(actualclosed==index['phase']['closed_cells'] and sum(COUNTS[r['dataset']] for r in quality if
            r['group_id']=='primary' and r['disposition']=='CLOSED_COMPLETE')==index['phase']['closed_quality_answer_events'],'false primary closed-quality count')
    for g in group_results:
        claimed=next(x for x in index['supplement_scope'] if x['group_id']==g['group_id'])
        declared_group=next(x for x in supplement['groups'] if x['group_id']==g['group_id'])
        qrows=[r for r in quality if r['group_id']==g['group_id']]
        require(g['new_answer_events']==claimed['new_answer_events'] and g['reused_answer_events']==claimed['reused_answer_events'],
                'fresh/reused followup event counts differ')
        require(claimed['view_cells']==len(declared_group['cells']) and
                claimed.get('declared_view_answer_events',0)==sum(r['expected_QA'] for r in declared_group['cells']) and
                claimed['closed_quality_answer_events']==sum(r['scored_QA'] for r in qrows if r['disposition']=='CLOSED_COMPLETE'),
                'followup declared-view/closed-quality scalar counts differ')
    require(sum(g['new_answer_events'] for g in group_results)==index['phase']['followup_new_answer_events'] and
            sum(g['reused_answer_events'] for g in group_results)==index['phase']['followup_reused_answer_events'],
            'combined primary/followup scope counts differ')
    primary_actual=sum(r['expected_QA'] for r in grouped_rows if r['_group_id']=='primary' and
                       r['disposition'] in ('CLOSED_COMPLETE','CLOSED_WITH_FAILURES'))
    require(index['phase']['supplement_scope']==index['supplement_scope'] and
            index['phase']['primary_scope_only_cells']==28 and index['phase']['primary_scope_only_planned_events']==1120 and
            index['phase']['primary_actual_answer_events']==primary_actual and
            index['phase']['total_declared_view_answer_events']==1120+sum(r['expected_QA'] for g in supplement['groups'] for r in g['cells']) and
            index['phase']['total_actual_view_answer_events']==primary_actual+sum(g['new_answer_events']+g['reused_answer_events'] for g in group_results),
            'combined scope view/actual scalar counts differ')
    require(index['phase']['total_actual_fresh_answer_events']==primary_actual+sum(g['new_answer_events'] for g in group_results),
            'combined fresh answer events confused with reused views')
    require(index['phase']['failed_cells']==sum(r['disposition']=='CLOSED_WITH_FAILURES' for r in quality if r['group_id']=='primary') and
            index['phase']['stopped_or_unrun_cells']==sum(r['disposition'] in ('STOPPED','NOT_RUN') for r in quality if r['group_id']=='primary'),
            'primary final-disposition scalar counts differ')
    require(index['publication_eligible'] is True,'unsealed/ineligible package')
    kra_calibration_result=verify_kra_calibration(root,supplement,entries,source,original,under,recompute_scores)
    return {'status':'PASS_PUBLIC_ROUND10_EXPORT_VERIFIED_V3','public_files':len(entries),'public_payload_bytes':total,
        'jsonl_line_identities':line_count,'primary_closed_quality_cells':actualclosed,'primary_declared_cells':28,
        'declared_view_cells':len(grouped_rows),'supplement_groups':group_results,
        'primary_stopped_unrun_or_failed_cells':28-actualclosed,'quality_recomputations':quality,'physical_owner_rows':physical_count,
        'same_input_positions_runtime_pair_checks':pair_checks,
        'paired_CI_recomputations':pairs,'calibration_scalar_recomputation':calibration_result,'archive_verified':bool(archive),
        'kra_calibration_scalar_recomputation':kra_calibration_result,
        'SourceRun_admission_claimed':False,'private_tensor_payloads_verified_or_replayed':False,
        'whole_Round10_goal_complete_claimed':False,'publication_eligible':True,'GitHub_upload_verified':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--archive',type=Path)
    p.add_argument('--recompute-scores',action='store_true');p.add_argument('--scratch',type=Path);p.add_argument('--receipt',type=Path)
    a=p.parse_args();result=verify(a.root,a.archive,a.recompute_scores,a.scratch)
    if a.receipt:
        with a.receipt.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(result,ensure_ascii=False));return 0


if __name__=='__main__':raise SystemExit(main())
