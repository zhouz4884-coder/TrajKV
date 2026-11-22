"""Portable cold Round10 custom-driver verification; never canonical admission.

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


def public_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


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
    for item in contract['files']:
        rel=source(item['path']);original(rel,item['sha256'])
    force=set(index['git_stage_plan']['force_paths'])
    require(force=={index['namespace']+'/'+p for p,e in entries.items() if e['expected_in_git']} and
            index['git_stage_plan']['verify_actual_tree_and_blob_bodies'] is True and
            index['git_stage_plan']['configs_data_ignore_rule_must_not_hide_required_files'] is True,'explicit expected Git tree differs')
    for rel in index['required_reference_files']:
        require(rel in entries and entries[rel]['expected_in_git'] and index['namespace']+'/'+rel in force,'reference/input swallowed by ignore planning')
    bad_suffix={'.pt','.pth','.safetensors','.bin','.parquet','.jpg','.jpeg','.png','.webp','.mp4','.wav'}
    require(not any(Path(p).suffix.lower() in bad_suffix for p in entries),'private tensor/model/image/parquet payload in public namespace')
    for rel,e in entries.items():
        ref=e['content_ref'];require(ref['member']==index['namespace']+'/'+rel and ref['public_body_sha256']==e['public_sha256'] and
            ref['archive_asset'].startswith('https://github.com/zhouz4884-coder/TrajKV/releases/download/'+index['release_tag']+'/'),'fixed content_ref differs')
    locks=[p for p,e in entries.items() if e.get('evidence_role')=='source_lock']
    require(len(locks)==1,'one exact Round10 reference source lock required');lock=load(root/locks[0])
    sys.path.insert(0,str(root/'code'))
    prior_vendor=sys.modules.get('vendor_textvqa_metric')
    public_module('vendor_textvqa_metric',root/'code/vendor_textvqa_metric.py')
    metric=public_module('_round10_verified_metrics',root/'code/metrics.py')
    if prior_vendor is not None:sys.modules['vendor_textvqa_metric']=prior_vendor
    bundles={};quality=[];physical_count=0
    for declared in contract['cells']:
        ds,mode=declared['dataset'],declared['mode'];state=declared['disposition'];n=declared['expected_QA']
        if state in ('STOPPED','NOT_RUN'):
            require(declared.get('primary_mean') is None and declared.get('quality_QA') in (None,0),'stopped/unrun has an invented primary')
            original(source(declared['disposition_evidence']['path']),declared['disposition_evidence']['sha256'])
            quality.append({'dataset':ds,'mode':mode,'disposition':state,'primary_mean':None,'scored_QA':0});continue
        ref=next(x for x in index['phase']['closed']+index['phase']['failures'] if (x['dataset'],x['mode'])==(ds,mode))
        folder=source(ref['raw']+'/manifest.json').rsplit('/',1)[0]
        scored=source(ref['scored']+'/summary.json').rsplit('/',1)[0]
        m=load(root/under(folder,'manifest.json'));c=load(root/under(folder,'complete.json'));cfg=m['config']
        require(m['schema']=='baseline_matrix.raw_inference.v1' and m['runner_schema']=='trajkv.round10_research.raw.v1' and
                m['references_read'] is False and m['prediction_cache_used'] is False and
                cfg['candidate']['mode']==('original' if mode in ('R7low','R7high') else mode) and
                cfg['candidate'].get('technical_only',False) is False and cfg['input_role']=='development' and
                cfg['start']==0 and cfg['count']==cfg['input_count']==n,'wrong custom driver/cohort/candidate')
        require(m['expected_QA']==c['expected_QA']==c['recorded_QA']==n and c['coverage_complete'] is True and c['quality_scored'] is False,'whole-domain closure differs')
        original(under(folder,'manifest.json'),c['manifest_sha256']);original(under(folder,'records.jsonl'),c['records_jsonl_sha256'])
        proof=load(root/under(folder,'source_retrieval_receipt_round10_v1.json'))
        require(proof['status']=='PASS_ROUND10_SEQUENCE_CHILD_EXECUTED_SOURCE_SHA' and proof['remote_hashes_computed_independently'] is True and
                proof['SourceRun_admission_established'] is False,'missing independent actual child source proof')
        require(len(proof['files'])==len({x['file'] for x in proof['files']}),'duplicate proof files')
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
        original(under(folder,'source_driver.py'),R10_DRIVER);original(under(folder,'source_config.json'),m['config_sha256'])
        require(m['source_hashes']==cfg['expected_source_hashes'] and m['source_hashes']['code/round10_runner_v1.py']==R10_DRIVER and
                m['source_hashes']['code/round10_value_bank_v2.py']==R10_FACTORY,'unapproved driver/factory epoch')
        for rel,digest in m['source_hashes'].items():
            require(inventory[rel]==digest,'snapshot source declaration differs');original(under(folder,'source_files/'+rel),digest)
        actual=c.get('executed_legacy_source_sha256',{})
        for e in proof['files']:
            if e['file'].startswith('legacy_sources/') and e['remote_file'] in actual:require(e['sha256']==actual[e['remote_file']],'actual legacy/source proof differs')
        if mode!='FP16':r7_verify_executed_roles(cfg,actual)
        for key,digest in cfg['expected_asset_sha256'].items():original(under(folder,'assets/base_'+key+'.json'),digest)
        original(under(folder,'assets/original_profile.json'),cfg['variant_source']['sha256'])
        original_doc=load(root/under(folder,'assets/original_profile.json'));variants=[v for v in original_doc['variants'] if v['name']==cfg['variant_source']['name']]
        strip=lambda v:{k:x for k,x in v.items() if k not in cfg['expected_asset_sha256']}
        require(len(variants)==1 and strip(variants[0])==strip(cfg['variant']),'original base profile numeric fields changed')
        inputname=PurePosixPath(cfg['inputs'].replace('\\','/')).name;refname=inputname.replace('_inputs.jsonl','_refs.jsonl')
        inputs=source(cfg['inputs']);original(inputs,m['inputs_sha256']);require(entries[inputs]['source_sha256']==lock['files'][inputname]['sha256'],'frozen input/source lock differs')
        refs=[p for p,e in entries.items() if e.get('evidence_role')=='reference_answers' and Path(p).name==refname]
        require(len(refs)==1,'exact cohort reference file missing/ambiguous');refs=refs[0];original(refs,lock['files'][refname]['sha256'])
        inrows=[json.loads(line) for line in (root/inputs).read_text(encoding='utf-8').splitlines()]
        references={r['id']:r for r in map(json.loads,(root/refs).read_text(encoding='utf-8').splitlines())}
        require(len(inrows)==len(references)==n and [r['id'] for r in inrows]==m['expected_ids'],'exact full-domain inputs/references differ')
        s=load(root/under(scored,'summary.json'));original(under(scored,'scored_records.jsonl'),s['derived_file_hashes']['scored_records.jsonl'])
        require(s['expected_run_questions']==n and s['integrity']['inputs_sha256']==m['inputs_sha256'] and
                s['integrity']['refs_sha256']==entries[refs]['source_sha256'] and
                s['integrity']['dataset_source_lock_sha256']==entries[locks[0]]['source_sha256'],'score source/reference bindings differ')
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
                if mode not in ('FP16','R7high'):
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
            bundles[ds,mode]={'context':native_context(m),'identity':ids,'scores':paired}
        quality.append({'dataset':ds,'mode':mode,'disposition':state,'primary_mean':s['primary_mean'],'scored_QA':len(values),
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
            # Its original-source guards must resolve original identities after
            # path sanitization. Numeric rows/means are recomputed from public bytes.
            def original_sha(path):
                rel=Path(path).resolve().relative_to(root).as_posix();require(rel in entries,'scalar verifier opened unarchived input')
                return entries[rel]['source_sha256']
            mod.sha=original_sha
            scratch=Path(scratch) if scratch else root.parent/'round10_cold_scratch'
            scratch.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='selector_',dir=scratch) as temporary:
                calibration_result.update(mod.summarize(root/cal,Path(temporary)/'scalar_recompute'))
            require(calibration_result['status']=='PASS_LOCAL_SELECTOR_AND_COUNTER_RECOMPUTATION','scalar selection does not independently recompute')
    else:require(calibration['disposition']=='STOPPED' and calibration['selected_parameters_claimed'] is False,'unclosed calibration pretends to select parameters')
    pairs=[]
    if recompute_scores:
        import numpy as np
        contrasts=[(c,a) for c in ('LS-FZ','LS-RE','FB0','FB-W') for a in ('FP16','R7low','R7high')]+[('LS-RE','LS-FZ'),('FB-W','FB0')]
        for ds in DOMAINS:
            for candidate,anchor in contrasts:
                if (ds,candidate) not in bundles or (ds,anchor) not in bundles:continue
                left,right=bundles[ds,anchor],bundles[ds,candidate]
                require(left['context']==right['context'] and left['identity']==right['identity'],'paired actual runtime/position/input identity differs')
                grouped={}
                for identity,(cluster,value) in left['scores'].items():
                    require(right['scores'][identity][0]==cluster,'paired source cluster differs')
                    entry=grouped.setdefault(cluster,[0.,0]);entry[0]+=right['scores'][identity][1]-value;entry[1]+=1
                values=np.asarray(list(grouped.values()));sums,counts=values[:,0],values[:,1]
                rng=np.random.default_rng(20261119);draws=rng.integers(0,len(values),size=(2000,len(values)))
                ci=np.percentile(sums[draws].sum(1)/counts[draws].sum(1),[2.5,97.5]).tolist()
                pairs.append({'dataset':ds,'candidate':candidate,'anchor':anchor,'QA':COUNTS[ds],
                    'difference_pp':100*float(sums.sum()/counts.sum()),'ci95_pp':[100*x for x in ci],
                    'unadjusted_descriptive_not_equivalence_or_algorithm_winner':True})
    actualclosed=sum(r['disposition']=='CLOSED_COMPLETE' for r in quality)
    require(actualclosed==index['phase']['closed_cells'] and sum(COUNTS[r['dataset']] for r in quality if r['disposition']=='CLOSED_COMPLETE')==index['phase']['closed_quality_answer_events'],'false published closed-quality count')
    require(index['publication_eligible'] is True,'unsealed/ineligible package')
    return {'status':'PASS_PUBLIC_ROUND10_EXPORT_VERIFIED_V1','public_files':len(entries),'public_payload_bytes':total,
        'jsonl_line_identities':line_count,'closed_quality_cells':actualclosed,'declared_cells':28,
        'stopped_unrun_or_failed_cells':28-actualclosed,'quality_recomputations':quality,'physical_owner_rows':physical_count,
        'paired_CI_recomputations':pairs,'calibration_scalar_recomputation':calibration_result,'archive_verified':bool(archive),
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
