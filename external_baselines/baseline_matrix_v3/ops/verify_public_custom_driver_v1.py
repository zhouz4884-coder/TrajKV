"""Cold verification of separately counted primary/auxiliary phase closures."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import zipfile

sys.dont_write_bytecode=True

# Explicit R7 source branch; canonical V3 branch below is unchanged.
import math
R7_SCHEMA='trajkv.r7_research.frozen_matrix500.v2'
R7_POLICY_FILE='results/publication/R7_custom_driver_support_v2/R7_SOURCE_POLICY.json'
R7_SOURCE_PINS={'code/r7_frozen_pilot_v2.py': '3ff896405be5ac91067bdbe8d736c0552e46f835dea90a93c9835ceeb8fd3a91', 'code/r7_legacy_bridge_v2.py': '608b1660896c8e8fd05484be47acf73e99f260db71731239937d0279ed842dad', 'code/r7_legacy_bridge_v1.py': 'b97241c42cdaea96209ba81644b16483b04784e8202080469635184010caa48b', 'code/dataset_registry.py': '4aaf39f4d0725080e66a926754ba50c33247f115253c36e55433805d79bf0983', 'code/cache_bridge.py': '36b860522276895dd361120d315d745a785fd020cbddbff9e75051e26a907025', 'code/attention_bridge.py': 'eef35d19dfc04fab12f0b6f9ba618a8b34ed48f4488edd566ea4578fb3311876', 'code/evaluate.py': 'dbb556d87827c146e596b1e01de6c58a7a8b08f788c4c13d18129f4b1f13f75c', 'code/model_registry.py': '969b39c1df27b63166f4b783187084022cf8548799a6c08178b8665b54d13c58', 'code/ownership.py': '8443adca9e991663e42da6eff0609b54eb7ef0b13fee8ac09f4e12d1894a8456', 'code/m1_protocol.py': '4ba12190fe3c6513c4a45822c4294c3e40cf82ea13ac791c6cb677aed3871427'}
R7_ROLE_PINS={'a2/round7_key_bank.py': '38b6e0ea2486a900e404fa6347c91972198795f8972dc7ab2bf90527fbc1cdc5', 'a2/round7_key_attention.py': 'b054e8629083b7685ad9e02289b500008b45fbeadbd2bc48455f91d0b6a728a0', 'a2/round7_key_codec.py': '80c2c16220097242b25d3f843a3b57569590131c02618e766e35f0f0791ab651', 'a2/round7_key_layout.py': '7b9773f65103062dd51c8ccc64d9b5047f88112d951b6eb10e7938f8fc3f5a04', 'a2/round7_key_artifacts.py': 'f429aa502dae27b7616e8dfca63ac6d498d16f18d8bf2db4ad674b5039760485', 'a2/round6_depth_codec.py': '333ef93165fd786bc40015f6f8c775ce3de20bb98da26ffa6555c38cba5f12d4', 'a2/fast_wht.py': '9dc73998f66d5b7bafc96a5aa3db57974d1ca8e604e432e7539625f104d90ee3', 'a2/quantizers.py': '314bc458c9c2d84579b2a02cb8a45908e0099b613368be40a07669ca5d273e38', 'a2/affine_rate_bank.py': '8090240cd1ee5edde07d1252c41f4dfcaa8aebf167a2b7d890d37ae57edcd2c0', 'a2/masked_bank.py': '7c313fddfdadc81939235b0a4fd843540961f8841c69d7400101b33a66737d9d', 'a2/metadata_direction_bank.py': 'bea958a75c8df84702b20ba32836ae6f4f7ffaae8f6a1024ea6de3c18659ff61', 'experiments/bank.py': 'f1ab22bf04f592263246bac82c415fd9a2b8c6adb4d813a5b46d056f08987ee5', 'experiments/fused_progressive.py': 'd0130e31d055f2c2ced6de1cc26a5cbfa99eea0642eb60ce5f4b6c359cd6520c', 'experiments/check_fused_progressive.py': 'b7578afafcb4bc522bc0742b99e4ee5ab49307fb120b3475dc553900076dac55', 'code/bytes.py': '00625b902a29027075c13d64ca9019753e4aa153215aea6f349d25902a848115', 'scripts/p185_model_state.py': '437dbb5e9bd34e699ddc1f589834d207248074e214dfa821143781cd4fe3c4af', 'scripts/p185_single_state.py': 'ca4bce75c1e80c0a99eb18451323f2d0c2a07a8c707a3238f1a4e8d5d9114b3c', 'scripts/p185_segmented_kernel.py': 'bf09b996f098aea5027f25de54b50aef78f305225de105d9b8c9343751eb3f64', 'scripts/p176_kivi_codec.py': '1c278ebd1f648170696208ffed89a01e96960c18670610948d742262ad6b7cbb', 'scripts/p201_refinement.py': 'a1b8b84b9dbc655c21ead0bffa032be1ae4af97b5da4768112fe2a6106e58afb', 'scripts/p202_refinement.py': '292aca3d0f793c164b1d4204af21573cd4092597eb3f25055ded6d879afa04aa', 'kivi_source': '3f1140b0faff2cb524a3eb684167e403dffd488477b301826497462fd36fe540', 'a2/round6_depth_bank.py': '1d423bf03cdc76c0650776230d6a1093710e9c9fb8ac2cd84974d791aa65c6f7', 'a2/round6_depth_attention.py': 'f96277c726c39f2880e8618525b4bdcbac0049e68e7c4d3e2eb699fa24dfd945'}
R7_MANIFEST_SHA='5d4d4c0f7a02604882ff3be393cda934f0e4f806bdffeed3fbab76a69f7a87fe'
R7_TOKENIZER_SHA='b336af8282760f5e596726eb9834ce20dd6625d55aa7df73030ef4c59028567f'
R7_ASSETS={'r7_joint_k1_gt_cond': {'direction_book_path': 'af29f26ddcbdf4ac6f098c3df497d2d741f0e7909c13ecc090194446846c9589', 'metadata_factor_path': '8f47c56af5fe5e4da99bae05b83084173c5a55601c46c91130894075907e9cea', 'round7_policy_path': '505dbbbc67d50023ec0f60fb31ab9a31b0a27212fe7c3e8a2e72de0307b3f38f'}, 'r7_budget_k2_cond': {'direction_book_path': '9d3ad3683cf6888c30f3c8ba9c6dd0d4dbcf40106a34567d578479245e7309ae', 'metadata_factor_path': 'd60fe4ffcf31e36306ed2c5a4069634627c83b3af2f0b86fdb5c37e456113671', 'round7_policy_path': 'e8a58e602c047adbd4ca6ae253f6ba2eb3961158d7c75444d3e27a41ec47ed6e'}}
R7_BASELINE39={'configs/eval/M1_F_textvqa_p2_500_v1.json': 'f7ddf90ae551e424ce1801e0dcf6ab33e819ef9796d2e44438004732078a4e91', 'configs/eval/M1_F_infovqa_p2_500_v1.json': '9bf91934d64af0d317c66b1d674144911f8a68b94acca26e6bdabcf3a35a62f0', 'configs/eval/M1_F_docvqa_p2_500_v1.json': '7893bf155224616625b0f457e368d477b9119fae7309cc1bfc6fffcdc11d558e', 'configs/eval/M1_F_chartqa_p2_500_v1.json': 'cfbdcf3a9128162c2fffa7a9c53fc60d59666bea4c7eaa8abbf849e4d1b336ad', 'configs/eval/M1_K2_R0_textvqa_p2_500_v1.json': 'b0e00f8468693bd73b382ea440aa20dffd5e7147136beeef96056189c68dcd45', 'configs/eval/M1_K2_R0_docvqa_p2_500_v1.json': '7b5820c068060354f05b51168edaa8f2599cf4f4520e598482d115ce978cb23c', 'configs/eval/M1_K2_R0_chartqa_p2_500_v1.json': '2d1424499d43f0c341497b5dd28cfae7c84120ea195e0fb35e2a0285799c2b7c', 'configs/eval/M1_K4_R0_textvqa_p2_500_v1.json': 'cc702946efa8107c4eaf47024b5b3011663c93aa6f6c9a3287e596d0ef50fbd4', 'configs/eval/M1_K2_g32R32_textvqa_p2_500_v1.json': 'f57462f89a773641a36f66c801812cf0a92747339a315b032eca3bdf5533f9d2', 'configs/eval/M1_K2_g32R32_infovqa_p2_500_v1.json': '59a3943ce1938cd8a67f51015553ba1a47bc5dbdfe5e31e7731e72d6943440d2', 'configs/eval/M1_K2_g32R32_docvqa_p2_500_v1.json': '691c20cdf1c85c23224cb02f94b7cb14de54b63f6899515dc246576da18e01e5', 'configs/eval/M1_K2_g32R32_chartqa_p2_500_v1.json': '154934e6170e6ade2eafcb2e5d77a21eb8a9478d8c849acfd804fdd65e89e2ac', 'configs/eval/M1_C1_bias32_textvqa_p2_500_v1_FP32Affine_Text32_v2.json': '377b9fffb6b5507d04944d4af7f643352c833a0bd408ec13175dfcad36f2580b', 'configs/eval/M1_C1_bias32_infovqa_p2_500_v1_FP32Affine_Text32_v2.json': '7bad4e385608efa424cb9f0c8c0e94883972febeda3a6ab4c02d1bcfb7906418', 'configs/eval/M1_C1_bias32_docvqa_p2_500_v1_FP32Affine_Text32_v2.json': '47580d27e409048a4264ac36917807c425c2260bd2acc701bdb8e6533e0f45bd', 'configs/eval/M1_C1_bias32_chartqa_p2_500_v1_FP32Affine_Text32_v2.json': '32a6db6b84e9f7ec3152e8e5e64a9a1d88b6eea35d76087ade0dc59520b6b389', 'configs/eval/M1_C2_bias32_textvqa_p2_500_v1_FP32Affine_Text32_v2.json': 'cafbb82ad112fdac79c758a5b994c0fa5d96bc285d4125a07aa487c8fe407a02', 'configs/eval/M1_C2_bias32_infovqa_p2_500_v1_FP32Affine_Text32_v2.json': '7657c99d889b2a5fe30a3f859b05e4f03eb51300723d36fea2f3fd6757990aff', 'configs/eval/M1_C2_bias32_docvqa_p2_500_representative_v1_FP32Affine_Text32_v2.json': 'd3ccfa573f4a5988e7d76b796a756f8c603807c052f45ca799437c2d9746a2e4', 'configs/eval/M1_C2_bias32_chartqa_p2_500_representative_v1_FP32Affine_Text32_v2.json': '7f8eba2cd868cc4ba009add2854e30112b0e0cb8add3b11ca3bebac53e1c9c82', 'configs/eval/M1_E_r0888_textvqa_p2_500_v1.json': 'b9c1aabbd3f057166dba9b6b2de7fdaa2cb40f4060a9f86dce60791cb719afb2', 'configs/eval/M1_E_r0888_docvqa_p2_500_v1.json': '7ff478548a0d14a728a39df4a4cffea227a4dcf602ad002654bb1053fc0de693', 'configs/eval/M1_E_r0888_chartqa_p2_500_v1.json': 'ea51e951c9b2e495f151ba923a2e7bdab84d1c95f95642142d36494b688f1714', 'configs/eval/M1_E_w8_drop0888_textvqa_p2_500_v1.json': 'f143e14691abef9da6f0761c48df7648029157d98270c30b8c82b65a7b3ef87b', 'configs/eval/M1_E_w8_drop0888_infovqa_p2_500_v1.json': '5a06e9bac147512eb683be766d4e1042713cc82c85e151d9af1dd4c915b14479', 'configs/eval/M1_E_w8_drop0888_docvqa_p2_500_representative_v1.json': 'db8d4c0d5d0dea3f9f307512b030352aa4452651e82dc8b32889ed64a6c6a6a8', 'configs/eval/M1_E_w8_drop0888_chartqa_p2_500_representative_v1.json': '9587869c185afb1c38bc1ab74c6d059f78fa27e1fa3a5562cb5a2c35f3b63668', 'configs/eval/M1_E_w8_drop091875_textvqa_p2_500_v1.json': '7b417e869e594ff3e470de61ae0ad0cbb2ea742c70399f8caf5fa243afc450aa', 'configs/eval/M1_E_w8_drop091875_infovqa_p2_500_v1.json': 'eef5d48078f24b56b7d1c21dad5b8f356a22580725669a8897a5c90e9ce953eb', 'configs/eval/M1_E_w8_drop091875_docvqa_p2_500_lowcompletion_target_v1.json': '604d43e8f821156a606145edd3bbec966af484d930600486a5a641a0de0bcbc9', 'configs/eval/M1_E_w8_drop091875_chartqa_p2_500_lowcompletion_target_v1.json': '8c3ad9081fc3f83dc27f2cef13ccaa56ff41dbd6425b1dda12394dc802be5605', 'configs/eval/M1_OT_Hmix12_textvqa_p2_500_target_v4.json': 'f705a575ec24e392ad1caf856311dc12610c201aa0c5bf8395107e6975905503', 'configs/eval/M1_OT_Hmix12_infovqa_p2_500_target_v4.json': '812cc893c36f5a70bbd73a1e5739ffb9a037d58c49fb8015aafba82653096d44', 'configs/eval/M1_OT_Hmix12_docvqa_p2_500_lowcompletion_target_v1.json': '34448f2b5e1ecc42dc1b1e20123062b487746a7f4db51f247fb216e7906825a5', 'configs/eval/M1_OT_Hmix12_chartqa_p2_500_lowcompletion_target_v1.json': '19ad65d8532421b490a851b20140377be2abf66e6367ad69e3f3159c7672a773', 'configs/eval/M1_OT_Hmix23_textvqa_p2_500_target_v6.json': '966ed14257f2c26c2499eff12b9b76202b9887162d064a72ec99c26ee1c63c95', 'configs/eval/M1_OT_Hmix23_infovqa_p2_500_target_v6.json': 'acfdd84feebccfc8ce159d5d0155b6b07a97a4ddea76e288afd60b5a69d18131', 'configs/eval/M1_OT_Hmix23_docvqa_p2_500_target_v7.json': 'bccb47c5ffc3ff57282aa647b795b2b57a226ded131fa810389343ddcdb8f2e5', 'configs/eval/M1_OT_Hmix23_chartqa_p2_500_target_v7.json': '5e5df7e88b7a4cfdb6015e6320a66f707650032d86cf22e355c2d28681a69417'}
DTYPE_BYTES={'torch.float16': 2, 'torch.bfloat16': 2, 'torch.float32': 4, 'torch.float64': 8, 'torch.int64': 8, 'torch.int32': 4, 'torch.int16': 2, 'torch.int8': 1, 'torch.uint8': 1, 'torch.bool': 1}


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
def relative(value):
    value=str(value).replace('\\','/')
    for marker in ('/data/registry/v3/','/data/refs/v3/'):
        if marker in value:return 'configs/datasets/v3/'+value.split(marker,1)[1]
    for marker in ('/results/','/configs/'):
        if marker in value:return marker.strip('/')+'/'+value.split(marker,1)[1]
    if value.startswith(('results/','configs/')):return value
    raise ValueError('Unresolvable public evidence path template')


def io_path(value):
    path=Path(value).resolve();prefix=chr(92)*2+'?'+chr(92)
    return Path(prefix+str(path)) if os.name=='nt' and not str(path).startswith(prefix) else path


def verify(root,archive=None,recompute_scores=False):
    root=io_path(root);index=load(root/'PUBLIC_CONTENT_REFS.json');entries=index['files']
    require(index['schema'] in ('baseline_matrix.public_content_refs.v2','baseline_matrix.public_content_refs.v3'),
            'Expected source/public identity schema')
    require(index['completion_scope']=='declared_matrix_only' and index['whole_external_library_completion_assessed'] is False,'Unsupported whole-library completion claim')
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
    sys.path.insert(0,str(root/'code'))
    def public_module(name,path):
        spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec)
        sys.modules[name]=obj;spec.loader.exec_module(obj);return obj
    # Use the verified public modules even when this verifier is called from
    # a process that already imported private workspace cache/metric modules.
    cache=public_module('_verified_public_prediction_cache_v2',root/'code/prediction_cache.py')
    previous_vendor=sys.modules.get('vendor_textvqa_metric')
    public_module('vendor_textvqa_metric',root/'code/vendor_textvqa_metric.py')
    metric=public_module('_verified_public_metrics_v2',root/'code/metrics.py');score_question=metric.score_question
    if previous_vendor is not None:sys.modules['vendor_textvqa_metric']=previous_vendor
    import numpy as np
    raw_meta={};raw_lines={};contracts={};visited=set();active=set()
    def entry(folder,name):
        path=folder+'/'+name;require(path in entries,'Evidence closure missing: '+path);return entries[path]
    def original_hash(folder,name):return entry(folder,name)['source_sha256']
    def original_graph(folder,name,expected):require(original_hash(folder,name)==expected,'Original SHA graph mismatch: '+folder+'/'+name)
    def source_file_reference(value):
        value=str(value).replace('\\','/')
        for alias,origin in [('${PROJECT_ROOT}/','project'),('${EVIDENCE_ROOT}/','evidence_mirror')]:
            if value.startswith(alias):
                matches=[p for p,e in entries.items() if e['source_origin']==origin and e['source_path']==value[len(alias):]]
                require(len(matches)==1,'Unresolved declared source file');return matches[0]
        if value in entries:return value
        candidate=relative(value)
        require(candidate in entries,'Declared source file missing');return candidate
    groups=[{'phase':index['phase'],'phase_artifacts':index['phase_artifacts'],
             'declared_matrix_file':index.get('declared_matrix_file'),
             'source_closure':index.get('primary_source_closure',index['source_closure'])}]+index.get('auxiliary_phases',[])
    require(len({g['phase']['phase_label'] for g in groups})==len(groups),'Duplicate independent phase labels')
    artifacts=[];declared={};artifact_group={}
    for group_number,group in enumerate(groups):
        phase=group['phase']
        require(phase['completion_scope']=='declared_matrix_only' and phase['whole_external_library_completion_assessed'] is False,
                'Unsupported auxiliary whole-library claim')
        matrix_name=group.get('declared_matrix_file')
        if matrix_name is None:
            candidates=[name for name,info in entries.items() if info['source_sha256']==phase['declared_matrix_source_sha256']]
            require(candidates,'Declared matrix original SHA not represented');matrix_name=candidates[0]
        require(entries[matrix_name]['source_sha256']==phase['declared_matrix_source_sha256'],'Wrong declared matrix source SHA')
        matrix=load(root/matrix_name);require(matrix['rows'],'Missing declared matrix rows')
        local={source_file_reference(row['config']):row for row in matrix['rows']}
        require(len(local)==len(matrix['rows'])==phase['expected_cells'] and
                sum(int(row['count']) for row in matrix['rows'])==phase['expected_questions'],'Declared independent matrix totals differ')
        require(not set(local)&set(declared),'Duplicate target across phases; shared reference views require their own support group')
        declared.update(local)
        require(len(group['phase_artifacts'])==len({a['config'] for a in group['phase_artifacts']}),'Duplicate phase artifact')
        require(all(a['config'] in local for a in group['phase_artifacts']),'Artifact lies outside its declared independent phase')
        for artifact in group['phase_artifacts']:
            artifacts.append(artifact);artifact_group[artifact['config']]=group_number
    def line_metadata(folder):
        path=folder+'/records.jsonl';side=entry(folder,'records.jsonl')['line_identities_file'];rows=[]
        declared_manifest=load(root/folder/'manifest.json')
        with (root/path).open('rb') as stream,(root/side).open(encoding='utf-8') as identity:
            for offset,line in enumerate(stream,1):
                row=json.loads(line);proof=json.loads(identity.readline())
                if declared_manifest.get('runner_schema')==R7_SCHEMA:
                    r7_verify_record_role_pins(declared_manifest['config'],row)
                rows.append({'id':row['id'],'status':row['status'],'source_line_sha256':proof['source_sha256'],
                    'public_line_sha256':proof['public_sha256'],'input_key_sha256':cache.input_key(row),
                    'execution_sha256':cache.execution_fingerprint(row),
                    'full_input_fingerprint':canonical({k:row[k] for k in ('original_image_sha256','input_ids_sha256','full_prompt_hash',
                        'question_suffix_sha256','native_position_ids_sha256','native_position_ids_shape','visual_positions',
                        'actual_visual_token_count','input_tokens','compression_boundary','question_suffix_tokens')})})
        return rows
    def check_proof(folder,name,required,status):
        proof=load(root/folder/name)
        require(proof['status']==status and proof['remote_hashes_computed_independently'] is True,'Missing independent original proof')
        values={p['file']:p for p in proof['files']};require(set(values)==set(required),'Proof file-set mismatch')
        for filename in required:
            original_graph(folder,filename,values[filename]['sha256'])
            require(entry(folder,filename)['source_bytes']==values[filename]['bytes'],'Original proof length mismatch')
    def public_contract(manifest,inventory,source_eval):
        identity=manifest['model_identity']
        return {'schema':'baseline_matrix.effective_prediction_contract.v1',
            'behavior':cache.normalize_behavior(manifest['config'],manifest['model_spec']),'model_spec':manifest['model_spec'],
            'model_files':sorted(({k:r[k] for k in ('file','bytes','sha256')} for r in identity['files']),key=lambda r:r['file']),
            'runtime':{k:manifest[k] for k in ('torch_version','transformers_version','cuda_version','python_version','hardware','runtime_executable')},
            'processor_id':manifest['processor_id'],'processor_version':manifest['processor_version'],
            'processor_file_hashes':manifest['processor_file_hashes'],'generation_native':manifest['generation_native'],
            'generation_overrides':manifest['generation_overrides'],'relevant_source_hashes':cache._dependencies(manifest,inventory,source_eval)}
    custom_folders=set()
    def verify_R7_source(folder,m,c):
        require(m.get('runner_schema')==R7_SCHEMA,'Unsupported custom/R9 producer; no canonical fallback')
        require(m['expected_QA']==c['expected_QA']==c['recorded_QA']==500 and c['counts']=={'PASS':500},'Custom R7 requires complete500')
        require(R7_POLICY_FILE in entries,'Missing exact custom source policy')
        policy=load(root/R7_POLICY_FILE)
        require(policy['required_baseline39_config_sha256']==R7_BASELINE39,'Frozen39 auxiliary source policy differs')
        require(policy['schema']=='trajkv.r7_public_source_policy.v1' and policy['runner_schema']==R7_SCHEMA and
                policy['SourceRun_admission'] is False and policy['canonical_cache_admission'] is False,'False custom admission policy')
        require(policy['required_source_hashes']==R7_SOURCE_PINS and
                {k:v['sha256'] for k,v in policy['required_role_sources'].items()}==R7_ROLE_PINS,'Custom source whitelist changed')
        original_manifest=policy['fixed8_manifest_file'];require(entries[original_manifest]['source_sha256']==R7_MANIFEST_SHA,'Wrong frozen8 manifest')
        frozen=[r for r in load(root/original_manifest)['rows'] if r['tier']=='fixed500']
        require(len(frozen)==8 and {(r['profile'],r['dataset']) for r in frozen}=={(p,d) for p in ('K1','K2') for d in ('textvqa','infovqa','docvqa','chartqa')},'Exact8 scope differs')
        proof=load(root/folder/'source_retrieval_receipt_r7_v2.json')
        require(proof['status']=='PASS_R7_EXECUTED_DRIVER_AND_DEPENDENCY_SOURCE_SHA' and proof['remote_hashes_computed_independently'] is True and
                proof['SourceRun_admission_established'] is False and proof['not_source_evaluate_alias'] is True and
                proof['actual_driver']=='code/r7_frozen_pilot_v2.py','Invalid custom source receipt')
        required={'source_inventory.json','launch_receipt.json','source_config.json','source_job.json','source_driver.py'}|{'source_files/'+name for name in R7_SOURCE_PINS}
        proof_files={x['file']:x for x in proof['files']}
        require(len(proof_files)==len(proof['files']) and required<=set(proof_files),'Missing/duplicate custom proof file')
        for name,item in proof_files.items():
            pure=Path(name)
            require(not pure.is_absolute() and '..' not in pure.parts and item['bytes']<=4*1024**2 and
                    (name in required or name.startswith('legacy_sources/') and name.endswith('.py')),'Invalid custom proof file scope')
            original_graph(folder,name,item['sha256']);require(entry(folder,name)['source_bytes']==item['bytes'],'Custom proof byte size differs')
        require(proof['executed_legacy_source_sha256']==c['executed_legacy_source_sha256'],'Actual legacy graph differs')
        legacy=[x for x in proof['files'] if x['file'].startswith('legacy_sources/')]
        require(len(legacy)==proof['legacy_source_files'] and
                {x['remote_file']:x['sha256'] for x in legacy}==proof['executed_legacy_source_sha256'],'Legacy graph not fully archived')
        for role,item in policy['required_role_sources'].items():
            require(item['file'] in entries and entries[item['file']]['source_sha256']==R7_ROLE_PINS[role],'Explicit dynamic/role source missing')
        inventory=load(root/folder/'source_inventory.json');launch=load(root/folder/'launch_receipt.json');job=load(root/folder/'source_job.json');cfg=load(root/folder/'source_config.json')
        r7_verify_executed_roles(cfg,proof['executed_legacy_source_sha256'])
        match=[r for r in frozen if r['config_sha256']==original_hash(folder,'source_config.json') and r['job_sha256']==original_hash(folder,'source_job.json')]
        require(len(match)==1 and cfg==m['config'] and m['config_sha256']==match[0]['config_sha256'],'Custom config/job is not frozen8')
        selected=match[0];base=job['argv'][job['argv'].index('--output')+1].split('/results/',1)[0]
        snapshot=base+'/jobs/'+job['name']+'.source';expected=[snapshot+'/code/r7_frozen_pilot_v2.py' if x==base+'/code/r7_frozen_pilot_v2.py' else x for x in job['argv']]
        require(type(launch['pid']) is int and launch['pid']>0 and launch['name']==job['name']==proof['producer_name'] and
                launch['job_sha256']==selected['job_sha256'] and launch['argv_executed']==expected and
                launch['source_snapshot']==m['runtime_source_root']==proof['source_snapshot']==snapshot and
                launch['runtime_python']==m['runtime_executable']==job['argv'][0] and
                str(launch['CUDA_VISIBLE_DEVICES'])==str(job['env']['CUDA_VISIBLE_DEVICES'])==str(selected['GPU']),'Custom producer/runtime binding differs')
        original_graph(folder,'source_inventory.json',launch['source_inventory_sha256'])
        require(m['source_hashes']==R7_SOURCE_PINS,'Custom declared source set differs')
        for name,digest in R7_SOURCE_PINS.items():
            require(inventory[name]==digest,'Snapshot numeric dependency differs');original_graph(folder,'source_files/'+name,digest)
        original_graph(folder,'source_driver.py',R7_SOURCE_PINS['code/r7_frozen_pilot_v2.py'])
        profile=cfg['variant']['name'];require(profile in R7_ASSETS and m['asset_sha256']==R7_ASSETS[profile]==cfg['expected_asset_sha256'],'Profile/assets changed')
        origin=cfg['variant_source'];origin_file=source_file_reference(origin['path'])
        require(entries[origin_file]['source_sha256']==origin['sha256']=='aba87db8c87bd1ca8ca71685baa7a942c0450c8ecc1302c9ccbe183c67703ef4' and
                m['variant_original_source']==origin,'Original profile source identity differs')
        original_variants=[x for x in load(root/origin_file)['variants'] if x['name']==profile]
        exclude=set(R7_ASSETS[profile]);numeric=lambda v:{k:x for k,x in v.items() if k not in exclude}
        require(len(original_variants)==1 and numeric(original_variants[0])==numeric(cfg['variant']) and
                m['frozen_variant']==cfg['variant'],'Original frozen profile numeric fields changed')
        require(policy['profiles'][profile]['variant']==cfg['variant'] and policy['profiles'][profile]['method_id']==cfg['method']['id'],'Frozen profile numeric fields differ')
        for name,item in policy['profiles'][profile]['assets'].items():
            require(item['file'] in entries and entries[item['file']]['source_sha256']==R7_ASSETS[profile][name],'Frozen asset bytes absent')
        token=policy['tokenizer_cleanup_proof'];require(entries[token['file']]['source_sha256']==token['sha256']==R7_TOKENIZER_SHA and load(root/token['file'])['clean_up_tokenization_spaces'] is False,'Baseline implicit cleanupFalse not proven')
        require(m['decode_overrides']=={'clean_up_tokenization_spaces':False,'skip_special_tokens':True} and
                m['processor_file_hashes']['tokenizer_config.json']==R7_TOKENIZER_SHA and m['parameter_fit_or_new_algorithm'] is False and
                m['native_FP16_visual_source_retained_for_serving'] is False,'Custom decode/serving contract differs')
        custom_folders.add(folder)
    def walk(folder):
        if folder in visited:return
        require(folder not in active,'Cyclic public donor closure');active.add(folder)
        m=load(root/folder/'manifest.json');c=load(root/folder/'complete.json')
        original_graph(folder,'manifest.json',c['manifest_sha256']);original_graph(folder,'records.jsonl',c['records_jsonl_sha256'])
        require(c['coverage_complete'] is True and c['quality_scored'] is False and m['references_read'] is False,'Raw/view reference-free coverage contract differs')
        rows=line_metadata(folder);raw_lines[folder]=rows
        require([r['id'] for r in rows]==m['expected_ids'] and len(rows)==len({r['id'] for r in rows})==m['expected_QA']==c['expected_QA']==c['recorded_QA'],'Raw/view coverage/order mismatch')
        require(dict(Counter(r['status'] for r in rows))==c['counts'],'Raw/view status counts mismatch')
        if m['schema']=='baseline_matrix.derived_prediction_view.v1':
            require(m['inference_run'] is False and m['prediction_cache_used'] is True and c['derived_prediction_view'] is True and c['inference_run'] is False,'Derived view mislabeled as inference')
            original_graph(folder,'provenance.jsonl',c['provenance_jsonl_sha256'])
            donors={}
            for descriptor in m['source_runs']:
                donor=relative(descriptor['raw_dir']);walk(donor);donors[donor]=descriptor
                require(descriptor['original_inference_run'] is True,'Expected original GPU donor')
                for filename,digest in {**descriptor['raw_hashes'],**descriptor['source_proof_hashes']}.items():original_graph(donor,filename,digest)
                original_graph(donor,'retrieval_receipt.json',descriptor['retrieval_receipt_sha256'])
                original_graph(donor,'source_retrieval_receipt_v2.json',descriptor['source_retrieval_receipt_sha256'])
                require(descriptor['effective_contract_sha256']==m['effective_contract_sha256'],'Original effective-contract identity differs')
                require(contracts[donor]==m['effective_contract'],'Public normalized effective contract differs after path templating')
            roles=Counter()
            with (root/folder/'provenance.jsonl').open(encoding='utf-8') as stream:
                for position,row in enumerate(rows,1):
                    proof=json.loads(stream.readline());donor=relative(proof['source_run'])
                    require(donor in donors and proof['role'] in ('fresh','reused'),'Unknown donor/origin role')
                    number=proof['source_line_number'];require(1<=number<=len(raw_lines[donor]),'Invalid original source row number')
                    original=raw_lines[donor][number-1]
                    require(proof['id']==row['id']==original['id'] and row['source_line_sha256']==original['source_line_sha256']==proof['source_exact_line_sha256'],'Original row identity changed across derived view')
                    require(row['public_line_sha256']==original['public_line_sha256'],'Sanitized derived/source row bytes differ')
                    require(proof['source_raw_hashes']==donors[donor]['raw_hashes'],'Per-row donor SHA set differs')
                    require(proof['input_key_sha256']==row['input_key_sha256']==original['input_key_sha256'] and
                            proof['source_execution_sha256']==row['execution_sha256']==original['execution_sha256'],'Input/execution identity differs')
                    require(proof['effective_contract_sha256']==m['effective_contract_sha256'],'Per-row contract differs')
                    roles[proof['role']]+=1
                require(not stream.readline(),'Unexpected extra provenance')
            require(dict(roles)==m['QA_origin_counts']==c['QA_origin_counts'],'Fresh/reused accounting differs')
            contracts[folder]=m['effective_contract']
        elif m.get('runner_schema') is not None:
            require(m['schema']=='baseline_matrix.raw_inference.v1','Unknown custom raw schema')
            check_proof(folder,'retrieval_receipt.json',('manifest.json','records.jsonl','complete.json'),'PASS_REMOTE_RAW_SHA256_EXACT')
            verify_R7_source(folder,m,c)
        else:
            require(m['schema']=='baseline_matrix.raw_inference.v1','Unknown original donor schema')
            check_proof(folder,'retrieval_receipt.json',('manifest.json','records.jsonl','complete.json'),'PASS_REMOTE_RAW_SHA256_EXACT')
            check_proof(folder,'source_retrieval_receipt_v2.json',('source_inventory.json','launch_receipt.json','source_evaluate.py'),'PASS_REMOTE_SOURCE_SHA256_EXACT')
            inventory=load(root/folder/'source_inventory.json');launch=load(root/folder/'launch_receipt.json')
            require(launch['source_inventory_sha256']==original_hash(folder,'source_inventory.json'),'Launch/inventory identity differs')
            require(launch['source_snapshot']==m['runtime_source_root'] and launch['runtime_python']==m['runtime_executable']==launch['argv_executed'][0],'Original runtime/snapshot binding differs')
            require(inventory['code/evaluate.py']==original_hash(folder,'source_evaluate.py'),'Original evaluate source differs')
            require(sha(root/folder/'source_evaluate.py')==inventory['code/evaluate.py'],'Producer evaluate source was transformed; explicit additional AST map required')
            contracts[folder]=public_contract(m,inventory,root/folder/'source_evaluate.py')
        raw_meta[folder]=(m,c);visited.add(folder);active.remove(folder)
    for artifact in artifacts:walk(artifact['raw_dir'])
    require(visited=={n['raw_dir'] for n in index['source_closure']},'Source closure is incomplete or contains unclaimed nodes')
    def reachable(folder,seen):
        if folder in seen:return
        seen.add(folder);manifest=raw_meta[folder][0]
        for descriptor in manifest.get('source_runs',[]):reachable(relative(descriptor['raw_dir']),seen)
    for group in groups:
        own=set()
        for artifact in group['phase_artifacts']:reachable(artifact['raw_dir'],own)
        require(own=={node['raw_dir'] for node in group['source_closure']},'Independent phase donor closure differs')
    require(len(custom_folders)==8,'This standalone verifier requires the exact8 custom R7 originals, not a canonical-only bundle')
    require(all(name in declared and name in {a['config'] for a in artifacts} and entries[name]['source_sha256']==digest for name,digest in R7_BASELINE39.items()),'Required39 completed baseline auxiliary configs missing or changed')
    if custom_folders:
        def actual_context(m):
            runtime={}
            for key,aliases in [('torch_version',('torch_version',)),('transformers_version',('transformers_version',)),
                ('cuda_version',('cuda_version','CUDA_version')),('python_version',('python_version',)),
                ('hardware',('hardware',)),('runtime_executable',('runtime_executable',))]:
                values=[m[x] for x in aliases if x in m]
                require(values and all(x==values[0] for x in values),'Missing/conflicting actual custom runtime field')
                runtime[key]=values[0]
            return {'model_spec':m['model_spec'],'model_files':sorted(({k:r[k] for k in ('file','bytes','sha256')} for r in m['model_identity']['files']),key=lambda r:r['file']),
                'runtime':runtime,**{k:m[k] for k in ('processor_id','processor_version','processor_file_hashes','generation_native','generation_overrides')}}
        anchors={}
        for a in artifacts:
            m=raw_meta[a['raw_dir']][0]
            if m['config']['method']['id']=='F_native_split' and m['expected_QA']==500:
                dataset=declared[a['config']]['dataset'];require(dataset not in anchors,'Duplicate same500 FP16 anchor');anchors[dataset]=a['raw_dir']
        require(set(anchors)=={'textvqa','infovqa','docvqa','chartqa'},'All4 same500 FP16 auxiliary anchors required')
        for folder in custom_folders:
            m=raw_meta[folder][0];a=next(x for x in artifacts if x['raw_dir']==folder);dataset=declared[a['config']]['dataset'];anchor=anchors[dataset]
            require(actual_context(m)==actual_context(raw_meta[anchor][0]),'R7/baseline actual model processor generation runtime differs')
            require([(r['id'],r['full_input_fingerprint']) for r in raw_lines[folder]]==
                    [(r['id'],r['full_input_fingerprint']) for r in raw_lines[anchor]],'R7/baseline image prompt visual positions boundary input differs')
    scoring=[];complete_cells=Counter();complete_questions=Counter()
    source_lock=load(root/'configs/datasets/v3/source_lock.json')
    for artifact in artifacts:
        folder=artifact['raw_dir'];m,c=raw_meta[folder];cfg=load(root/artifact['config'])
        require(artifact['config'] in declared,'Artifact outside declared matrix')
        declaration=declared[artifact['config']]
        require(int(declaration['count'])==artifact['expected_QA'],'Artifact declared count differs')
        if declaration.get('model') is not None:require(cfg['model_id']==declaration['model'],'Declared model differs')
        require(cfg==m['config'],'Declared target/view configuration differs')
        inputs=relative(cfg['inputs']);refs=inputs.replace('_inputs.jsonl','_refs.jsonl')
        for path in (inputs,refs):
            lock=source_lock['files'][Path(path).name];require(entries[path]['source_sha256']==lock['sha256'] and entries[path]['source_bytes']==lock['bytes'],'Original input/reference lock differs')
        input_rows=[json.loads(line) for line in (root/inputs).read_text(encoding='utf-8').splitlines()]
        input_lookup={r['id']:r for r in input_rows};ref_lookup={r['id']:r for r in (json.loads(line) for line in (root/refs).read_text(encoding='utf-8').splitlines())}
        expected=cfg.get('ids') if cfg.get('ids') is not None else [r['id'] for r in input_rows[int(cfg.get('start',0)):int(cfg.get('start',0))+int(cfg.get('count',len(input_rows)))]]
        require(expected==m['expected_ids'] and len(expected)==artifact['expected_QA'],'Declared target cohort differs')
        for row in raw_lines[folder]:require(row['input_key_sha256']==cache.input_key(input_lookup[row['id']]),'Target frozen input differs from raw')
        if not artifact['scored_dir']:continue
        scored=artifact['scored_dir'];summary=load(root/scored/'summary.json')
        require(summary['expected_run_questions']==len(expected),'Scored expected coverage differs')
        require(summary['model_id']==cfg['model_id'] and summary['method']==cfg['method']['id'],'Scored model/method label differs')
        if declaration.get('dataset') is not None:require(summary['dataset']==declaration['dataset'],'Declared/scored dataset differs')
        for name,value in summary['integrity']['raw_files'].items():original_graph(folder,name,value)
        for name,value in summary['derived_file_hashes'].items():original_graph(scored,name,value)
        require(summary['integrity']['refs_sha256']==entries[refs]['source_sha256'] and summary['integrity']['inputs_sha256']==entries[inputs]['source_sha256'],'Scoring input/reference identity differs')
        if m.get('schema')=='baseline_matrix.derived_prediction_view.v1':
            proof=summary['integrity']['derived_source_provenance'];require(proof['inference_run'] is False and proof['origin_counts']==m['QA_origin_counts'],'Scoring provenance label differs')
        if recompute_scores:
            means=[];clusters=defaultdict(list);group_rows=defaultdict(list);failures=0
            with (root/folder/'records.jsonl').open(encoding='utf-8') as raw,(root/scored/'scored_records.jsonl').open(encoding='utf-8') as scores:
                for number,expected_id in enumerate(expected,1):
                    row=json.loads(raw.readline());value=json.loads(scores.readline());identity=raw_lines[folder][number-1]
                    if folder in custom_folders:
                        actual=physical_complete_bytes(row,m['model_spec'])
                        require(abs(16*actual/row['full_kv_byte_denominator']-row['actual_effective_visual_bits'])<1e-12,'Custom complete bits differ')
                        first=row['first_generation_call']
                        require(first['native_first_logits_injected'] is False and first['cache_tokens_before']==row['compression_boundary'] and
                                first['input_tokens']==row['question_suffix_tokens'] and first['input_ids_sha256']==row['question_suffix_sha256'],'Custom suffix/firsttoken differs')
                    require(row['id']==value['id']==expected_id and value['raw_status']==row['status'],'Scored/raw ID/status differs')
                    require(value['raw_jsonl_line_number']==number and value['raw_jsonl_exact_line_sha256']==identity['source_line_sha256'],'Scored original line proof differs')
                    if row['status']!='PASS':require(value['score'] is None,'Fake failure score');failures+=1;continue
                    for field in ('raw_prediction','output_token_ids','original_image_sha256','actual_visual_token_count','stored_visual_KV_bytes'):require(row.get(field)==value.get(field),'Scored/source field differs')
                    score=score_question(row['dataset'],row['raw_prediction'],ref_lookup[row['id']])
                    require(abs(score['score']-value['score'])<1e-12 and score['metric_id']==value['metric_id'] and score['reference_texts']==value['reference_texts'],'Score/reference does not recompute')
                    require(score['group']==value.get('group'),'Scored ChartQA group differs from frozen references')
                    means.append(score['score']);clusters[value['source_cluster_id']].append(score['score'])
                    if score['group'] in ('human','aug'):group_rows[score['group']].append((value['source_cluster_id'],score['score']))
                require(not raw.readline() and not scores.readline(),'Unexpected scored/raw rows')
            if failures:require(summary['primary_mean'] is None,'Failure retained a false full-cohort mean')
            else:
                require(abs(sum(means)/len(means)-summary['primary_mean'])<1e-12,'Mean does not recompute')
                ordered=sorted(clusters);sums=np.asarray([sum(clusters[k]) for k in ordered]);sizes=np.asarray([len(clusters[k]) for k in ordered])
                if len(ordered)>=2:
                    b=summary['bootstrap'];rng=np.random.default_rng(b['seed']);draws=rng.integers(0,len(ordered),size=(b['repetitions'],len(ordered)))
                    interval=np.quantile(sums[draws].sum(1)/sizes[draws].sum(1),[.025,.975])
                    require(np.allclose(interval,summary['primary_ci95'],rtol=0,atol=1e-12),'Cluster CI differs')
                for group,values in group_rows.items():
                    require(group in summary['groups'],'Missing ChartQA subgroup')
                    data=summary['groups'][group];require(data['expected_questions']==len(values),'ChartQA subgroup count differs')
                    require(abs(sum(v for _,v in values)/len(values)-data['primary']['mean'])<1e-12,'ChartQA subgroup mean differs')
                    cluster=defaultdict(list)
                    for key,value in values:cluster[key].append(value)
                    ordered=sorted(cluster)
                    if len(ordered)>=2:
                        sums=np.asarray([sum(cluster[k]) for k in ordered]);sizes=np.asarray([len(cluster[k]) for k in ordered])
                        b=summary['bootstrap'];rng=np.random.default_rng(b['seed']);draws=rng.integers(0,len(ordered),size=(b['repetitions'],len(ordered)))
                        interval=np.quantile(sums[draws].sum(1)/sizes[draws].sum(1),[.025,.975])
                        require(np.allclose(interval,data['primary']['ci95'],rtol=0,atol=1e-12),'ChartQA subgroup CI differs')
        passed=summary['status']=='PASS_SCORED_COMPLETE' and c['status']=='PASS_RAW_COMPLETE'
        group_number=artifact_group[artifact['config']]
        if passed:complete_cells[group_number]+=1;complete_questions[group_number]+=len(expected)
        scoring.append({'cell':artifact['cell'],'questions':len(expected),'mean':summary['primary_mean'],'passed':passed,
                        'phase_label':groups[group_number]['phase']['phase_label'],
                        'kind':artifact['kind'],'origin_counts':m.get('QA_origin_counts',{'fresh':len(expected)}),
                        'metrics_recomputed':bool(recompute_scores)})
    verified_groups=[];all_complete=True
    for group_number,group in enumerate(groups):
        phase=group['phase']
        require(phase['verified_scored_cells']==complete_cells[group_number] and phase['verified_scored_questions']==complete_questions[group_number],
                'Published independent phase progress differs from closed artifacts')
        is_complete=complete_cells[group_number]==phase['expected_cells'] and complete_questions[group_number]==phase['expected_questions'] and not phase['missing']
        require((phase['status']=='PASS_DECLARED_MATRIX_COMPLETE')==is_complete,'False independent phase completion')
        all_complete=all_complete and is_complete
        verified_groups.append({'phase_label':phase['phase_label'],'cells':complete_cells[group_number],
                                'view_questions':complete_questions[group_number],'primary':group_number==0})
    if custom_folders:
        require(len(custom_folders)==8 and groups[0]['phase']['family']=='r7_m1_anchor8_custom_driver_v2' and
                groups[0]['phase']['expected_cells']==8 and groups[0]['phase']['expected_questions']==4000 and
                {a['raw_dir'] for a in groups[0]['phase_artifacts']}==custom_folders,'Custom primary must be exact8/4000')
        require(index.get('custom_driver_provenance_schema')=='trajkv.public_custom_driver.v1','Custom export identity missing')
    eligible=all_complete and not index.get('fixture_only',False)
    require(index['publication_eligible']==eligible,'Incomplete preview/fixture publication eligibility differs')
    phase=index['phase']
    return {'status':'PASS_PUBLIC_CUSTOM_DRIVER_EXPORT_VERIFIED_V1','public_files':len(entries),'public_payload_bytes':total,
        'independent_phases_verified':verified_groups,'phase_group_totals_are_separate':True,
        'supporting_evidence_groups_verified':len(index.get('supporting_evidence_groups',[])),
        'jsonl_line_identities':line_count,'raw_complete_runs_verified':len(visited),
        'original_donors':sum(n['kind']=='original_inference_run' for n in index['source_closure']),
        'derived_views':sum(n['kind']=='derived_prediction_view' for n in index['source_closure']),
        'source_closure_verified':True,'custom_R7_originals':len(custom_folders),'custom_SourceRun_admission_claimed':False,'score_recomputations':scoring,'archive_verified':bool(archive),
        'phase':phase,'publication_eligible':eligible,'fixture_only':bool(index.get('fixture_only',False)),
        'completion_scope':'declared_matrix_only',
        'whole_external_library_completion_assessed':False,'github_upload_verified':False}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--archive',type=Path);parser.add_argument('--recompute-scores',action='store_true');parser.add_argument('--receipt',type=Path)
    args=parser.parse_args();result=verify(args.root,args.archive,args.recompute_scores)
    if args.receipt:
        with args.receipt.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
