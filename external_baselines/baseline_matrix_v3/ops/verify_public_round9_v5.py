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

import math
R9_SCHEMA='trajkv.round9_research.raw.v1'
R9_DRIVERS={'code/round9_runner_v4.py': 'cf4885b77b03fd94aadd5bbabc5a1e6d62e3259ba6037b7a708fabab71758961', 'code/round9_runner_v5.py': 'ed102ab76346ba70a1b5cbddd5887b3f87e4216e837b9b8ff71bad9e23391abb'}
R9_EPOCHS={'code/round9_runner_v4.py': {'code/r7_legacy_bridge_v1.py': 'b97241c42cdaea96209ba81644b16483b04784e8202080469635184010caa48b', 'code/r7_legacy_bridge_v2.py': '608b1660896c8e8fd05484be47acf73e99f260db71731239937d0279ed842dad', 'code/m1_protocol.py': '4ba12190fe3c6513c4a45822c4294c3e40cf82ea13ac791c6cb677aed3871427', 'code/model_registry.py': '969b39c1df27b63166f4b783187084022cf8548799a6c08178b8665b54d13c58', 'code/dataset_registry.py': '4aaf39f4d0725080e66a926754ba50c33247f115253c36e55433805d79bf0983', 'code/evaluate.py': 'dbb556d87827c146e596b1e01de6c58a7a8b08f788c4c13d18129f4b1f13f75c', 'code/cache_bridge.py': '36b860522276895dd361120d315d745a785fd020cbddbff9e75051e26a907025', 'code/attention_bridge.py': 'eef35d19dfc04fab12f0b6f9ba618a8b34ed48f4488edd566ea4578fb3311876', 'code/ownership.py': '8443adca9e991663e42da6eff0609b54eb7ef0b13fee8ac09f4e12d1894a8456', 'code/round9_runner_v4.py': 'cf4885b77b03fd94aadd5bbabc5a1e6d62e3259ba6037b7a708fabab71758961', 'code/round9_value_bank_v1.py': '14d242a6643a738457c9b8593d0f0eb6e383c5e979b2e196fcca28e7d37a5f2e', 'code/round9_inputs_v1.py': '46ccb0d3fffe93994b1339474887ecf86b730eb3edd9f7cee9fa4d8b96e68ac9', 'code/round9_extended_bank_v1.py': '57ec77e7c2d7f3b8abbd3f9fcaaa996c60228a2821d4085dcaf3604a98ba6001', 'code/round9_extended_bank_v2.py': '5745e4cda9c1c5d1e2256a49908c15a350321834e1d0d36097a7fac99734acff', 'code/round9_key_controls_v1.py': 'e73e8594e80f7b3b14dfc6bc3c271858dc42798de9f67843de14b7376f474878', 'code/round9_key_controls_v2.py': '44500c370f4605e377750fc003b1dfeff052f129f122a7fabb1ba3c70d4dfdf4', 'code/round9_precision_exchange_v1.py': '15e075936d235cdd0a3935961b8083c2f10c9df59c545f1a6bd2c7aecd3dd9fe'}, 'code/round9_runner_v5.py': {'code/r7_legacy_bridge_v1.py': 'b97241c42cdaea96209ba81644b16483b04784e8202080469635184010caa48b', 'code/r7_legacy_bridge_v2.py': '608b1660896c8e8fd05484be47acf73e99f260db71731239937d0279ed842dad', 'code/m1_protocol.py': '4ba12190fe3c6513c4a45822c4294c3e40cf82ea13ac791c6cb677aed3871427', 'code/model_registry.py': '969b39c1df27b63166f4b783187084022cf8548799a6c08178b8665b54d13c58', 'code/dataset_registry.py': '4aaf39f4d0725080e66a926754ba50c33247f115253c36e55433805d79bf0983', 'code/evaluate.py': 'dbb556d87827c146e596b1e01de6c58a7a8b08f788c4c13d18129f4b1f13f75c', 'code/cache_bridge.py': '36b860522276895dd361120d315d745a785fd020cbddbff9e75051e26a907025', 'code/attention_bridge.py': 'eef35d19dfc04fab12f0b6f9ba618a8b34ed48f4488edd566ea4578fb3311876', 'code/ownership.py': '8443adca9e991663e42da6eff0609b54eb7ef0b13fee8ac09f4e12d1894a8456', 'code/round9_runner_v5.py': 'ed102ab76346ba70a1b5cbddd5887b3f87e4216e837b9b8ff71bad9e23391abb', 'code/round9_value_bank_v1.py': '14d242a6643a738457c9b8593d0f0eb6e383c5e979b2e196fcca28e7d37a5f2e', 'code/round9_inputs_v1.py': '46ccb0d3fffe93994b1339474887ecf86b730eb3edd9f7cee9fa4d8b96e68ac9', 'code/round9_extended_bank_v1.py': '57ec77e7c2d7f3b8abbd3f9fcaaa996c60228a2821d4085dcaf3604a98ba6001', 'code/round9_extended_bank_v2.py': '5745e4cda9c1c5d1e2256a49908c15a350321834e1d0d36097a7fac99734acff', 'code/round9_key_controls_v1.py': 'e73e8594e80f7b3b14dfc6bc3c271858dc42798de9f67843de14b7376f474878', 'code/round9_key_controls_v2.py': '44500c370f4605e377750fc003b1dfeff052f129f122a7fabb1ba3c70d4dfdf4', 'code/round9_precision_exchange_v1.py': '15e075936d235cdd0a3935961b8083c2f10c9df59c545f1a6bd2c7aecd3dd9fe', 'code/round9_precision_exchange_v2.py': '7658d920970a9eba23b4edaed2c110e523ca66326aacafb7031ca188aa37bd99'}}
R9_POLICY_FILE='results/publication/R9_newsource_replication_support_v5/R9_SOURCE_POLICY.json'
R9_MANIFEST_SHA='01f02913da791a21d390262ebd162ded6daaec2abcd7d4d5246c0508d863155a'
R9_CAL_COMPLETE_SHA='c838b3a79cb970233a9579f8972ae4276157fad1001fd9b4e9118945bc0dc521'
R9_CAL_INPUT_SHA='2de0d5308892ee86bee48c3205c6eb49eee196ffee32ef6f8bcab19a30b6c216'
R9_BASE_BOOK='af29f26ddcbdf4ac6f098c3df497d2d741f0e7909c13ecc090194446846c9589'
R7_ROLE_PINS={'a2/round7_key_bank.py': '38b6e0ea2486a900e404fa6347c91972198795f8972dc7ab2bf90527fbc1cdc5', 'a2/round7_key_attention.py': 'b054e8629083b7685ad9e02289b500008b45fbeadbd2bc48455f91d0b6a728a0', 'a2/round7_key_codec.py': '80c2c16220097242b25d3f843a3b57569590131c02618e766e35f0f0791ab651', 'a2/round7_key_layout.py': '7b9773f65103062dd51c8ccc64d9b5047f88112d951b6eb10e7938f8fc3f5a04', 'a2/round7_key_artifacts.py': 'f429aa502dae27b7616e8dfca63ac6d498d16f18d8bf2db4ad674b5039760485', 'a2/round6_depth_codec.py': '333ef93165fd786bc40015f6f8c775ce3de20bb98da26ffa6555c38cba5f12d4', 'a2/fast_wht.py': '9dc73998f66d5b7bafc96a5aa3db57974d1ca8e604e432e7539625f104d90ee3', 'a2/quantizers.py': '314bc458c9c2d84579b2a02cb8a45908e0099b613368be40a07669ca5d273e38', 'a2/affine_rate_bank.py': '8090240cd1ee5edde07d1252c41f4dfcaa8aebf167a2b7d890d37ae57edcd2c0', 'a2/masked_bank.py': '7c313fddfdadc81939235b0a4fd843540961f8841c69d7400101b33a66737d9d', 'a2/metadata_direction_bank.py': 'bea958a75c8df84702b20ba32836ae6f4f7ffaae8f6a1024ea6de3c18659ff61', 'experiments/bank.py': 'f1ab22bf04f592263246bac82c415fd9a2b8c6adb4d813a5b46d056f08987ee5', 'experiments/fused_progressive.py': 'd0130e31d055f2c2ced6de1cc26a5cbfa99eea0642eb60ce5f4b6c359cd6520c', 'experiments/check_fused_progressive.py': 'b7578afafcb4bc522bc0742b99e4ee5ab49307fb120b3475dc553900076dac55', 'code/bytes.py': '00625b902a29027075c13d64ca9019753e4aa153215aea6f349d25902a848115', 'scripts/p185_model_state.py': '437dbb5e9bd34e699ddc1f589834d207248074e214dfa821143781cd4fe3c4af', 'scripts/p185_single_state.py': 'ca4bce75c1e80c0a99eb18451323f2d0c2a07a8c707a3238f1a4e8d5d9114b3c', 'scripts/p185_segmented_kernel.py': 'bf09b996f098aea5027f25de54b50aef78f305225de105d9b8c9343751eb3f64', 'scripts/p176_kivi_codec.py': '1c278ebd1f648170696208ffed89a01e96960c18670610948d742262ad6b7cbb', 'scripts/p201_refinement.py': 'a1b8b84b9dbc655c21ead0bffa032be1ae4af97b5da4768112fe2a6106e58afb', 'scripts/p202_refinement.py': '292aca3d0f793c164b1d4204af21573cd4092597eb3f25055ded6d879afa04aa', 'kivi_source': '3f1140b0faff2cb524a3eb684167e403dffd488477b301826497462fd36fe540', 'a2/round6_depth_bank.py': '1d423bf03cdc76c0650776230d6a1093710e9c9fb8ac2cd84974d791aa65c6f7', 'a2/round6_depth_attention.py': 'f96277c726c39f2880e8618525b4bdcbac0049e68e7c4d3e2eb699fa24dfd945'}
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


BRPE_HELPER_SHA='15e075936d235cdd0a3935961b8083c2f10c9df59c545f1a6bd2c7aecd3dd9fe'
BRPE_CAL_SOURCE_SHA='8b986799062ef8d3fbc67624344d1740cfea4726b64aa3fc4d995ffaebc54438'
BRPE_PROFILE_SHA='7ec305a76cebf5ca9e358842e11e68fe6a77e6af9f2d83c39af723ff17eb5c22'
BRPE_DONORS=[[14, 0, 0], [21, 1, 0], [16, 1, 0], [21, 2, 0]]
BRPE_MODES={'BRPE_Lookup8': 'Lookup8', 'BRPE_Drop4': 'Drop4', 'BRPE_Exchange': 'Exchange'}
R9_STAGE_RUNS=['FP16_textvqa_new32_v1', 'FP16_infovqa_new32_v1', 'original_textvqa_new32_v1', 'original_infovqa_new32_v1', 'BRPE_Drop4_textvqa_new32_v1', 'BRPE_Drop4_infovqa_new32_v1', 'BRPE_Exchange_textvqa_new32_v1', 'BRPE_Exchange_infovqa_new32_v1', 'BRPE3_DropD_textvqa_new32_v1', 'BRPE3_DropD_infovqa_new32_v1', 'BRPE3_Exchange_textvqa_new32_v1', 'BRPE3_Exchange_infovqa_new32_v1']
BASE_BOOK='af29f26ddcbdf4ac6f098c3df497d2d741f0e7909c13ecc090194446846c9589'
CALIB_COMPLETE_SHA='c838b3a79cb970233a9579f8972ae4276157fad1001fd9b4e9118945bc0dc521'

def verify_brpe_profile(profile):
    require(profile["schema"] == "trajkv.round9.precision_exchange_profile.v1" and profile["base_profile"] == "r7_joint_k1_gt_cond" and
            profile["upgraded_layer"] == 14 and profile["dropped_k_units"] == BRPE_DONORS, "Unreviewed conditional exchange profile")
    require(profile["low_book_sha256"] == BASE_BOOK and profile["high_book_sha256"] == "9d3ad3683cf6888c30f3c8ba9c6dd0d4dbcf40106a34567d578479245e7309ae" and
            profile["base_metadata_factor_sha256"] == "8f47c56af5fe5e4da99bae05b83084173c5a55601c46c91130894075907e9cea", "Static books/original factors differ")
    require(profile["calibration_phase"] == "suffix_last" and profile["calibration_sources"] == 32 and profile["events"] == 896 and profile["references_read"] is False and
            profile["K112_and_V224_factors_unchanged"] is True and profile["retained_K_stages"] == 19 and profile["V_stages_unchanged"] == 20, "Conditional profile scope differs")
    source = profile["source_provenance"]
    require(source["calibrator"] == BRPE_CAL_SOURCE_SHA and source["exchange_helper"] == BRPE_HELPER_SHA and source["collection_complete"] == CALIB_COMPLETE_SHA,
            "Failed v1/new native-query epoch cannot become actual BRPE selector")
    norm = profile["normalization"]
    require(norm["implementation"] == "explicit FP64 amax/shift/exp/sum/log" and norm["row_sum_atol"] == 1e-12 and norm["original_KL_chain_and_additivity_thresholds_unchanged"] is True,
            "Selector normalization/thresholds differ")
    require(abs(profile["route_loss_after"] - profile["route_loss_predicted_from_conditional_marginals"]) <= 1e-8 * (1 + abs(profile["route_loss_after"])), "Conditional four-drop replay not verified")

def verify_brpe_row(row, spec, mode):
    require(mode in BRPE_MODES, "Unknown BRPE mode")
    actual = physical_complete_bytes(row, spec); n = row["actual_visual_token_count"]
    b0 = 3816 * n + 89733 + 18 * n * 43
    delta = -1344 if mode == "BRPE_Lookup8" else -72 * n - 1304 if mode == "BRPE_Drop4" else 744 - 8 * n
    bridge = row["r7_bridge"]; proof = bridge["round9_precision_exchange"]
    require(n >= 256 and n % 32 == 0 and actual == b0 + delta <= b0 and proof["before_unique_bytes"] == b0 and
            proof["actual_delta_bytes"] == proof["expected_delta_bytes"] == delta and proof["after_unique_bytes"] == actual, "BRPE complete physical ledger differs")
    require(proof["mode"] == BRPE_MODES[mode] and proof["source_sha256"] == BRPE_HELPER_SHA and proof["lookup_dtype"] == "torch.int8" and
            proof["lookup_bytes"] == 448 and proof["old_lookup_and_original_pools_released"] is True and proof["V20_base_parameters_and_residual_values_unchanged"] is True, "BRPE signed lookup/release/V20 proof differs")
    require(bridge["selected_stage_count"] == (43 if mode == "BRPE_Lookup8" else 39) and sum(x[0] == "v" for x in bridge["selected_stages"]) == 20, "K donors/V stages differ")
    if mode != "BRPE_Lookup8": require(proof["dropped_k_units"] == BRPE_DONORS and proof["format_bytes"] == 28 and proof["donor_descriptor_bytes"] == 12, "Real descriptors not paid")
    require(proof["extra_high_book_bytes"] == (2048 if mode == "BRPE_Exchange" else 0) and proof["old_low_book_bytes"] == 128, "Both static book constants not charged")
    tensors = row["owner_before_suffix"]["tensors"]
    for layer in range(28):
        item = tensors[f"layers.{layer}.visual.encoder_constants.lookup_view"]
        require(item["dtype"] == "torch.int8" and item["shape"] == [2, 4, 2] and item["storage_bytes"] == 448, "All28 views must use actual448B signed storage")
    budget = row["BRPE_budget_vs_original_R7_low"]
    require(budget["old_R7_low_B0_bytes"] == b0 and budget["actual_complete_bytes"] == actual and budget["exact_expected_delta_bytes"] == delta and
            budget["at_or_below_R7_low_B0"] is True and budget["strict1p3_budget_is_separate"] is True, "Byte envelope cannot masquerade as strict1.3")
    return actual

BRPE3_HELPER_SHA='7658d920970a9eba23b4edaed2c110e523ca66326aacafb7031ca188aa37bd99'
BRPE3_PROFILE_SHA='349514921518520383c375b9b5ead97ceaab356484f080cb546a91b1e188aa1a'
BRPE3_MODES={'BRPE3_DropD': 'DropD', 'BRPE3_Exchange': 'Exchange'}
BRPE3_DONORS=[[14, 0, 0], [15, 2, 0], [16, 1, 0], [17, 2, 0], [21, 1, 0], [21, 2, 0], [22, 1, 0], [22, 3, 0], [22, 0, 0], [25, 3, 0], [26, 2, 0]]
DP_PINS={'analyze.py': '9762f6d6487ae5c125b655708d8d14d3eb3185ae742b08062ecc0144f2e11b51', 'summary.json': '14986aa373ea23f70e57058afe11b77c826dae4b2a86be399af5f882a3af687d', 'export_U3D11_profile.py': '4a222e7035287f815977c2bd0d3a7c1d6648fa8c0638dfc7d042ac039657854e'}
COST_TABLE_SHA='28de9d0ecbb30a38eb81d9dbe36fac7213b66334e2af19df55b53936b98fd81a'
BASE_BOOK=R9_BASE_BOOK
def verify_brpe3_profile(profile):
    require(profile["schema"] == "trajkv.round9.precision_exchange_profile.v2" and profile["profile_role"] == "cal_only_DP_selected_U3D11" and
            profile["base_profile"] == "r7_joint_k1_gt_cond" and profile["upgraded_layers"] == [14,22,26] and
            profile["dropped_k_units"] == BRPE3_DONORS and profile["U"] == 3 and profile["D"] == 11, "Only exact cal-only U3D11 profile, not diagnostic fixture")
    require(profile["low_book_sha256"] == BASE_BOOK and profile["high_book_sha256"] == "9d3ad3683cf6888c30f3c8ba9c6dd0d4dbcf40106a34567d578479245e7309ae" and
            profile["base_metadata_factor_sha256"] == "8f47c56af5fe5e4da99bae05b83084173c5a55601c46c91130894075907e9cea", "Unchanged static books/factors required")
    require(profile["original_K_stages"] == 23 and profile["retained_K_stages"] == 12 and profile["V_stages_unchanged"] == 20 and profile["total_stages"] == 32 and
            profile["calibration_phase"] == "suffix_last" and profile["calibration_sources"] == 32 and profile["actual_source_table_events"] == 896, "U3D11 calibration/coverage differs")
    p=profile["source_provenance"];q=profile["selection_provenance"]
    require(p["calibrator"] == BRPE_CAL_SOURCE_SHA and p["exchange_helper"] == BRPE_HELPER_SHA and p["collection_complete"] == CALIB_COMPLETE_SHA,
            "DP must derive from the real old compressed-query calibration")
    require(q["DP_script_sha256"] == DP_PINS["analyze.py"] and q["DP_summary_sha256"] == DP_PINS["summary.json"] and
            q["profile_export_script_sha256"] == DP_PINS["export_U3D11_profile.py"] and q["candidate_cost_table_sha256"] == COST_TABLE_SHA and
            q["original_single_profile_sha256"] == BRPE_PROFILE_SHA and q["task_scores_or_answers_used"] is False and q["new_model_forwards"] == q["new_GPU_calls"] == 0,
            "No new-source answer-dependent fitting/selection may be substituted")
    require(profile["symbolic_complete_byte_delta"] == "765-6*N" and profile["Dmin_all_N_ge256"] == 11 and
            profile["whole_model_or_task_optimum_claimed"] is False, "Cal-only finite-table optimum is not a model/task optimum")

def verify_brpe3_row(row, spec, mode):
    require(mode in BRPE3_MODES, "Unknown U3D11 mode")
    actual=physical_complete_bytes(row,spec);n=row["actual_visual_token_count"];b0=3816*n+89733+18*n*43
    delta=-198*n-1283 if mode == "BRPE3_DropD" else 765-6*n
    bridge=row["r7_bridge"];p=bridge["round9_precision_exchange"]
    require(type(n) is int and n>=256 and n%32==0 and actual==b0+delta<=b0 and
            p["before_unique_bytes"]==b0 and p["after_unique_bytes"]==actual and p["actual_delta_bytes"]==p["expected_delta_bytes"]==delta,
            "U3D11 actual complete storage differs from exact B0 envelope")
    require(p["schema"]=="trajkv.round9.precision_exchange.v2" and p["mode"]==BRPE3_MODES[mode] and p["U"]==3 and p["D"]==11 and
            p["dropped_k_units"]==BRPE3_DONORS and p["source_sha256"]==BRPE3_HELPER_SHA and p["numerical_fields_v1_sha256"]==BRPE_HELPER_SHA,
            "Wrong U3D11 source or actual donor allocation")
    require(p["lookup_dtype"]=="torch.int8" and p["lookup_bytes"]==448 and p["format_bytes"]==28 and p["donor_descriptor_bytes"]==33 and
            p["old_lookup_and_original_pools_released"] is True and p["V20_base_parameters_and_residual_values_unchanged"] is True and
            p["extra_high_book_bytes"]==(2048 if mode=="BRPE3_Exchange" else 0) and p["old_low_book_bytes"]==128 and
            p["high_book_storage_shared_across_all_upgraded_layers"] is True, "Every constant/backing must be paid; V20 unchanged")
    require(p["upgraded_layers"]==([14,22,26] if mode=="BRPE3_Exchange" else []) and bridge["selected_stage_count"]==32 and
            sum(x[0]=="k" for x in bridge["selected_stages"])==12 and sum(x[0]=="v" for x in bridge["selected_stages"])==20, "Not actual K12/V20 stage32")
    for layer in range(28):
        x=row["owner_before_suffix"]["tensors"][f"layers.{layer}.visual.encoder_constants.lookup_view"]
        require(x["dtype"]=="torch.int8" and x["shape"]==[2,4,2] and x["storage_bytes"]==448, "Global int8 448B owner views differ")
    b=row["BRPE3_budget_vs_original_R7_low"]
    require(b["old_R7_low_B0_bytes"]==b0 and b["actual_complete_bytes"]==actual and b["exact_expected_delta_bytes"]==delta and
            b["at_or_below_R7_low_B0"] is True and b["shared_high_book_count"]==(1 if mode=="BRPE3_Exchange" else 0) and
            b["strict1p3_budget_is_separate"] is True, "Near-1.3 storage is not strict all-request1.3")
    return actual


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
        # Original Windows manifests may retain doubled separators. Normalize
        # separators only; origin, full relative source path and uniqueness stay exact.
        while '//' in value:value=value.replace('//','/')
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
        rm=load(root/folder/'manifest.json')
        with (root/path).open('rb') as stream,(root/side).open(encoding='utf-8') as identity:
            for offset,line in enumerate(stream,1):
                row=json.loads(line);proof=json.loads(identity.readline())
                if rm.get('runner_schema')==R9_SCHEMA:
                    physical_complete_bytes(row,rm['model_spec'])
                    if rm['config']['candidate']['mode']!='FP16':r7_verify_record_role_pins(rm['config'],row)
                    if rm['config']['candidate']['mode'] in BRPE_MODES:verify_brpe_row(row,rm['model_spec'],rm['config']['candidate']['mode'])
                    if rm['config']['candidate']['mode'] in BRPE3_MODES:verify_brpe3_row(row,rm['model_spec'],rm['config']['candidate']['mode'])
                rows.append({'id':row['id'],'status':row['status'],'source_line_sha256':proof['source_sha256'],
                    'public_line_sha256':proof['public_sha256'],'input_key_sha256':cache.input_key(row),
                    'execution_sha256':cache.execution_fingerprint(row)})
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
    round9_folders=set()
    def r9_context(m):
        runtime={}
        for key,aliases in {'torch_version':('torch_version',),'transformers_version':('transformers_version',),
            'cuda_version':('cuda_version','CUDA_version'),'python_version':('python_version',),'hardware':('hardware',),'runtime_executable':('runtime_executable',)}.items():
            values=[m[x] for x in aliases if x in m];require(values and all(x==values[0] for x in values),'Unrecorded/conflicting native runtime');runtime[key]=values[0]
        require(m['model_identity']['status']=='PASS_EXACT_HF_REQUIRED_TREE' and m['model_identity']['revision']==m['model_spec']['revision']==m['config']['model_revision'],'Original HF identity/revision differs')
        return {'runtime':runtime,'model_spec':m['model_spec'],'model_files':sorted(({k:r[k] for k in ('file','bytes','sha256')} for r in m['model_identity']['files']),key=lambda r:r['file']),
            **{k:m[k] for k in ('processor_id','processor_version','processor_file_hashes','generation_native','generation_overrides','decode_overrides')}}
    def verify_round9(folder,m,c):
        require(m.get('runner_schema')==R9_SCHEMA,'Unknown custom producer; no SourceRun fallback')
        cfg=m['config'];candidate=cfg['candidate'];count=cfg['count']
        require(cfg['schema']==R9_SCHEMA and cfg['cohort_kind']=='development' and type(count) is int and count==cfg['input_count']==32 and cfg['start']==0,
                'Only exact new-source32 components, not old704/formal500')
        mode=candidate['mode'];ds='textvqa' if '_textvqa_' in Path(folder).name else 'infovqa'
        require(Path(folder).name in R9_STAGE_RUNS and Path(folder).name==f'{mode}_{ds}_new32_v1' and
                mode in ('FP16','original','BRPE_Drop4','BRPE_Exchange','BRPE3_DropD','BRPE3_Exchange'), 'Named replication source/algorithm label differs')
        require(cfg['preprocessing']=={'max_patches':8} and cfg['max_new_tokens']==64 and cfg['variant_source']['name']=='r7_joint_k1_gt_cond' and
                not any(candidate.get(k) for k in ('book_path','book_sha256','theta_path','theta_sha256','diagnostic_only')),
                'Frozen low/cap8 static candidates, no newsource-trained parameters')
        require(count==m['expected_QA']==c['expected_QA']==c['recorded_QA'] and c['counts']=={'PASS':count} and
                m['prediction_cache_used'] is False,'Partial/failed/reused run is not original fresh R9')
        require(R9_POLICY_FILE in entries,'Missing exact R9 source policy')
        policy=load(root/R9_POLICY_FILE)
        require(policy['schema']=='trajkv.public_round9_source_policy.v5' and policy['SourceRun_admission'] is False and
                policy['source_epochs']==R9_EPOCHS and policy['drivers']==R9_DRIVERS,'R9 producer/dependency whitelist differs')
        proof=load(root/folder/'source_retrieval_receipt_round9_v2.json')
        require(proof['status']=='PASS_INDEPENDENT_CUSTOM_DRIVER_SOURCE_INSPECTION' and proof['SourceRun_admission_established'] is False,'Independent R9 V2 source proof required')
        pf={x['file']:x for x in proof['files']};require(len(pf)==len(proof['files']),'Duplicate source proof files')
        driver=proof['actual_driver'];require(driver in R9_DRIVERS,'Unreviewed source epoch')
        needed={'manifest.json','records.jsonl','complete.json','source_inventory.json','launch_receipt.json','source_config.json','source_job.json','source_driver.py'}|{'source_files/'+x for x in R9_EPOCHS[driver]}
        require(needed<=set(pf),'Missing actual source dependencies')
        for name,value in pf.items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe source path')
            original_graph(folder,name,value['sha256']);require(entry(folder,name)['source_bytes']==value['bytes'],'Independent source bytes differ')
        job=load(root/folder/'source_job.json');launch=load(root/folder/'launch_receipt.json');scfg=load(root/folder/'source_config.json');iv=load(root/folder/'source_inventory.json')
        require(scfg==cfg and m['config_sha256']==original_hash(folder,'source_config.json') and
                launch['job_sha256']==original_hash(folder,'source_job.json') and launch['source_inventory_sha256']==original_hash(folder,'source_inventory.json'),'Actual config/job/inventory binding differs')
        base=job['argv'][job['argv'].index('--output')+1].split('/results/',1)[0];snap=base+'/jobs/'+job['name']+'.source'
        argv=[snap+x[len(base):] if x.startswith(base+'/code/') and x.endswith('.py') else x for x in job['argv']]
        require(proof['producer_name']==job['name']==launch['name'] and type(launch['pid']) is int and launch['pid']>0 and
                launch['argv_executed']==argv and launch['source_snapshot']==m['runtime_source_root']==snap and
                launch['runtime_python']==m['runtime_executable']==job['argv'][0] and str(launch['CUDA_VISIBLE_DEVICES'])==str(job['env']['CUDA_VISIBLE_DEVICES']),'Wrong actual producer/runtime/argv/GPU')
        require(driver==Path(argv[2]).relative_to(snap).as_posix() and m['source_hashes']==R9_EPOCHS[driver],'Actual entry/dependency version differs')
        if driver=='code/round9_runner_v4.py':
            require(mode in ('FP16','original','BRPE_Drop4','BRPE_Exchange'),'v4 source only original/newsource single-layer modes')
        else:require(driver=='code/round9_runner_v5.py' and mode in BRPE3_MODES,'Only exact U3D11 v5 source/modes')
        if mode in BRPE_MODES or mode in BRPE3_MODES:
            expected_profile=BRPE3_PROFILE_SHA if mode in BRPE3_MODES else BRPE_PROFILE_SHA
            require(candidate.get('profile_sha256')==expected_profile,'Wrong cal-only frozen profile')
            original_graph(folder,'assets/candidate_profile.json',expected_profile);profile=load(root/folder/'assets/candidate_profile.json')
            (verify_brpe3_profile if mode in BRPE3_MODES else verify_brpe_profile)(profile)
            original_graph(folder,'assets/candidate_high_book.json',profile['high_book_sha256'])
            cal=policy['BRPE_calibration_root'];fp=load(root/cal/'FETCH_RECEIPT.json')
            require(fp['status']=='PASS_INDEPENDENT_REAL_EXECUTED_SOURCE_AND_METADATA' and fp['actual_executed_driver_sha256']==BRPE_CAL_SOURCE_SHA and
                    fp['independent_probe_then_SFTP_then_second_remote_rehash'] is True and fp['private_PT_WO_weights_or_images_transferred'] is False,'Actual selector independent proof absent')
            for x in fp['files']:original_graph(cal,x['file'],x['sha256'])
            original_graph(cal,'profile.json',BRPE_PROFILE_SHA);cc=load(root/cal/'complete.json');cs=load(root/cal/'summary.json')
            require(cc['status']==cs['status']=='PASS_CAL_ONLY_CONDITIONAL_EXCHANGE_PROFILE' and cc['actual_GPU'] is cs['actual_GPU'] is True and cc['calibration_sources']==32 and
                    cc['actual_events']==896 and cc['new_model_forwards']==0 and cc['profile_sha256']==BRPE_PROFILE_SHA,'Failed/unrun selector cannot become passed profile')
            original_graph(cal,'summary.json',cc['summary_sha256']);cj=load(root/cal/'actual_job.json');cl=load(root/cal/'launch_receipt.json');ci=load(root/cal/'source_inventory.json')
            require(original_hash(cal,'actual_job.json')==cl['job_sha256'] and original_hash(cal,'source_inventory.json')==cl['source_inventory_sha256'] and cj['name']==cl['name']==fp['producer'] and
                    type(cl['pid']) is int and cl['pid']>0 and ci['code/round9_exchange_calibrate_v2.py']==BRPE_CAL_SOURCE_SHA and ci['code/round9_precision_exchange_v1.py']==BRPE_HELPER_SHA,'Selector actual producer/source binding differs')
            cargv=[cl['source_snapshot']+'/code/round9_exchange_calibrate_v2.py' if x.endswith('/code/round9_exchange_calibrate_v2.py') else x for x in cj['argv']]
            require(cargv==cl['argv_executed'],'Selector actual argv differs')
            original_graph(cal,'executed_source/round9_exchange_calibrate_v2.py',BRPE_CAL_SOURCE_SHA)
            # Actual compressed-query calibration32 metadata is independently bound,
            # even though this stage does not include or fit a global64 codebook.
            calmeta=policy['calibration_metadata_root'];cp=load(root/calmeta/'INDEPENDENT_SMALL_METADATA_RETRIEVAL.json')
            require(cp['status']=='PASS_ACTUAL_CAL32_EXECUTED_SOURCE_AND_SMALL_METADATA' and cp['native_query_captured'] is False and
                    cp['private_PT_WO_images_downloaded'] is False and cp['private_tensor_hashes_are_producer_recorded_not_rehashed_by_this_small_fetch'] is True,'Actual old-cal availability/query differs')
            for x in cp['files']:original_graph(calmeta,x['file'],x['sha256'])
            cj=load(root/calmeta/'source_job.json');cl=load(root/calmeta/'launch_receipt.json');ci=load(root/calmeta/'source_inventory.json');ccfg=load(root/calmeta/'source_config.json');cmf=load(root/calmeta/'manifest.json')
            require(ccfg==cmf['config'] and ccfg['inputs_sha256']==R9_CAL_INPUT_SHA and original_hash(calmeta,'source_job.json')==cl['job_sha256'] and
                    original_hash(calmeta,'source_inventory.json')==cl['source_inventory_sha256'] and cj['name']==cl['name']==cp['producer_name'] and
                    type(cl['pid']) is int and cl['pid']>0,'Actual calibration job/config/inventory binding differs')
            cargv=[cl['source_snapshot']+'/code/round9_collect_v1.py' if x.endswith('/code/round9_collect_v1.py') else x for x in cj['argv']]
            require(cl['argv_executed']==cargv and ci['code/round9_collect_v1.py']==cp['actual_driver_sha256']=='2d709806122c8127a766747ef5d29821fb22f389238f8448f0dfd25c363a61df','Actual old-cal executed entry differs')
            original_graph(calmeta,'complete.json',R9_CAL_COMPLETE_SHA);original_graph(calmeta,'inputs/calibration32_inputs.jsonl',R9_CAL_INPUT_SHA)
            if mode in BRPE3_MODES:
                dp=policy['DP_analysis_root'];dp_profile=policy['DP_profile_file']
                require(entries[dp_profile]['source_sha256']==BRPE3_PROFILE_SHA and load(root/dp_profile)==profile,'DP artifact not linked to deployed profile')
                require(policy['DP_source_pins']==DP_PINS,'Unreviewed DP scripts or summary')
                for name,digest in DP_PINS.items():original_graph(dp,name,digest)
                original_graph(cal,'candidate_profiles.json',COST_TABLE_SHA)
                dps=load(root/dp/'summary.json')
                require(dps['status']=='PASS_FINITE_CONDITIONAL_TABLE_DP_CAL_ONLY_NOT_MODEL_OR_OWNER' and dps['GPU_calls']==dps['new_model_forwards']==0,
                        'CPU finite-table DP is not a new model/GPU or task optimum')
        require(cfg['model_id']=='M1' and m['model_spec']['model_id']=='M1' and m['decode_overrides']=={'skip_special_tokens':True,'clean_up_tokenization_spaces':False},'M1 native decode contract differs')
        r9_context(m)
        for name,digest in R9_EPOCHS[driver].items():
            require(iv[name]==digest,'Snapshot dependency mismatch');original_graph(folder,'source_files/'+name,digest)
        original_graph(folder,'source_driver.py',R9_DRIVERS[driver])
        # Preserve V1 defect and its independent label-only correction if present.
        old_file=folder+'/source_retrieval_receipt_round9_v1.json'
        if old_file in entries:
            old=load(root/old_file)
            require({k:v for k,v in old.items() if k!='producer_name'}=={k:v for k,v in proof.items() if k!='producer_name'},'More than old producer label changed')
            if old['producer_name']!=proof['producer_name']:
                correction=load(root/folder/'PRODUCER_LABEL_CORRECTION_v1.json')
                require(correction['status']=='PASS_PRODUCER_LABEL_CORRECTED_BY_INDEPENDENT_READBACK' and
                        correction['old_receipt_sha256']==entries[old_file]['source_sha256'] and correction['new_receipt_sha256']==original_hash(folder,'source_retrieval_receipt_round9_v2.json') and
                        correction['actual_producer']==job['name'] and correction['model_predictions_and_scores_changed'] is False and
                        old['producer_name']==list(c['executed_legacy_source_sha256'])[-1],'Unproven original label-only correction')
        if candidate['mode']!='FP16':
            r7_verify_executed_roles(cfg,c['executed_legacy_source_sha256'])
            actual={x['remote_file']:x['sha256'] for x in proof['files'] if x['file'].startswith('legacy_sources/') and not x['file'].endswith('dynamic_kivi.py')}
            require(actual==c['executed_legacy_source_sha256'],'Actual legacy graph not fully archived')
        for name in ('direction_book_path','metadata_factor_path','round7_policy_path'):
            original_graph(folder,'assets/base_'+name+'.json',cfg['expected_asset_sha256'][name])
        original_graph(folder,'assets/original_profile.json',cfg['variant_source']['sha256'])
        if cfg['cohort_kind']=='development':
            require(entries[policy['input_manifest_file']]['source_sha256']==R9_MANIFEST_SHA,'Frozen new-source reserve changed')
            frozen=load(root/policy['input_manifest_file']);own=frozen['input_and_ref_files']
            require(Path(cfg['inputs']).name in own and own[Path(cfg['inputs']).name]['sha256']==cfg['inputs_sha256'],'R9 cohort not frozen new-source input')
        if candidate.get('book_sha256'):
            digest=candidate['book_sha256'];original_graph(folder,'assets/candidate_book.json',digest);book=load(root/folder/'assets/candidate_book.json')
            require(book['original_book_sha256']==R9_BASE_BOOK and book['dimension']==4 and book['centroids']==16 and len(book['centers'])==16 and all(len(x)==4 for x in book['centers']),'Codebook capacity/original basis differs')
            require(digest in policy['fits'],'No exact fitted artifact source closure')
            fit=policy['fits'][digest]['folder'];fm=load(root/fit/'manifest.json');fc=load(root/fit/'complete.json');fp=load(root/fit/'EXECUTED_FIT_SOURCE_RETRIEVAL.json')
            require(fp['status']=='PASS_REAL_GPU_FIT_ARTIFACT_AND_EXECUTED_SOURCE' and fp['actual_GPU'] is True and fp['artifact_sha256']==digest==fc['artifact_sha256'],'No actual fitted artifact proof')
            objective=book['round9_fit']['objective'];kind={'mse':'K-MSE','functional':'K-Func'}[objective]
            require(fm['config']['kind']==fc['kind']==kind,'Fit kind does not match book objective')
            if candidate['mode'] in ('K-MSE','K-Func'):require(candidate['mode']==kind,'Candidate label does not match fitted book objective')
            for x in fp['files']:original_graph(fit,x['file'],x['sha256'])
            fj=load(root/fit/'source_job.json');fl=load(root/fit/'launch_receipt.json');fi=load(root/fit/'source_inventory.json')
            require(fl['job_sha256']==original_hash(fit,'source_job.json') and fl['source_inventory_sha256']==original_hash(fit,'source_inventory.json') and
                    fl['name']==fj['name'] and type(fl['pid']) is int and fl['pid']>0,'Actual fit launch/inventory differs')
            executed=[fl['source_snapshot']+'/code/round9_train_v1.py' if x.endswith('/code/round9_train_v1.py') else x for x in fj['argv']]
            require(fl['argv_executed']==executed,'Actual fit argv differs')
            fitcfg=source_file_reference(fj['argv'][fj['argv'].index('--config')+1]);require(entries[fitcfg]['source_sha256']==fm['config_sha256'] and load(root/fitcfg)==fm['config'],'Actual fit config differs')
            expected_fit={'round9_train_v1.py':'4809904d5a88643090535b925b7de9f090f2409c53d219671c07614dda466779',
                'round9_calibration_v1.py':'757cc51740ad2f92d8ebd9ad9adadf2719131ae634b16b24052363f8e45ba84f',
                'round9_keyfit_v1.py':'96adf8b9a94e51da00ab99eaeebb83989b005f15bb3762b18d6c01d4ef5c6ec9'}
            for name,value in expected_fit.items():require(fi['code/'+name]==value==original_hash(fit,'source_files/'+name),'Executed fit source epoch differs')
            require(fm['collection_complete_sha256']==R9_CAL_COMPLETE_SHA and fm['query_source']=='compressed_query_same_student_text' and
                    fc['student_state_refreshes']==0 and fc['frozen_student_snapshot_only'] is True,'Unrun native-q/refresh substituted into S0 fit')
            cal=policy['calibration_metadata_root'];cp=load(root/cal/'INDEPENDENT_SMALL_METADATA_RETRIEVAL.json')
            require(cp['status']=='PASS_ACTUAL_CAL32_EXECUTED_SOURCE_AND_SMALL_METADATA' and cp['native_query_captured'] is False and
                    cp['private_PT_WO_images_downloaded'] is False and cp['private_tensor_hashes_are_producer_recorded_not_rehashed_by_this_small_fetch'] is True,'Calibration availability/query claim differs')
            for x in cp['files']:original_graph(cal,x['file'],x['sha256'])
            cj=load(root/cal/'source_job.json');cl=load(root/cal/'launch_receipt.json');ci=load(root/cal/'source_inventory.json');ccfg=load(root/cal/'source_config.json');cmf=load(root/cal/'manifest.json')
            require(ccfg==cmf['config'] and ccfg['inputs_sha256']==R9_CAL_INPUT_SHA and
                    original_hash(cal,'source_job.json')==cl['job_sha256'] and original_hash(cal,'source_inventory.json')==cl['source_inventory_sha256'] and
                    cj['name']==cl['name']==cp['producer_name'] and type(cl['pid']) is int and cl['pid']>0,'Actual calibration job/config/inventory binding differs')
            cargv=[cl['source_snapshot']+'/code/round9_collect_v1.py' if x.endswith('/code/round9_collect_v1.py') else x for x in cj['argv']]
            require(cl['argv_executed']==cargv and ci['code/round9_collect_v1.py']==cp['actual_driver_sha256']=='2d709806122c8127a766747ef5d29821fb22f389238f8448f0dfd25c363a61df','Actual calibration executed entry/argv differs')
            original_graph(cal,'complete.json',R9_CAL_COMPLETE_SHA);original_graph(cal,'inputs/calibration32_inputs.jsonl',R9_CAL_INPUT_SHA)
            original_graph(cal,'source_files/code/round9_collect_v1.py','2d709806122c8127a766747ef5d29821fb22f389238f8448f0dfd25c363a61df')
            prov=book['round9_fit']['train_provenance'];cm=load(root/cal/'complete.json')
            require(prov['collection_provenance']['complete_sha256']==R9_CAL_COMPLETE_SHA and fm['private_WO_sha256']==cm['wo_sha256'] and
                    sorted(prov['private_train_shards'])==sorted([[x['file'],x['sha256']] for x in cm['sources']]) and
                    prov['train_labels_or_development_answers_read'] is False and prov['fixture_only'] is False,'Fit tensor/source linkage differs')
            calinputs=load(root/policy['input_manifest_file'])['input_files']['calibration32']['path'];calinputfile=source_file_reference(calinputs)
            ids=[json.loads(x)['id'] for x in (root/calinputfile).read_text(encoding='utf-8').splitlines()]
            require(sorted(prov['calibration_ids'])==sorted(ids) and [x['id'] for x in cm['sources']]==ids,'Actual calibration source32 IDs differ')
            for item in prov['callback_sources'].values():
                name=Path(item['file']).name;value=item['sha256']
                if '.source/code/' in item['file']:require(fi['code/'+name]==value,'Callback not pinned by actual fit snapshot')
                require(any(e['source_sha256']==value and e['public_sha256']==value and Path(path).name==name for path,e in entries.items()),'Exact callback code bytes not archived')
        round9_folders.add(folder)
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
            check_proof(folder,'retrieval_receipt.json',('manifest.json','records.jsonl','complete.json'),'PASS_REMOTE_RAW_SHA256_EXACT')
            verify_round9(folder,m,c)
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
    require(round9_folders and index.get('custom_driver_provenance_schema')=='trajkv.public_round9.v5','This verifier requires explicit R9 source phase')
    require(len(artifacts)==12 and sum(x['expected_QA'] for x in artifacts)==384 and len(round9_folders)==12,'Newsource V5 requires exact12/384; old704 is not fresh here')
    require({Path(a['raw_dir']).name for a in artifacts}==set(R9_STAGE_RUNS),'Wrong newsource12 components')
    policy=load(root/R9_POLICY_FILE);frozen=load(root/policy['input_manifest_file'])
    require(entries[policy['input_manifest_file']]['source_sha256']==R9_MANIFEST_SHA and frozen['schema']=='trajkv.round9.unobserved_confirmation_reserve.v1' and
            frozen['selection_reads_model_outputs_or_correctness'] is False and frozen['source_cluster_rawSHA_intersection_with_current160_formal2000_historical_compression_or_confirm']==0,
            'Reserve selection/source-isolation evidence changed')
    cohorts={}
    for ds in ('textvqa','infovqa'):
        item=frozen['input_and_ref_files'][ds+'_confirm32_inputs.jsonl'];path=source_file_reference(item['path'])
        require(entries[path]['source_sha256']==item['sha256'],'Frozen new-source input changed')
        cohorts[ds]=[json.loads(x) for x in (root/path).read_text(encoding='utf-8').splitlines()]
        require(len(cohorts[ds])==32 and all(r['split']==('train' if ds=='infovqa' else 'validation') for r in cohorts[ds]),'Actual split/count differs')
    selected=cohorts['textvqa']+cohorts['infovqa'];forbidden=[]
    old=load(root/policy['calibration_input_manifest_file'])
    require(entries[policy['calibration_input_manifest_file']]['source_sha256']==policy['calibration_input_manifest_sha256']=='583860001cbb4e7fdd1dc974263c3bf630d8daef917d5c1bb089f5a65a57a650','Original cal/dev160 manifest changed')
    for item in old['input_files'].values():
        path=source_file_reference(item['path']);require(entries[path]['source_sha256']==item['sha256'],'Original calibration/development input changed')
        forbidden += [json.loads(x) for x in (root/path).read_text(encoding='utf-8').splitlines()]
    for ds in ('textvqa','infovqa','docvqa','chartqa'):
        forbidden += [json.loads(x) for x in (root/('configs/datasets/v3/'+ds+'_p2_inputs.jsonl')).read_text(encoding='utf-8').splitlines()]
    for key in ('id','source_cluster_id','original_image_sha256'):
        require(len({r[key] for r in selected})==64 and not {r[key] for r in selected}&{r[key] for r in forbidden},'Newsource/cal/dev/formal overlap')
    for key in ('exclusion_audit','exclusion_identity_file'):
        item=frozen[key];path=source_file_reference(item['path']);require(entries[path]['source_sha256']==item['sha256'],'Frozen historical exclusion identity proof changed')
    contexts={}
    for folder in round9_folders:
        m=raw_meta[folder][0];key=m['config']['inputs_sha256'];ctx=r9_context(m)
        if key in contexts:require(contexts[key]==ctx,'Methods differ in actual model/processor/generation/runtime')
        contexts[key]=ctx
    # Logical points retain real source components; no synthesized raw64 run.
    for group in groups:
        matrix=load(root/group['declared_matrix_file']);logical=defaultdict(list)
        for row in matrix['rows']:
            cfgref=source_file_reference(row['config']);a=next(x for x in group['phase_artifacts'] if x['config']==cfgref);logical[row['logical_point']].append(a['raw_dir'])
        for point,folders in logical.items():
            source=[raw_meta[x][0] for x in folders];source.sort(key=lambda x:x['config']['start']);ids=[];cursor=source[0]['config']['start']
            for m in source:
                cfg=m['config'];require(cfg['count'] in (32,64,500),'Small diagnostics cannot enter main quality groups');require(cfg['start']==cursor,'Logical union overlaps/gaps');cursor+=cfg['count'];ids+=m['expected_ids']
                for key in ('candidate','method','inputs_sha256','cohort_kind','preprocessing','max_new_tokens','question_suffix'):require(cfg[key]==source[0]['config'][key],'Logical union protocol/algorithm differs')
                require(m['source_hashes']==source[0]['source_hashes'],'Logical union different numeric source epochs')
                require(r9_context(m)==r9_context(source[0]),'Logical union actual model/processor/generation/runtime differs')
            diagnostic=source[0]['config']['candidate'].get('diagnostic_only',False)
            if diagnostic:require(source[0]['config']['candidate'].get('method_frontier_eligible') is False,'Diagnostic promoted to method')
            declared_point=group['phase']['logical_points'][point]
            require(ids==declared_point['ids'] and len(ids)==len(set(ids))==declared_point['QA'] and len(source)==declared_point['source_runs'],'Logical union sources/count differs')
            require(declared_point['single_fresh64_claim']==(len(source)==1 and len(ids)==64),'Union mislabeled as fresh64')
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
        own_source_lock=load(root/'configs/round9_confirm_reserve_v1/source_lock.json') if inputs.startswith('configs/round9_confirm_reserve_v1/') else source_lock
        for path in (inputs,refs):
            lock=own_source_lock['files'][Path(path).name];require(entries[path]['source_sha256']==lock['sha256'] and entries[path]['source_bytes']==lock['bytes'],'Original input/reference lock differs')
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
    eligible=all_complete and not index.get('fixture_only',False)
    require(index['publication_eligible']==eligible,'Incomplete preview/fixture publication eligibility differs')
    phase=index['phase']
    return {'status':'PASS_PUBLIC_ROUND9_EXPORT_VERIFIED_V5','public_files':len(entries),'public_payload_bytes':total,
        'independent_phases_verified':verified_groups,'phase_group_totals_are_separate':True,
        'supporting_evidence_groups_verified':len(index.get('supporting_evidence_groups',[])),
        'jsonl_line_identities':line_count,'raw_complete_runs_verified':len(visited),
        'original_donors':sum(n['kind']=='original_inference_run' for n in index['source_closure']),
        'derived_views':sum(n['kind']=='derived_prediction_view' for n in index['source_closure']),
        'source_closure_verified':True,'custom_R9_originals':len(round9_folders),'round9_actual_runs':len(round9_folders),'actual_fresh_QA':sum(raw_meta[f][0]['expected_QA'] for f in round9_folders),'custom_SourceRun_admission_claimed':False,'score_recomputations':scoring,'archive_verified':bool(archive),
        'phase':phase,'publication_eligible':eligible,'fixture_only':bool(index.get('fixture_only',False)),
        'completion_scope':'declared_matrix_only',
        'whole_external_library_completion_assessed':False,'whole_Round9_complete':False,'old704_recounted_as_fresh':0,'independent_source_questions':64,'github_upload_verified':False}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--archive',type=Path);parser.add_argument('--recompute-scores',action='store_true');parser.add_argument('--receipt',type=Path)
    args=parser.parse_args();result=verify(args.root,args.archive,args.recompute_scores)
    if args.receipt:
        with args.receipt.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
