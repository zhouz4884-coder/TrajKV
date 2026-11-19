"""CPU-only stub/actual-codec checks for the isolated complete2x2 runner."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from round8_codec import encode_key as legacy_encode, decode_key, key_template
from round6_depth_codec import encode_sign_depth, decode_sign_stage, plan_depth_stages
from round8_joint_gain import joint_refit_key
from round8_spherical_codec import encode_key as spherical_encode
from round8_spherical_joint_runner import (validate_variant, validate_config, scoped_dispatch,
    spherical_joint_class, source_fingerprints, finalize_run, sha)


FROZEN_V1 = {
    'round8_spherical_codec.py': 'bbd478fcf075a303e3b669eaed1905f2c716ff192cd71142c3fd76389f22f4dd',
    'round8_spherical_runner.py': 'f7a67aa1bcf1321a8e3a00940cae748791bae2e49e1abd0644f19ff5456e3dfe',
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def rejected(function):
    try:
        function()
    except (ValueError, RuntimeError):
        return True
    return False


def controls(output):
    torch.set_num_threads(1)
    torch.manual_seed(20261118)
    sources = source_fingerprints()
    require(all(sources[key] == value for key, value in FROZEN_V1.items()), 'Immutable v1 source changed')
    root = Path(__file__).resolve().parents[2]
    frozen = json.loads((root / 'configs/a2/round7_v1/confirm_doc128_s0.json').read_text(encoding='utf-8-sig'))
    reference = next(v for v in frozen['variants'] if v['name'] == 'r7_joint_k1_gt_cond')
    priority = reference['selected_stages']
    book = torch.full((16, 4), 20., dtype=torch.float16)
    book[0] = 1.
    book[1] = torch.tensor([1.0146484375, 1.013671875]).repeat(2).half()
    bookpath = output / 'synthetic_book.json'
    bookpath.write_text(json.dumps(dict(centers=book.float().tolist())), encoding='utf-8')
    base_cfg = dict(method='round8_joint', direction_book_path=str(bookpath),
        metadata_reference_book_path=str(bookpath), metadata_factor_path='synthetic-fixed-factor',
        selected_stages=priority, res_stages_limit=43, key_encoding='reencode',
        key_amplitude='norm', value_encoding='binary', v_group_size='all',
        round8_factor_policy='inherit_from_reference_book', budget_bits=1.8, cap=8)
    variants = [dict(base_cfg, name=f'cd{int(cd)}_joint{int(joint)}',
        key_assignment='row_cosine_cd' if cd else 'l2', joint_gain_refit=joint)
        for cd in (False, True) for joint in (False, True)]
    config = dict(variants=variants)
    declarations = validate_config(config)
    require(rejected(lambda: validate_config(dict(variants=variants[:3]))), 'Missing factorial cell accepted')
    bad = dict(variants[0], selected_stages=priority[:43])
    require(rejected(lambda: validate_variant(bad)), 'Truncated ranking accepted')
    bad = dict(variants[-1], joint_gain_refit='true')
    require(rejected(lambda: validate_variant(bad)), 'Nonboolean joint declaration accepted')
    changed = list(map(list, priority)); changed[42], changed[43] = changed[43], changed[42]
    require(rejected(lambda: validate_config(dict(variants=variants[:3] + [dict(variants[-1], selected_stages=changed)]))), 'Cross-cell mask mismatch accepted')
    require(rejected(lambda: validate_variant(dict(variants[-1], value_encoding='reencode'))), 'V side intervention accepted')
    require(rejected(lambda: validate_config(dict(variants=variants[:3] + [dict(variants[-1], row_cosine_sweeps=2)]))), 'Different CD settings accepted across joint cells')
    nv, lh = 1792, 112
    base_bytes = lh * nv * 34 + lh * 128 * 6 + 128 + nv * 8
    plans = [plan_depth_stages(layers=28, heads=4, visual_tokens=nv,
        visual_full_bytes=lh * nv * 512, base_bytes=base_bytes, factor_bytes=1344,
        budget_bits=v['budget_bits'], selected_stages=v['selected_stages'], stage_limit=43) for v in variants]
    require(all(p['predicted_bytes'] == 8315013 and len(p['ranking_constant']) == 448 and len(p['selected_stages']) == 43 for p in plans), 'Four cells differ in complete owner')
    module = SimpleNamespace(encode_key=legacy_encode)
    class Base:
        settings = {}
        def _initialize(self, source, dictionary, *, fail=False):
            cfg = type(self).settings
            self.events = ['encode']
            self.code, self.gain = module.encode_key(source, dictionary,
                assignment=cfg['key_assignment'], gain_factor=torch.tensor([[1.073]]))
            if fail:
                raise ValueError('intentional constructor failure')
            self.events.append('regenerate_sign_alpha')
            self.pre_joint_gain = self.gain.clone()
            self.base = decode_key(self.code, self.gain, dictionary)
            codes, alphas, decoded, _ = encode_sign_depth(source, self.base, 1)
            self.sign, self.alpha = codes[0], alphas[0]
            self.pre_joint_sign, self.pre_joint_alpha = self.sign.clone(), self.alpha.clone()
            self.before_joint = decoded.clone()
            self.joint_gain_enabled = cfg['joint_gain_refit']
            self.joint_gain_statistics = []
            if self.joint_gain_enabled:
                self.events.append('joint_existing_gain_alpha')
                self.gain, self.alpha, stats = joint_refit_key(source, key_template(self.code, dictionary),
                    self.gain, self.sign, self.alpha)
                self.joint_gain_statistics.append(stats)
            self.final = decode_sign_stage(decode_key(self.code, self.gain, dictionary), self.sign, self.alpha)
            self.requested_residual_stages = [tuple(x) for x in cfg['selected_stages']]
            self.selected_residual_stages = self.requested_residual_stages[:43]
            self.ranking_constant_bytes = 448
        def result_metadata(self):
            return dict(owned=8315013, key_gain_formula='original joint physical formula' if self.joint_gain_enabled else 'original norm formula',
                key_gain_rule='joint-effective-gain-and-alpha' if self.joint_gain_enabled else 'norm',
                joint_gain_refit=self.joint_gain_enabled, joint_gain_statistics=self.joint_gain_statistics)
    cls = spherical_joint_class(Base, module)
    x = torch.tensor([2., 1.]).repeat(64)[None]
    cells, observations = {}, []
    for variant in variants:
        cls.settings = variant
        instance = cls(); instance._initialize(x, book)
        cd, joint = variant['key_assignment'] == 'row_cosine_cd', variant['joint_gain_refit']
        expected_code, expected_gain = (spherical_encode(x, book, gain_factor=torch.tensor([[1.073]]))
            if cd else legacy_encode(x, book, gain_factor=torch.tensor([[1.073]])))
        require(torch.equal(instance.code, expected_code) and torch.equal(instance.pre_joint_gain, expected_gain), 'Encode fields differ from declared cell')
        require(instance.events == ['encode', 'regenerate_sign_alpha'] + (['joint_existing_gain_alpha'] if joint else []), 'Wrong execution order')
        require(torch.equal(instance.sign, instance.pre_joint_sign), 'Joint changed existing sign bytes')
        meta = instance.result_metadata()
        require(meta['row_cosine_joint_factorial'] == dict(cd=cd, joint=joint) and meta['owned'] == 8315013 and meta['row_cosine_joint_added_fields'] == 0, 'Factorial metadata/owner incorrect')
        require(meta['key_gain_formula'] == ('original joint physical formula' if joint else 'original norm formula') and meta['key_gain_rule'] == ('joint-effective-gain-and-alpha' if joint else 'norm'), 'Final physical gain metadata was overwritten')
        require('pre-joint' in meta['row_cosine_statistics_stage'], 'CD statistics stage is ambiguous after joint refit')
        require(bool(meta['row_cosine_statistics']) == cd and module.encode_key is legacy_encode, 'Statistics/dispatcher restoration failed')
        before = (x.double() - instance.before_joint.double()).square().sum(-1)
        after = (x.double() - instance.final.double()).square().sum(-1)
        require(bool((after <= before).all()), 'Existing fixed-field joint guard worsened its own input')
        cells[(cd, joint)] = instance
        observations.append(dict(cd=cd, joint=joint, events=instance.events,
            base_SSE_pre_res_pre_joint=float((x.double()-instance.base.double()).square().sum()),
            selected_K_final_SSE=float(after.sum()), final_gain_rule=meta['key_gain_rule'], same_complete_owner_bytes=meta['owned']))
    for cd in (False, True):
        off, on = cells[(cd, False)], cells[(cd, True)]
        require(torch.equal(off.code, on.code) and torch.equal(off.sign, on.sign), 'Joint column does not isolate fixed J/sign')
        require(float((x.double()-on.final.double()).square().sum()) <= float((x.double()-off.final.double()).square().sum()), 'Pairwise fixed-field inequality failed')
    cls.settings = variants[-1]
    require(rejected(lambda: cls()._initialize(x, book, fail=True)) and module.encode_key is legacy_encode, 'CD constructor failure did not restore dispatcher')
    try:
        with scoped_dispatch(module, lambda *a, **k: None):
            raise ValueError('intentional')
    except ValueError:
        pass
    require(module.encode_key is legacy_encode, 'Scoped error restoration failed')
    fixture = output / 'postcheck_pass'; fixture.mkdir()
    expected_config = 'synthetic-config-sha256'
    (fixture / 'manifest.json').write_text(json.dumps(dict(config_sha256=expected_config, external_codec_inputs={})), encoding='utf-8')
    (fixture / 'complete.json').write_text(json.dumps(dict(status='COMPLETE', records=4, expected=4)), encoding='utf-8')
    marker = finalize_run(fixture, {str(bookpath): sha(bookpath)}, sources, expected_config, declarations)
    require(marker['manifest_sha256'] == sha(fixture/'manifest.json') and marker['engine_complete_sha256'] == sha(fixture/'complete.json') and marker['spherical_joint_source_sha256'] == sources, 'Postcheck hash binding failed')
    bad_sources = dict(sources); bad_sources['round8_spherical_joint_runner.py'] = '0'*64
    require(rejected(lambda: finalize_run(fixture, {}, bad_sources, expected_config, declarations)), 'Changed source fingerprint passed postcheck')
    require(source_fingerprints() == sources, 'Source changed during CPU control')
    return dict(status='PASS', schema='trajkv.round8.spherical-joint-cpu-check.v1',
        scope='CPU synthetic actual codecs/NNLS plus stub bank orchestration; no model/GPU/SSH/quality/timing measurement',
        complete_2x2=True, observations=observations,
        full_requested_priority=224, selected_stages=43, ranking_bytes=448,
        complete_owner_bytes_N1792_all_four=8315013, added_retained_tensor_bytes=0,
        source_fingerprints=sources, frozen_spherical_v1_unchanged=True,
        fixed_J_sign_pairwise_joint_guard=True, constructor_exception_restores_dispatch=True,
        final_physical_gain_metadata_preserved=True, full_postcheck_hash_binding=True,
        negative_controls=7,
        limitation='No comparison guarantee across CDoff/on after residual regeneration/NNLS; no task monotonicity')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = controls(args.output)
    report['checker_sha256'] = sha(__file__)
    (args.output/'check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.output/'complete.json').write_text(json.dumps(dict(status='PASS', GPU_used=False,
        factorial_cells=4, report_sha256=sha(args.output/'check.json'))), encoding='utf-8')
    print(json.dumps(dict(status='PASS', factorial_cells=4, GPU_used=False)))


if __name__ == '__main__':
    main()
