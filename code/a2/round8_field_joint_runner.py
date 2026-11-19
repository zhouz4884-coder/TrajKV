"""Isolated L2 K-field x V-field refit factorial; frozen bank/reader unchanged.

The inherited bank does all base encoding and original sign generation. K uses
its existing joint_gain_refit branch. This subclass follows the V _encode_axis
call with native fixed-bit/sign m/S/alpha updates in the same physical slots.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path


SOURCE_FILES = ('round8_field_joint_runner.py', 'round8_value_joint_refit.py',
    'round8_bank.py', 'round8_codec.py', 'round8_attention.py', 'round8_joint_gain.py',
    'round8_runner.py', 'round6_depth_codec.py', 'round7_key_layout.py',
    'quantizers.py', 'affine_rate_bank.py', 'fast_wht.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fingerprints():
    result = {name: sha(Path(__file__).with_name(name)) for name in SOURCE_FILES}
    decoder = Path(__file__).resolve().parents[1] / 'experiments/fused_progressive.py'
    result['experiments/fused_progressive.py'] = sha(decoder)
    return result


def validate_variant(variant):
    if variant['method'] != 'round8_joint':
        return None
    if variant.get('key_assignment', 'l2') != 'l2' or variant.get('key_amplitude', 'norm') != 'norm':
        raise ValueError('Field factorial preserves original l2/norm K encoding; no CD')
    k, v = variant.get('joint_gain_refit', False), variant.get('value_joint_refit', False)
    if not isinstance(k, bool) or not isinstance(v, bool):
        raise ValueError('Both refit flags must be boolean')
    if variant.get('value_joint_rounds', 2) != 2:
        raise ValueError('First V field factorial fixes exactly two rounds')
    if variant.get('key_encoding', 'reencode') != 'reencode' or variant.get('value_encoding', 'binary') != 'binary' or variant.get('v_group_size', 'all') != 'all':
        raise ValueError('Old reencoded K and binary/all V are fixed')
    if variant.get('key_metric_path') or variant.get('encoder_book_path') or variant.get('value_joint_book_path') or variant.get('residual_layout_reference', 'actual') not in (None, 'actual'):
        raise ValueError('No extra books/metric or reference-layout side intervention')
    stages = variant.get('selected_stages')
    limit = variant.get('res_stages_limit', variant.get('res_units_limit', variant.get('residual_units_limit')))
    if stages is None or len(stages) != 224 or limit != 43:
        raise ValueError('Preserve complete224 ranking and explicit limit43')
    seen = set()
    for item in stages:
        if len(item) != 4:
            raise ValueError('Expected [axis,layer,head,stage]')
        axis, layer, head, depth = item
        if axis not in ('k', 'v') or not isinstance(layer, int) or not isinstance(head, int) or not 0 <= layer < 28 or not 0 <= head < 4 or depth != 0 or tuple(item) in seen:
            raise ValueError('Exactly the unique28x4x2 first-stage ranking is required')
        seen.add(tuple(item))
    return dict(key_joint=k, value_joint=v, value_rounds=2, ranking_bytes=448,
        requested_stages=[tuple(x) for x in stages], selected_stages=[tuple(x) for x in stages[:43]])


def validate_config(config):
    declarations, common, cells = {}, None, set()
    for variant in config['variants']:
        if variant['name'] in declarations:
            raise ValueError('Duplicate variant name')
        options = validate_variant(variant)
        declarations[variant['name']] = options
        if options is None:
            continue
        signature = dict(stages=options['requested_stages'], inputs={key: variant.get(key) for key in
            ('direction_book_path', 'metadata_reference_book_path', 'metadata_factor_path',
             'round8_factor_policy', 'budget_bits', 'depth_policy', 'cap')})
        if common is not None and common != signature:
            raise ValueError('Four cells must share book/factors/cap/budget and full stage ranking')
        common = signature
        cell = (options['key_joint'], options['value_joint'])
        if cell in cells:
            raise ValueError('Exactly one variant per factorial cell required')
        cells.add(cell)
    if cells != {(False, False), (False, True), (True, False), (True, True)}:
        raise ValueError('Complete Kjoint off/on x Vjoint off/on factorial required')
    return declarations


@contextmanager
def scoped_registration(factory_module, executor_module, choose):
    previous_factory, previous_executor = factory_module.bank_class, executor_module.bank_class
    factory_module.bank_class = executor_module.bank_class = choose
    try:
        yield
    finally:
        factory_module.bank_class, executor_module.bank_class = previous_factory, previous_executor


def field_joint_class(base_class):
    """CPU-stubbable subclass; production base is unmodified Round8Bank."""
    from round8_value_joint_refit import joint_refit_value

    class FieldJointBank(base_class):
        def _initialize(self, *args, **kwargs):
            cfg = type(self).settings
            self.field_joint_options = validate_variant(cfg)
            self.value_joint_statistics = []
            if self.field_joint_options is not None:
                book, reference = Path(cfg['direction_book_path']), Path(cfg['metadata_reference_book_path'])
                if sha(book) != sha(reference):
                    raise ValueError('Factorial requires unchanged old factor-reference book')
                centers = json.loads(book.read_text(encoding='utf-8-sig'))['centers']
                if len(centers) != 16 or any(len(row) != 4 for row in centers):
                    raise ValueError('Only old global16x4 K book is supported')
            result = super()._initialize(*args, **kwargs)
            options = self.field_joint_options
            if options is not None:
                if list(map(tuple, self.selected_residual_stages)) != options['selected_stages'] or list(map(tuple, self.requested_residual_stages)) != options['requested_stages'] or self.ranking_constant_bytes != 448:
                    raise RuntimeError('Inherited bank changed the full224 ranking/actual43 mask')
                if self.joint_gain_enabled != options['key_joint']:
                    raise RuntimeError('Inherited K refit differs from declaration')
                expected_v = sum(axis == 'v' for axis, _, _, _ in options['selected_stages']) if options['value_joint'] else 0
                if len(self.value_joint_statistics) != expected_v:
                    raise RuntimeError('V refit missed/duplicated an eligible selected head')
            return result

        def _encode_axis(self, layer, state, axis, selected, source, base):
            result = super()._encode_axis(layer, state, axis, selected, source, base)
            options = self.field_joint_options
            if axis != 'v' or options is None or not options['value_joint']:
                return result
            nv = source.shape[-2]
            if self.value_encoding != 'binary' or self.vbook is not None or self.vgroup != nv:
                raise ValueError('V field hook requires actual binary/all native frame')
            heads = sorted({head for item_axis, head, _ in selected if item_axis == 'v'})
            for head in heads:
                if sum(a == 'v' and h == head for a, h, _ in selected) != 1 or ('v', head, 0) not in selected:
                    raise ValueError('V field hook requires exactly one existing first sign stage')
                slot = selected.index(('v', head, 0))
                scale, minimum, alpha, stats = joint_refit_value(source[head], state.v_code[0, head],
                    state.v_scale[0, head], state.v_min[0, head],
                    state.depth_residual_code[slot], state.depth_residual_alpha[slot], rounds=2)
                state.v_scale[0, head].copy_(scale)
                state.v_min[0, head].copy_(minimum)
                state.depth_residual_alpha[slot].copy_(alpha)
                entries = [entry for entry in self.depth_distortions
                           if entry['axis'] == 'v' and entry['layer'] == layer and entry['head'] == head]
                if len(entries) != 1 or entries[0]['depth'] != 1:
                    raise RuntimeError('Expected exactly one inherited V distortion record')
                entry = entries[0]
                entry['pre_value_joint_squared_error_by_depth'] = list(entry['squared_error_by_depth'])
                entry['squared_error_by_depth'] = [stats['source_base_sse_after'], stats['source_sse_after']]
                entry['scope'] = 'post-refit native binary/all base and first sign stage, actual sequential FP16 decode; coordinate error only'
                entry['value_joint_refit'] = True
                self.value_joint_statistics.append(dict(layer=layer, head=head, **stats))
            return result

        def result_metadata(self):
            result = super().result_metadata()
            options = self.field_joint_options
            if options is None:
                return result
            result.update(stage='round8 isolated K/V existing-field refit factorial; unchanged packed reader/owner',
                field_joint_factorial=dict(key_joint=options['key_joint'], value_joint=options['value_joint']),
                value_joint_refit=options['value_joint'], value_joint_rounds=2,
                value_joint_statistics=self.value_joint_statistics,
                value_joint_frame_formula='Q16(Q16(Q16(S*b)+m)+alpha*fixed_sign); stored physical FP16 m/S/alpha',
                value_joint_frame_parameter_policy=('initial inherited folded frame, then native fixed-bit/sign refit in original m/S/alpha slots; no factor/offset reapplied'
                    if options['value_joint'] else 'original inherited folded binary/all frame and original first alpha'),
                value_joint_scope='selected first-stage V heads only; signs/bits/positions/grouping unchanged; total native hard SSE acceptance is not an attention/task guarantee',
                value_joint_added_retained_tensor_bytes=0, field_joint_added_fields=0,
                field_joint_reader_unchanged=True,
                field_joint_source_sha256=source_fingerprints())
            return result
    return FieldJointBank


def finalize_run(output, external, sources, config_sha256, declarations):
    output = Path(output)
    if (output / 'complete_round8.json').exists():
        raise ValueError('Completed postcheck is immutable; use a new output')
    if any(sha(path) != value for path, value in external.items()) or source_fingerprints() != sources:
        raise RuntimeError('Source/codec input changed; preserve run and use new output')
    complete_path, manifest_path = output/'complete.json', output/'manifest.json'
    complete = json.loads(complete_path.read_text(encoding='utf-8-sig'))
    if complete.get('status') != 'COMPLETE' or complete.get('records') != complete.get('expected') or not complete.get('expected', 0):
        raise RuntimeError('Incomplete engine record coverage')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if manifest.get('config_sha256') != config_sha256:
        raise RuntimeError('Engine consumed a different config')
    manifest.setdefault('external_codec_inputs', {}).update(external)
    manifest.update(stage='round8 complete Kjoint x Vjoint existing-field factorial',
        round8_engine_complete_requires_postcheck=True, field_joint_declarations=declarations,
        field_joint_source_sha256=sources, field_joint_factorial_complete=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    marker = dict(status='COMPLETE', engine_complete_sha256=sha(complete_path), manifest_sha256=sha(manifest_path),
        records_sha256=sha(output/'records.jsonl'),
        encoder_inputs_unchanged=True, source_inputs_unchanged=True,
        isolated_field_joint_prototype=True, field_joint_source_sha256=sources)
    with (output/'complete_round8.json').open('x', encoding='utf-8') as stream:
        json.dump(marker, stream, indent=2)
    return marker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    config_sha256 = sha(args.config)
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    declarations, sources = validate_config(config), source_fingerprints()
    if args.plan_only:
        print(json.dumps(dict(status='PLAN_ONLY', factorial_complete=True, variants=declarations,
            source_sha256=sources, GPU_executed=False, added_retained_tensor_bytes=0)), flush=True)
        return
    import round8_runner as original_runner
    import round8_bank
    import factory
    import run_representation_variants as executor
    previous = factory.bank_class
    cls = field_joint_class(round8_bank.Round8Bank)
    def choose(method):
        return cls if method == 'round8_joint' else previous(method)
    external = {variant[key]: sha(variant[key]) for variant in config['variants']
                for key in original_runner.EXTRA_KEYS if variant.get(key)}
    with scoped_registration(factory, executor, choose):
        executor.main()
    if sha(args.config) != config_sha256:
        raise RuntimeError('Config changed; preserve run and use new output')
    finalize_run(args.output, external, sources, config_sha256, declarations)


if __name__ == '__main__':
    main()
