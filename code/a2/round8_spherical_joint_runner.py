"""Isolated 2x2 full-row cosine CD x existing joint-gain runner.

No existing source is edited. CD selects J and the original folded norm gain;
the inherited bank then regenerates sign/alpha and optionally refits only the
existing effective gain/alpha at fixed J/sign. This is not a task monotonicity
claim or a joint search over all fields. The v1 spherical codec is reused.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path


SOURCE_FILES = ('round8_spherical_joint_runner.py', 'round8_spherical_codec.py',
                'round8_spherical_runner.py', 'round8_bank.py', 'round8_codec.py',
                'round8_joint_gain.py', 'round8_runner.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fingerprints():
    return {name: sha(Path(__file__).with_name(name)) for name in SOURCE_FILES}


def validate_variant(variant):
    if variant['method'] != 'round8_joint':
        return None
    assignment = variant.get('key_assignment', 'l2')
    if assignment not in ('l2', 'row_cosine_cd') or variant.get('key_amplitude', 'norm') != 'norm':
        raise ValueError('Factorial control supports only l2/row_cosine_cd and norm')
    joint = variant.get('joint_gain_refit', False)
    if not isinstance(joint, bool):
        raise ValueError('joint_gain_refit must be boolean')
    if variant.get('key_encoding', 'reencode') != 'reencode' or variant.get('value_encoding', 'binary') != 'binary' or variant.get('v_group_size', 'all') != 'all':
        raise ValueError('Factorial control freezes reencode old K and binary/all V')
    if variant.get('key_metric_path') or variant.get('encoder_book_path') or variant.get('value_joint_book_path') or variant.get('residual_layout_reference', 'actual') not in (None, 'actual'):
        raise ValueError('No metric, additional books or K2-reference-layout intervention')
    stages = variant.get('selected_stages')
    limit = variant.get('res_stages_limit', variant.get('res_units_limit', variant.get('residual_units_limit')))
    if stages is None or len(stages) != 224 or limit != 43:
        raise ValueError('Preserve all224 ranking entries and explicit limit43, not a truncated mask')
    seen = set()
    for item in stages:
        if len(item) != 4:
            raise ValueError('Expected [axis,layer,head,stage] ranking')
        axis, layer, head, stage = item
        if axis not in ('k', 'v') or not isinstance(layer, int) or not isinstance(head, int) or not 0 <= layer < 28 or not 0 <= head < 4 or stage != 0 or tuple(item) in seen:
            raise ValueError('Expected unique full28x4x2 first-stage ranking')
        seen.add(tuple(item))
    sweeps = int(variant.get('row_cosine_sweeps', 1))
    hard = variant.get('row_cosine_hard_accept', True)
    row_chunk = int(variant.get('row_cosine_chunk', 8192))
    if sweeps not in (1, 2) or not isinstance(hard, bool) or row_chunk < 1:
        raise ValueError('One/two sweeps, boolean hard guard and positive chunk required')
    return dict(cd=assignment == 'row_cosine_cd', joint=joint, sweeps=sweeps,
        hard_accept=hard, row_chunk=row_chunk, requested_stages=[tuple(x) for x in stages],
        reference_stages=[tuple(x) for x in stages[:43]], ranking_bytes=448)


def validate_config(config):
    declarations, common, cells, cd_options = {}, None, set(), None
    for variant in config['variants']:
        if variant['name'] in declarations:
            raise ValueError('Duplicate variant name')
        options = validate_variant(variant)
        declarations[variant['name']] = options
        if options is None:
            continue
        signature = dict(stages=options['requested_stages'],
            inputs={key: variant.get(key) for key in ('direction_book_path', 'metadata_reference_book_path',
                'metadata_factor_path', 'round8_factor_policy', 'budget_bits', 'depth_policy', 'cap')})
        if common is not None and signature != common:
            raise ValueError('Factorial cells must share complete ranking, book/factors and budget/cap')
        common = signature
        cell = (options['cd'], options['joint'])
        if cell in cells:
            raise ValueError('Each factorial cell must have exactly one declared variant')
        cells.add(cell)
        if options['cd']:
            settings = {key: options[key] for key in ('sweeps', 'hard_accept', 'row_chunk')}
            if cd_options is not None and settings != cd_options:
                raise ValueError('Both CD cells must share sweeps, hard guard and chunk')
            cd_options = settings
    if cells != {(False, False), (False, True), (True, False), (True, True)}:
        raise ValueError('Explicit complete CDoff/on x jointoff/on factorial required')
    return declarations


@contextmanager
def scoped_dispatch(bank_module, dispatch):
    original = bank_module.encode_key
    bank_module.encode_key = dispatch
    try:
        yield original
    finally:
        bank_module.encode_key = original


def spherical_joint_class(base_class, bank_module):
    """Constructor-only dispatcher; actual sign/joint logic stays inherited."""
    from round8_spherical_codec import encode_key as spherical_encode

    class SphericalJointBank(base_class):
        def _initialize(self, *arguments, **keywords):
            cfg = type(self).settings
            options = validate_variant(cfg)
            self.row_cosine_joint_options = options
            self.row_cosine_statistics = []
            if options is None:
                return super()._initialize(*arguments, **keywords)
            book_path, reference = Path(cfg['direction_book_path']), Path(cfg['metadata_reference_book_path'])
            if sha(book_path) != sha(reference):
                raise ValueError('All four cells must retain the old factor-reference G4_16 book')
            centers = json.loads(book_path.read_text(encoding='utf-8-sig'))['centers']
            if len(centers) != 16 or any(len(row) != 4 for row in centers):
                raise ValueError('Only global16x4 old book is supported')
            original = bank_module.encode_key

            def dispatch(*args, **kwargs):
                if kwargs.get('assignment', 'l2') != 'row_cosine_cd':
                    return original(*args, **kwargs)
                code, gain, measured = spherical_encode(*args, **kwargs,
                    sweeps=options['sweeps'], hard_accept=options['hard_accept'],
                    row_chunk=options['row_chunk'], return_stats=True)
                self.row_cosine_statistics.append(dict(layer_call_index=len(self.row_cosine_statistics), **measured))
                return code, gain

            if options['cd']:
                with scoped_dispatch(bank_module, dispatch):
                    result = super()._initialize(*arguments, **keywords)
            else:
                result = super()._initialize(*arguments, **keywords)
            if list(map(tuple, self.selected_residual_stages)) != options['reference_stages']:
                raise RuntimeError('Inherited budget changed the declared43-stage mask')
            if list(map(tuple, self.requested_residual_stages)) != options['requested_stages'] or self.ranking_constant_bytes != 448:
                raise RuntimeError('Inherited bank did not retain the full224 ranking/448B')
            if self.joint_gain_enabled != options['joint']:
                raise RuntimeError('Inherited joint-gain execution differs from declaration')
            return result

        def result_metadata(self):
            result = super().result_metadata()
            options = self.row_cosine_joint_options
            if options is None:
                return result
            result.update(stage='round8 isolated CD x existing joint-gain factorial; unchanged packed reader/owner',
                key_bcd_iterations=0,
                encoding_assignment=('full-row signed-cosine coordinate proposal with declared actual-FP16 base-SSE guard'
                    if options['cd'] else 'unchanged legacy normalized nearest assignment'),
                row_cosine_joint_factorial=dict(cd=options['cd'], joint=options['joint']),
                row_cosine_sweeps=options['sweeps'] if options['cd'] else 0,
                row_cosine_hard_accept=options['hard_accept'] if options['cd'] else None,
                row_cosine_statistics=self.row_cosine_statistics,
                row_cosine_statistics_stage='pre-residual and pre-joint base, not the final joint-adjusted base',
                row_cosine_extra_retained_tensor_constant_bytes=0,
                row_cosine_unchanged_serving_reader=True,
                row_cosine_source_coordinates='actual centered WHT; original mean/frame remains fixed',
                row_cosine_reference_mask_sha256=hashlib.sha256(json.dumps(self.selected_residual_stages, separators=(',', ':')).encode()).hexdigest(),
                row_cosine_joint_execution_order=[
                    'CD selects J and original norm->FP16->factor->FP16 gain' if options['cd'] else 'legacy l2 selects J and original folded norm gain',
                    'inherited bank regenerates sign/alpha from actual FP16 base',
                    'existing NNLS proposals refit stored effective gain/alpha at fixed J/sign, with hard-FP16 acceptance' if options['joint'] else 'joint refit disabled'],
                row_cosine_joint_guarantee_scope='CD guard covers its base before residual regeneration only; NNLS acceptance covers its given J/sign and existing physical gain/alpha; no cross-cell total-residual/task monotonicity guarantee',
                row_cosine_joint_added_fields=0,
                row_cosine_joint_source_sha256=source_fingerprints())
            return result
    return SphericalJointBank


def finalize_run(output, external, sources, config_sha256, declarations):
    """Postcheck is CPU-testable; marker is emitted only after all checks."""
    output = Path(output)
    if (output / 'complete_round8.json').exists():
        raise ValueError('Completed postcheck is immutable; use a new output')
    if any(sha(path) != expected for path, expected in external.items()) or source_fingerprints() != sources:
        raise RuntimeError('Input/source changed; preserve run and use a new output')
    complete_path, manifest_path = output / 'complete.json', output / 'manifest.json'
    complete = json.loads(complete_path.read_text(encoding='utf-8-sig'))
    if complete.get('status') != 'COMPLETE' or complete.get('records') != complete.get('expected') or not complete.get('expected', 0):
        raise RuntimeError('Engine did not produce every declared record')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if manifest.get('config_sha256') != config_sha256:
        raise RuntimeError('Engine consumed a different config')
    manifest.setdefault('external_codec_inputs', {}).update(external)
    manifest.update(stage='round8 isolated full2x2 row_cosine_cd x existing joint_gain_refit',
        round8_engine_complete_requires_postcheck=True,
        spherical_joint_declarations=declarations, spherical_joint_source_sha256=sources,
        spherical_joint_factorial_complete=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    marker = dict(status='COMPLETE', engine_complete_sha256=sha(complete_path),
        manifest_sha256=sha(manifest_path), encoder_inputs_unchanged=True,
        source_inputs_unchanged=True, isolated_spherical_joint_prototype=True,
        spherical_joint_source_sha256=sources)
    with (output / 'complete_round8.json').open('x', encoding='utf-8') as stream:
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
    declarations = validate_config(config)
    sources = source_fingerprints()
    if args.plan_only:
        print(json.dumps(dict(status='PLAN_ONLY', variants=declarations, source_sha256=sources,
            factorial_complete=True, GPU_executed=False, added_retained_tensor_bytes=0)), flush=True)
        return
    import round8_runner as original_runner
    import round8_bank
    import factory
    original_choose = factory.bank_class
    cls = spherical_joint_class(round8_bank.Round8Bank, round8_bank)
    def choose(method):
        return cls if method == 'round8_joint' else original_choose(method)
    external = {variant[key]: sha(variant[key]) for variant in config['variants']
                for key in original_runner.EXTRA_KEYS if variant.get(key)}
    factory.bank_class = choose
    executor, executor_original_choose = None, None
    try:
        import run_representation_variants
        executor = run_representation_variants
        executor_original_choose = executor.bank_class
        executor.bank_class = choose
        executor.main()
    finally:
        if executor is not None:
            executor.bank_class = executor_original_choose
        factory.bank_class = original_choose
    if sha(args.config) != config_sha256:
        raise RuntimeError('Config changed; preserve run and use a new output')
    finalize_run(args.output, external, sources, config_sha256, declarations)


if __name__ == '__main__':
    main()
