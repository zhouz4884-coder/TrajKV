"""Independent runner for row_cosine_cd; never edits frozen bank/codec sources.

The original method name round8_joint and tensor owner stay unchanged. Only
the declared new assignment gets a scoped constructor-time encode dispatcher.
All l2 calls go directly to the original function, and patches restore on error.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_variant(variant):
    if variant.get('key_assignment', 'l2') != 'row_cosine_cd':
        return None
    if variant['method'] != 'round8_joint' or variant.get('key_amplitude', 'norm') != 'norm':
        raise ValueError('row_cosine_cd is a separate round8_joint/norm assignment')
    if variant.get('key_encoding', 'reencode') != 'reencode' or variant.get('value_encoding', 'binary') != 'binary' or variant.get('joint_gain_refit', False):
        raise ValueError('First spherical control freezes old reencode K, binary V and disables joint-gain refit')
    if variant.get('v_group_size', 'all') != 'all':
        raise ValueError('First spherical control keeps the frozen all-image V frame')
    if variant.get('key_metric_path') or variant.get('residual_layout_reference', 'actual') not in (None, 'actual'):
        raise ValueError('First spherical control has no metric or K2-reference-layout side intervention')
    stages = variant.get('selected_stages')
    limit = variant.get('res_stages_limit', variant.get('res_units_limit', variant.get('residual_units_limit')))
    if stages is None or len(stages) < 43 or (len(stages) != 43 and limit != 43) or (limit is not None and limit != 43):
        raise ValueError('Explicit frozen43-stage mask/prefix required; no silent mask/budget override')
    chosen, seen = [tuple(item) for item in stages[:43]], set()
    for item in chosen:
        if len(item) != 4 or item[0] not in ('k', 'v') or item[1] < 0 or item[2] < 0 or item[3] not in (0, 1) or item in seen or (item[3] == 1 and (*item[:3], 0) not in seen):
            raise ValueError('Invalid explicit spherical reference stage prefix')
        seen.add(item)
    sweeps = int(variant.get('row_cosine_sweeps', 1))
    hard = variant.get('row_cosine_hard_accept', True)
    row_chunk = int(variant.get('row_cosine_chunk', 8192))
    if sweeps not in (1, 2) or not isinstance(hard, bool) or row_chunk < 1:
        raise ValueError('One/two sweeps, boolean hard guard and positive row chunk required')
    return dict(sweeps=sweeps, hard_accept=hard, row_chunk=row_chunk, reference_stages=chosen)


@contextmanager
def scoped_dispatch(bank_module, dispatch):
    original = bank_module.encode_key
    bank_module.encode_key = dispatch
    try:
        yield original
    finally:
        bank_module.encode_key = original


def spherical_class(base_class, bank_module):
    """Lazy class factory also permits CPU stub checks without importing Triton."""
    from round8_spherical_codec import encode_key as spherical_encode

    class SphericalBank(base_class):
        def _initialize(self, *arguments, **keywords):
            cfg = type(self).settings
            options = validate_variant(cfg)
            self.row_cosine_statistics = None
            if options is None:
                return super()._initialize(*arguments, **keywords)
            book_path, reference = Path(cfg['direction_book_path']), Path(cfg['metadata_reference_book_path'])
            if sha(book_path) != sha(reference):
                raise ValueError('First spherical prototype must use the old factor-reference G4_16 book unchanged')
            centers = json.loads(book_path.read_text(encoding='utf-8-sig'))['centers']
            if len(centers) != 16 or any(len(row) != 4 for row in centers):
                raise ValueError('First spherical prototype is only global G4_16')
            original = bank_module.encode_key
            statistics = []

            def dispatch(*args, **kwargs):
                if kwargs.get('assignment', 'l2') != 'row_cosine_cd':
                    return original(*args, **kwargs)
                code, gain, measured = spherical_encode(*args, **kwargs,
                    sweeps=options['sweeps'], hard_accept=options['hard_accept'],
                    row_chunk=options['row_chunk'], return_stats=True)
                statistics.append(dict(layer_call_index=len(statistics), **measured))
                return code, gain

            with scoped_dispatch(bank_module, dispatch):
                result = super()._initialize(*arguments, **keywords)
            if list(map(tuple, self.selected_residual_stages)) != options['reference_stages']:
                raise RuntimeError('Actual budget selected a different43-stage mask; preserve run rather than changing it')
            self.row_cosine_statistics = statistics
            return result

        def result_metadata(self):
            result = super().result_metadata()
            if self.row_cosine_statistics is None:
                return result
            cfg = type(self).settings
            result.update(stage='round8 isolated full-row cosine assignment prototype; unchanged serving tensor format',
                encoding_assignment='one/two full-row signed-cosine coordinate sweeps; original J retained, actual FP16 base-SSE guard declared separately',
                key_bcd_iterations=0, row_cosine_sweeps=int(cfg.get('row_cosine_sweeps', 1)),
                row_cosine_hard_accept=bool(cfg.get('row_cosine_hard_accept', True)),
                row_cosine_statistics=self.row_cosine_statistics,
                row_cosine_extra_retained_tensor_constant_bytes=0,
                row_cosine_guarantee_scope='valid rows and accepted base reconstructions only; original bank recomputes all K/V sign residuals, with no post-residual or task-loss monotonicity claim',
                row_cosine_source_coordinates='actual centered WHT source; no internal re-centering/rotation',
                row_cosine_reference_book_sha256=getattr(self, 'metadata_reference_book_hash', None),
                row_cosine_reference_mask_sha256=hashlib.sha256(json.dumps(self.selected_residual_stages, separators=(',', ':')).encode()).hexdigest(),
                row_cosine_complexity='per row O(sweeps*32*(16*4+128)); transient chunk scratch and extra encoding compute are paid',
                row_cosine_unchanged_serving_reader=True)
            return result
    return SphericalBank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    declarations = {variant['name']: validate_variant(variant) for variant in config['variants']}
    if args.plan_only:
        print(json.dumps(dict(status='PLAN_ONLY', variants=declarations,
            GPU_executed=False, added_retained_tensor_bytes=0)), flush=True)
        return
    # The old registrations and bank files are imported, never rewritten.
    import round8_runner as original_runner
    import round8_bank
    import factory
    original_choose = factory.bank_class
    cls = spherical_class(round8_bank.Round8Bank, round8_bank)

    def choose(method):
        return cls if method == 'round8_joint' else original_choose(method)

    external = {variant[key]: sha(variant[key]) for variant in config['variants']
                for key in original_runner.EXTRA_KEYS if variant.get(key)}
    factory.bank_class = choose
    executor = None
    executor_original_choose = None
    try:
        import run_representation_variants
        executor = run_representation_variants
        executor_original_choose = executor.bank_class
        executor.bank_class = choose
        run_representation_variants.main()
    finally:
        if executor is not None:
            executor.bank_class = executor_original_choose
        factory.bank_class = original_choose
    if any(sha(path) != digest for path, digest in external.items()):
        raise RuntimeError('Spherical input changed; preserve run and use a new output')
    manifest_path = args.output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['external_codec_inputs'].update(external)
    manifest.update(stage='round8 isolated row_cosine_cd assignment prototype',
        round8_engine_complete_requires_postcheck=True,
        spherical_declarations=declarations,
        spherical_source_sha256={name: sha(Path(__file__).with_name(name)) for name in
            ('round8_spherical_codec.py', 'round8_spherical_runner.py')})
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (args.output / 'complete_round8.json').write_text(json.dumps(dict(status='COMPLETE',
        engine_complete_sha256=sha(args.output / 'complete.json'), manifest_sha256=sha(manifest_path),
        encoder_inputs_unchanged=True, isolated_spherical_prototype=True), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
