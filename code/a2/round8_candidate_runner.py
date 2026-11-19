"""Thin selected-candidate entry point; every numerical backend stays frozen.

Config must declare backend_family='spherical' or 'field'. Within one process
only, replace the chosen module's factorial-config validator and source-index
function. Call its original main unchanged. Its hardcoded full-factorial
metadata is preserved as a before_candidate_metadata audit copy, then corrected
and rebound in the final manifest/complete marker; raw producer records are
never rewritten. A subset is always declared as such, even if all cells occur.

For new diagnostic/profile wrappers, ``with backend_registration(family) as
cls`` registers the actual chosen subclass and yields it. Set cls.settings and
the diagnostic executor's ProgressiveBank explicitly, and keep all execution
and metadata collection inside that context. No old diagnostic is edited.
"""
import argparse
from contextlib import contextmanager, redirect_stdout
import importlib
import io
import json
from pathlib import Path


FAMILIES = {'spherical': 'round8_spherical_joint_runner', 'field': 'round8_field_joint_runner'}


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_backend(family):
    if family not in FAMILIES:
        raise ValueError('Explicit backend_family must be spherical or field')
    return importlib.import_module(FAMILIES[family])


def validate_subset_config(config, family, backend):
    if config.get('backend_family') != family:
        raise ValueError('Config must explicitly declare the one selected backend_family')
    if not config.get('variants'):
        raise ValueError('At least one explicitly named candidate/control is required')
    declarations, common = {}, None
    for variant in config['variants']:
        if variant['name'] in declarations:
            raise ValueError('Duplicate variant name')
        if variant.get('backend_family', family) != family:
            raise ValueError('Cross-family mixing is not supported')
        if family == 'spherical' and variant.get('value_joint_refit', False):
            raise ValueError('Spherical backend does not implement V field refit')
        if family == 'field' and variant.get('key_assignment', 'l2') == 'row_cosine_cd':
            raise ValueError('Field backend does not implement spherical CD')
        options = backend.validate_variant(variant)
        declarations[variant['name']] = options
        if options is None:
            if variant.get('method') != 'round8_joint' and (variant.get('joint_gain_refit', False)
                    or variant.get('value_joint_refit', False) or variant.get('key_assignment') == 'row_cosine_cd'):
                raise ValueError('Intervention fields cannot be silently ignored by a control method')
            continue
        signature = dict(stages=options['requested_stages'], inputs={key: variant.get(key) for key in
            ('direction_book_path', 'metadata_reference_book_path', 'metadata_factor_path',
             'round8_factor_policy', 'budget_bits', 'depth_policy', 'cap')})
        if common is not None and common != signature:
            raise ValueError('Selected format candidates must share book/factors/cap/budget and complete ranking')
        common = signature
    return declarations


@contextmanager
def scoped_backend(family):
    """The only two function replacements used by the candidate CLI."""
    backend = load_backend(family)
    original_validate, original_sources = backend.validate_config, backend.source_fingerprints
    def selected(config):
        return validate_subset_config(config, family, backend)
    def sources():
        result = dict(original_sources())
        result['round8_candidate_runner.py'] = sha(__file__)
        return result
    backend.validate_config, backend.source_fingerprints = selected, sources
    try:
        yield backend
    finally:
        backend.validate_config, backend.source_fingerprints = original_validate, original_sources


@contextmanager
def backend_registration(family):
    """Yield the frozen numerical subclass; restore both registry pointers.

    This imports the original R8 registration first, as the standard runners do.
    Only factory.bank_class and run_representation_variants.bank_class are
    patched. A direct session caller still sets session.ProgressiveBank=cls.
    """
    with scoped_backend(family) as backend:
        import round8_runner  # noqa: F401; retain its original baseline registration.
        import round8_bank
        import factory
        import run_representation_variants as executor
        cls = (backend.spherical_joint_class(round8_bank.Round8Bank, round8_bank)
               if family == 'spherical' else backend.field_joint_class(round8_bank.Round8Bank))
        previous_factory, previous_executor = factory.bank_class, executor.bank_class
        def choose(method):
            return cls if method == 'round8_joint' else previous_factory(method)
        factory.bank_class = executor.bank_class = choose
        try:
            yield cls
        finally:
            factory.bank_class, executor.bank_class = previous_factory, previous_executor


def finish_subset_metadata(output, family, expected_sources):
    """Rebind only this new run's metadata after the original full postcheck.

    Remove the standard complete marker first: if any wrapper check fails,
    downstream analysis cannot mistake the backend's intermediate template
    marker for successful subset completion. Preserve both original byte sets.
    """
    output = Path(output)
    marker_path, manifest_path = output/'complete_round8.json', output/'manifest.json'
    audit = output/'audit'
    prior_marker = audit/'backend_complete_round8_before_candidate.json'
    prior_manifest = audit/'backend_manifest_before_candidate.json'
    if prior_marker.exists() or prior_manifest.exists() or (output/'candidate_subset_postcheck.json').exists() or (output/'complete_candidate.json').exists():
        raise ValueError('Do not overwrite prior subset metadata evidence')
    audit.mkdir(exist_ok=True)
    marker_path.replace(prior_marker)
    marker = json.loads(prior_marker.read_text(encoding='utf-8-sig'))
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode('utf-8-sig'))
    if marker.get('status') != 'COMPLETE' or marker.get('manifest_sha256') != sha(manifest_path) or marker.get('engine_complete_sha256') != sha(output/'complete.json'):
        raise RuntimeError('Backend postcheck identity mismatch; preserved as an incomplete subset')
    source_key = 'spherical_joint_source_sha256' if family == 'spherical' else 'field_joint_source_sha256'
    if marker.get(source_key) != expected_sources or manifest.get(source_key) != expected_sources:
        raise RuntimeError('Backend postcheck did not pin the candidate wrapper source')
    if expected_sources.get('round8_candidate_runner.py') != sha(__file__):
        raise RuntimeError('Candidate wrapper source changed after backend postcheck')
    if manifest.get('config', {}).get('backend_family') != family:
        raise RuntimeError('Producer manifest does not preserve the selected backend family')
    if any(sha(path) != value for path, value in manifest.get('external_codec_inputs', {}).items()):
        raise RuntimeError('Codec input changed after backend postcheck')
    records_sha = sha(output/'records.jsonl')
    if marker.get('records_sha256', records_sha) != records_sha:
        raise RuntimeError('Producer records changed after backend postcheck')
    with prior_manifest.open('xb') as stream:
        stream.write(manifest_bytes)
    manifest.update(stage='round8 selected candidate subset using frozen '+family+' backend',
        backend_family=family, candidate_subset=True, factorial_complete=False,
        candidate_wrapper_sha256=sha(__file__),
        executed_candidate_variants=[variant['name'] for variant in manifest['config']['variants']],
        record_stage_scope='raw record stages are unchanged backend-family labels; executed candidates and subset scope are fixed by this manifest')
    for key in ('spherical_joint_factorial_complete', 'field_joint_factorial_complete'):
        if key in manifest:
            manifest[key] = False
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    postcheck = dict(status='COMPLETE', backend_family=family, candidate_subset=True, factorial_complete=False,
        records_sha256=records_sha, engine_complete_sha256=sha(output/'complete.json'),
        manifest_sha256=sha(manifest_path), source_sha256=expected_sources,
        backend_manifest_before_metadata_sha256=sha(prior_manifest),
        backend_complete_before_metadata_sha256=sha(prior_marker),
        scope='only subset/full-factorial metadata is corrected; original numerical backend, raw records and source hashes are unchanged')
    postcheck_path = output/'candidate_subset_postcheck.json'
    with postcheck_path.open('x', encoding='utf-8') as stream:
        json.dump(postcheck, stream, indent=2)
    marker.update(backend_family=family, candidate_subset=True, factorial_complete=False,
        manifest_sha256=sha(manifest_path), records_sha256=records_sha,
        candidate_wrapper_sha256=sha(__file__), candidate_subset_postcheck_sha256=sha(postcheck_path))
    with marker_path.open('x', encoding='utf-8') as stream:
        json.dump(marker, stream, indent=2)
    candidate = dict(status='COMPLETE', backend_family=family, candidate_subset=True, factorial_complete=False,
        complete_round8_sha256=sha(marker_path), manifest_sha256=sha(manifest_path), records_sha256=records_sha,
        config_sha256=manifest.get('config_sha256'), wrapper_sha256=sha(__file__),
        source_sha256=expected_sources, candidate_subset_postcheck_sha256=sha(postcheck_path),
        accepted_only_after_this_marker=True)
    with (output/'complete_candidate.json').open('x', encoding='utf-8') as stream:
        json.dump(candidate, stream, indent=2)
    return candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    family = config.get('backend_family')
    with scoped_backend(family) as backend:
        if args.plan_only:
            stream = io.StringIO()
            with redirect_stdout(stream):
                backend.main()
            for line in stream.getvalue().splitlines():
                value = json.loads(line)
                if value.get('status') == 'PLAN_ONLY':
                    value.update(backend_family=family, candidate_subset=True, factorial_complete=False)
                print(json.dumps(value), flush=True)
            return
        backend.main()
        sources = backend.source_fingerprints()
        marker = finish_subset_metadata(args.output, family, sources)
        print(json.dumps(dict(status=marker['status'], backend_family=family,
            candidate_subset=True, factorial_complete=False)), flush=True)


if __name__ == '__main__':
    main()
