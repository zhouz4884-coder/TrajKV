"""Register the new Round8 backend without altering frozen baseline modules."""
import hashlib
import json
from pathlib import Path
import sys

import round7_runner
import factory

previous = factory.bank_class


def choose(method):
    if method == 'round8_joint':
        from round8_bank import Round8Bank
        return Round8Bank
    return previous(method)


factory.bank_class = choose

EXTRA_KEYS = ('direction_book_path', 'metadata_factor_path', 'metadata_reference_book_path',
              'encoder_book_path', 'value_joint_book_path', 'round7_policy_path',
              'depth_policy_path')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == '__main__':
    cp = Path(sys.argv[sys.argv.index('--config') + 1])
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    config = json.loads(cp.read_text(encoding='utf-8-sig'))
    external = {variant[key]: sha(variant[key]) for variant in config['variants']
                for key in EXTRA_KEYS if variant.get(key)}
    import run_representation_variants
    run_representation_variants.main()
    if any(sha(path) != value for path, value in external.items()):
        (output / 'round8_postcheck.json').write_text(json.dumps(dict(status='FAILED_INPUT_CHANGED')),
                                                     encoding='utf-8')
        raise RuntimeError('Round8 encoder input changed; preserve this run and use a new run')
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    manifest['external_codec_inputs'].update(external)
    manifest['stage'] = 'round8 actual packed full-model image-boundary free generation'
    manifest['round8_engine_complete_requires_postcheck'] = True
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (output / 'complete_round8.json').write_text(json.dumps(dict(
        status='COMPLETE', engine_complete_sha256=sha(output / 'complete.json'),
        manifest_sha256=sha(manifest_path), encoder_inputs_unchanged=True), indent=2), encoding='utf-8')
