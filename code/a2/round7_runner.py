"""Round7 registration around preserved prior full-model GPU executors."""
import hashlib
import json
import sys
from pathlib import Path
import round6_runner
import factory

previous = factory.bank_class


def choose(method):
    if method == 'round7_key':
        from round7_key_bank import KeyBank
        return KeyBank
    return previous(method)


factory.bank_class = choose


if __name__ == '__main__':
    path = Path(sys.argv[sys.argv.index('--config') + 1])
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    extras = {v[key]: hashlib.sha256(Path(v[key]).read_bytes()).hexdigest()
              for v in config['variants']
              for key in ('key_metric_path', 'round7_policy_path', 'depth_policy_path', 'gamma_factor_path')
              if v.get(key)}
    import run_representation_variants
    run_representation_variants.main()
    for file, digest in extras.items():
        if hashlib.sha256(Path(file).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Round7 input changed during execution: ' + file)
    manifest = json.loads((output / 'manifest.json').read_text())
    manifest['external_codec_inputs'].update(extras)
    manifest['stage'] = 'round7: actual packed full-model image-boundary free generation'
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
