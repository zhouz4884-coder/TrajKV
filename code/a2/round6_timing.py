"""Round6 backends in the established quiet, identical-output GPU timer."""
import hashlib
import json
from pathlib import Path
import sys
import round6_runner
from run_metadata_timing import main


if __name__ == '__main__':
    config = json.loads(Path(sys.argv[sys.argv.index('--config') + 1]).read_text())
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    additional = {v[key]: hashlib.sha256(Path(v[key]).read_bytes()).hexdigest()
                  for v in config['variants'] for key in ('depth_policy_path', 'gamma_factor_path') if v.get(key)}
    main()
    for path, digest in additional.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Codec input changed during timing: ' + path)
    manifest = json.loads((output / 'manifest.json').read_text())
    manifest['external_codec_inputs'].update(additional)
    manifest['sources'].update({'code/a2/' + name: digest for name, digest in list(manifest['sources'].items())})
    root = Path(__file__).resolve().parents[2]
    for name in ('code/experiments/bank.py', 'code/experiments/fused_progressive.py', 'code/bytes.py'):
        manifest['sources'][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
