"""New campaign registration; Round5 inference sources remain untouched."""
import hashlib,json,sys
from pathlib import Path
import round5_runner
import factory
previous=factory.bank_class


def choose(method):
    if method=='round6_depth':
        from round6_depth_bank import DepthResidualBank
        return DepthResidualBank
    if method=='round6_gamma':
        from round6_gamma_bank import GammaResidualBank
        return GammaResidualBank
    return previous(method)


factory.bank_class=choose

if __name__=='__main__':
    cp=Path(sys.argv[sys.argv.index('--config')+1])
    out=Path(sys.argv[sys.argv.index('--output')+1])
    config=json.loads(cp.read_text())
    external={v[key]:hashlib.sha256(Path(v[key]).read_bytes()).hexdigest()
        for v in config['variants'] for key in ('gamma_factor_path','depth_policy_path') if v.get(key)}
    import run_representation_variants
    run_representation_variants.main()
    for path,digest in external.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Input changed during run: '+path)
    manifest=json.loads((out/'manifest.json').read_text())
    manifest['external_codec_inputs'].update(external)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
