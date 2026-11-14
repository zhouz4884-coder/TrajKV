"""Register new round-five backends without changing frozen runners."""
import sys,hashlib,json
from pathlib import Path
import factory
original=factory.bank_class

def choose(method):
    if method=='gain_quant_residual':
        from gain_quant_residual_bank import GainQuantResidualBank
        return GainQuantResidualBank
    if method=='mixed_residual':
        from mixed_residual_bank import MixedResidualBank
        return MixedResidualBank
    if method=='mean_corrected':
        from mean_corrected_bank import MeanCorrectedBank
        return MeanCorrectedBank
    if method=='metric_pq':
        from metric_pq_bank import MetricPQBank
        return MetricPQBank
    if method=='sparse_v_residual':
        from sparse_v_residual_bank import SparseVResidualBank
        return SparseVResidualBank
    if method=='query_spectrum_port':
        from query_spectrum_port import QuerySpectrumPort
        return QuerySpectrumPort
    if method=='direction_functional_v':
        from functional_v_bank import FunctionalVBank
        return FunctionalVBank
    return original(method)

factory.bank_class=choose

if __name__=='__main__':
    config_path=Path(sys.argv[sys.argv.index('--config')+1])
    output=Path(sys.argv[sys.argv.index('--output')+1])
    config=json.loads(config_path.read_text())
    extras={v[k]:hashlib.sha256(Path(v[k]).read_bytes()).hexdigest()
            for v in config['variants'] for k in ('functional_v_factor_path','spectrum_artifact_path','metric_pq_artifact_path','interaction_policy_path') if v.get(k)}
    import run_representation_variants
    run_representation_variants.main()
    for path,digest in extras.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('External input changed during run: '+path)
    manifest=json.loads((output/'manifest.json').read_text())
    manifest['external_codec_inputs'].update(extras)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
