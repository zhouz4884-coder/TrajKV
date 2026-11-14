"""Register round-five banks for explicitly teacher-forced GPU diagnostics."""
import hashlib,json,sys
from pathlib import Path
import round5_runner
from run_continuous import main

if __name__=='__main__':
    path=Path(sys.argv[sys.argv.index('--config')+1])
    output=Path(sys.argv[sys.argv.index('--output')+1])
    config=json.loads(path.read_text())
    external={v[key]:hashlib.sha256(Path(v[key]).read_bytes()).hexdigest()
        for v in config['variants'] for key in ('direction_book_path','metadata_factor_path',
            'functional_v_factor_path','spectrum_artifact_path','metric_pq_artifact_path') if v.get(key)}
    main()
    for path,digest in external.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Diagnostic input changed during run')
    manifest=json.loads((output/'manifest.json').read_text())
    manifest['external_codec_inputs']=external
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
