"""Independent HF LLaVA first-question native/Full identity smoke.

The root schedules this on the existing data-disk launcher. Never reads task
references and never modifies frozen runners, model files or legacy scripts.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/experiments'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from round6_second_session import first_question_features, model_geometry, run_first_question
    torch.set_num_threads(4)
    torch.manual_seed(1212)
    torch.backends.cuda.matmul.allow_tf32 = False
    arguments = SimpleNamespace(**config['arguments'])
    if int(getattr(arguments, 'turns', 1)) != 1:
        raise ValueError('This adapter only supports the first question')
    arguments.max_tokens = int(getattr(arguments, 'max_tokens', 32))
    legacy = str(Path(arguments.legacy_scripts))
    if legacy not in sys.path:
        sys.path.insert(0, legacy)
    rows = json.loads(Path(arguments.samples).read_text())
    rows = rows[config.get('start', 0):config.get('start', 0) + config['count']]
    if len(rows) != config['count'] or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Incomplete/duplicate smoke subset')
    if config['variants'][0]['method'] != 'native_hf':
        raise ValueError('Native HF reference must run first')
    for variant in config['variants'][1:]:
        if variant['method'] not in ('full', 'direction_meta'):
            raise ValueError('Unapproved second-backbone smoke codec')
    processor = AutoProcessor.from_pretrained(arguments.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(arguments.model, local_files_only=True,
        dtype=torch.float16, device_map='cuda:0', attn_implementation='sdpa').eval()
    geometry = model_geometry(model)
    external = {v[key]:sha(v[key]) for v in config['variants']
                for key in ('direction_book_path', 'metadata_factor_path') if v.get(key)}
    manifest = dict(schema='trajkv.round6.second-backbone-first-question.v1', config=config,
        config_sha256=sha(args.config), samples_sha256=sha(arguments.samples),
        selected_ids=[row['id'] for row in rows], geometry=geometry,
        answers_visible_to_runner=False, model_weights_frozen=True, text_KV='FP16',
        model_config_sha256=sha(Path(arguments.model) / 'config.json'), external_codec_inputs=external,
        sources={str(path.relative_to(ROOT)):sha(path) for path in Path(__file__).parent.glob('round6_second_*.py')},
        torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        scope='second-model first-question smoke; development sources; not a new blind confirmation')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    all_full_matches, n = True, 0
    with torch.inference_mode(), (args.output / 'records.jsonl').open('x') as stream:
        for row in rows:
            features, visual, prompt_metadata = first_question_features(processor, model, row)
            native = None
            for variant in config['variants']:
                result, trace = run_first_question(model, processor, features, visual, arguments, variant, native)
                result.update(session_id=row['id'], image_id=row['image_id'],
                    dataset=row['dataset'], source_document_id=row.get('source_document_id'),
                    source_original_image_id=row.get('source_original_image_id'),
                    split=row['split'], model=arguments.model, prompt_metadata=prompt_metadata)
                if native is None:
                    native = trace
                if variant['method'] == 'full':
                    all_full_matches = all_full_matches and result['native_complete_tokens_equal']
                stream.write(json.dumps(result) + '\n')
                stream.flush()
                n += 1
                print(json.dumps(dict(done=n, id=row['id'], variant=variant['name'],
                    visual_tokens=result['visual_tokens'], native_tokens_equal=result['native_complete_tokens_equal'])), flush=True)
    for path, digest in external.items():
        if sha(path) != digest:
            raise RuntimeError('External codec artifact changed during smoke')
    (args.output / 'complete.json').write_text(json.dumps(dict(status='COMPLETE',
        records=n, expected=len(rows) * len(config['variants']), full_native_complete_tokens_all_equal=all_full_matches,
        identity_scope='free-running full token sequences, with same-prefix numerical logit diagnostics',
        compressed_variants_run=any(v['method'] == 'direction_meta' for v in config['variants']))))


if __name__ == '__main__':
    main()
