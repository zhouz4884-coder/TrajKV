"""R7 official-template LLaVA-Next native/Full first-question smoke.

No visual-token padding or quantized-weight substitution is permitted. The
existing R6 session adapter stays frozen; its real token count is reported.
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
    arguments = SimpleNamespace(**config['arguments'])
    model_config = json.loads((Path(arguments.model) / 'config.json').read_text())
    if model_config.get('quantization_config'):
        raise ValueError('Native FP16 smoke cannot use AWQ/GPTQ model weights')
    if model_config['model_type'] != 'llava_next':
        raise ValueError('R7 strong-backbone smoke requires actual LLaVA-Next')
    if sha(Path(arguments.model) / 'config.json') != config['expected_model_config_sha256']:
        raise ValueError('Installed model identity changed since inventory')
    template_path = Path(config['official_chat_template_path'])
    if sha(template_path) != config['official_chat_template_sha256']:
        raise ValueError('Official processor template bytes changed')
    if config['variants'][0]['method'] != 'native_hf' or any(x['method'] != 'full' for x in config['variants'][1:]):
        raise ValueError('This entry point only enables native HF and Full')
    rows = json.loads(Path(arguments.samples).read_text())
    rows = rows[config.get('start', 0):config.get('start', 0) + config['count']]
    if len(rows) != config['count'] or len({row['image_id'] for row in rows}) != len(rows):
        raise ValueError('Need the predeclared number of distinct source images')
    if int(arguments.turns) != 1:
        raise ValueError('Strong-backbone parity entry point is first-question only')
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from round6_second_session import first_question_features, model_geometry, run_first_question
    torch.set_num_threads(4)
    torch.manual_seed(1212)
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, arguments.legacy_scripts)
    processor = AutoProcessor.from_pretrained(arguments.model, local_files_only=True)
    installed_template = getattr(processor, 'chat_template', None) or getattr(processor.tokenizer, 'chat_template', None)
    injected = not bool(installed_template)
    if injected:
        processor.chat_template = json.loads(template_path.read_text())['chat_template']
    model = AutoModelForImageTextToText.from_pretrained(arguments.model, local_files_only=True,
        dtype=torch.float16, device_map='cuda:0', attn_implementation='sdpa').eval()
    geometry = model_geometry(model)
    manifest = dict(schema='trajkv.round7.llava-next-native-full-smoke.v1', config=config,
        config_sha256=sha(args.config), samples_sha256=sha(arguments.samples), geometry=geometry,
        selected_ids=[r['id'] for r in rows], official_template_injected_if_missing=injected,
        template_provenance=config['official_chat_template_source'],
        original_session_adapter_sha256=sha(Path(__file__).with_name('round6_second_session.py')),
        source_sha256=sha(__file__), answers_visible_to_runner=False, model_weights_frozen=True,
        weight_quantization=False, weight_dtype=str(model.dtype), text_KV='FP16',
        visual_padding_allowed=False, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        scope='4 development source images, real variable Nv, first-question free-running identity; no packed or task-generalization claim')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    matches, n, counts = True, 0, []
    with torch.inference_mode(), (args.output / 'records.jsonl').open('x') as stream:
        for row in rows:
            features, visual, prompt = first_question_features(processor, model, row)
            counts.append(int(visual.numel()))
            native = None
            for variant in config['variants']:
                result, trace = run_first_question(model, processor, features, visual, arguments, variant, native)
                result.update(session_id=row['id'], dataset=row['dataset'], image_id=row['image_id'],
                    source_document_id=row.get('source_document_id'), split=row['split'],
                    model=arguments.model, prompt_metadata=prompt,
                    visual_tokens_mod32=int(visual.numel()) % 32)
                if native is None:
                    native = trace
                if variant['method'] == 'full':
                    matches = matches and result['native_complete_tokens_equal']
                stream.write(json.dumps(result) + '\n')
                stream.flush()
                n += 1
                print(json.dumps(dict(done=n, id=row['id'], variant=variant['name'],
                    visual_tokens=result['visual_tokens'], padded_visual_tokens=0,
                    native_complete_tokens_equal=result['native_complete_tokens_equal'])), flush=True)
    (args.output / 'complete.json').write_text(json.dumps(dict(status='COMPLETE', records=n,
        expected=len(rows) * len(config['variants']), full_native_complete_tokens_all_equal=matches,
        actual_visual_token_counts=counts, actual_visual_token_counts_mod32=[x % 32 for x in counts],
        padded_visual_tokens=0, packed_codec_run=False)))


if __name__ == '__main__':
    main()
