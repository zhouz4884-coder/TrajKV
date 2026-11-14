"""Collect fixed-Full-query local comparisons from real GPU first questions."""
import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import session_boundary
from bank import install_legacy
from fixed_query_bank import FixedQueryBank


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    arg = p.parse_args()
    c = json.loads(arg.config.read_text())
    arg.output.mkdir(parents=True, exist_ok=False)
    a = argparse.Namespace(**c['arguments'])
    for key in ('model', 'legacy_scripts', 'kivi_source', 'samples'):
        setattr(a, key, Path(getattr(a, key)))
    for key, value in dict(max_tokens=16, margin=0., max_replays=0, save_states=False,
                           read_budget_bytes=None, gain_table_data=None).items():
        if not hasattr(a, key):
            setattr(a, key, value)
    a.turns = 1
    install_legacy(a.legacy_scripts)
    torch.manual_seed(1212)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    chosen, seen = [], set()
    for row in json.loads(a.samples.read_text(encoding='utf-8')):
        identity = row.get('source_document_id') or row.get('image_id') or row.get('source_image_id')
        if not identity:
            raise ValueError('Missing source identity')
        if identity in seen:
            continue
        seen.add(identity)
        chosen.append(row)
        if len(chosen) >= int(c.get('images', 16)):
            break
    if not chosen:
        raise ValueError('No source samples')
    settings = c['codec']
    manifest = dict(config=c, config_sha256=sha(arg.config), samples_sha256=sha(a.samples),
        selected_ids=[r['id'] for r in chosen], labels_used=False,
        generation_scope='actual Full first-question free generation; no candidate states or outputs committed',
        external_inputs={k: sha(v) for k, v in settings.items() if k.endswith('_path')},
        torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        source_sha256={path.name: sha(path) for path in Path(__file__).parent.glob('*.py')})
    (arg.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    processor = AutoProcessor.from_pretrained(a.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(a.model, local_files_only=True,
        dtype=torch.float16, device_map='cuda:0', attn_implementation='sdpa').eval()
    FixedQueryBank.settings = settings
    FixedQueryBank.sink = []
    session_boundary.ProgressiveBank = FixedQueryBank
    aggregates = defaultdict(lambda: defaultdict(float))
    coverage = []
    with torch.inference_mode(), (arg.output/'records.jsonl').open('w') as records, \
            (arg.output/'local_metrics.jsonl').open('w') as metrics:
        for index, row in enumerate(chosen):
            start = time.perf_counter()
            result, _ = session_boundary.run_session(model, processor, row, a, 'fixed_query_collector', 'base')
            item = FixedQueryBank.sink.pop()
            counts = item['sample_counts']
            layers = len(model.model.language_model.layers)
            if any(counts.get((layer, 'question_prefill'), 0) != 1 for layer in range(layers)):
                raise RuntimeError('Missing exactly one final-question query for each layer')
            row_coverage = dict(session_id=row['id'], question_prefill_layers=layers,
                decode_calls_per_layer=[counts.get((layer, 'decode'), 0) for layer in range(layers)])
            coverage.append(row_coverage)
            for metric in item['local_metrics']:
                metric['session_id'] = row['id']
                metrics.write(json.dumps(metric)+'\n')
                key = (metric['candidate'], metric['value_representation'], metric['phase'])
                aggregate = aggregates[key]
                aggregate['events'] += 1
                for name, value in metric.items():
                    if name.endswith(('_sse', '_sum', '_elements')) or name in ('head_count',):
                        aggregate[name] += value
                aggregate['top1_equal_head_sum'] += metric['top1_same_fraction']*metric['head_count']
            metrics.flush()
            records.write(json.dumps(result)+'\n')
            records.flush()
            print(json.dumps(dict(done=index+1, total=len(chosen), id=row['id'],
                local_rows=len(item['local_metrics']), seconds=time.perf_counter()-start)), flush=True)
    table = []
    for (candidate, values, phase), sums in sorted(aggregates.items()):
        safe = lambda numerator, denominator: sums[numerator]/sums[denominator] if sums[denominator] else None
        table.append(dict(candidate=candidate, value_representation=values, phase=phase, **dict(sums),
            score_mse=safe('score_error_sse', 'score_elements'),
            score_centered_error_mse=safe('score_centered_error_sse', 'score_elements'),
            score_centered_relative_mse=safe('score_error_sse', 'score_reference_centered_sse'),
            score_centered_shape_relative_mse=safe('score_centered_error_sse', 'score_reference_centered_sse'),
            attention_kl_mean_per_head=safe('attention_kl_sum', 'head_count'),
            visual_mass_mae_per_head=safe('visual_mass_abs_error_sum', 'head_count'),
            output_mse=safe('output_error_sse', 'output_elements'),
            output_relative_mse=safe('output_error_sse', 'reference_output_sse'),
            k_effect_output_relative_mse=safe('k_effect_output_sse', 'reference_output_sse'),
            top1_same_fraction=safe('top1_equal_head_sum', 'head_count')))
    report = dict(status='COMPLETE', images=len(chosen), selected_ids=manifest['selected_ids'],
        scope='development local counterfactuals at identical Full queries/text; no candidate task accuracy or packed deployment claims',
        table=table, coverage=coverage,
        metric_scope='all 28 query-head outputs combined through W_O before squared error',
        phase_scope='first question final prefill position and first configured decode calls, including EOS/closure if reached')
    (arg.output/'summary.json').write_text(json.dumps(report, indent=2))
    (arg.output/'complete.json').write_text(json.dumps(dict(status='COMPLETE', images=len(chosen))))


if __name__ == '__main__':
    main()
