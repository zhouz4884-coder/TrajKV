"""Bounded fixed-Full-history diagnostic, not task accuracy or E2E timing.

Each grouped QA is serialized independently. Full first supplies <=8 free
teacher IDs, then Full and every candidate consume exactly that entire target
history. A margins hook observes the actual logits before each forced ID.
Optional canonical-GT passes have their own independent history and label.
"""
import argparse
import copy
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from contextlib import AbstractContextManager
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/experiments'))
import torch

METRICS = ('KL_teacher_to_student_mean', 'target_NLL_delta_mean',
           'teacher_winner_signed_margin_delta_mean', 'fixed_teacher_top2_delta_mean',
           'fixed_teacher_top2_delta_RMS', 'teacher_winner_flip_rate')


def source_identity(row):
    return row.get('source_id') or row.get('source_image_id') or row['image_id']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def tensor_hash(value):
    return hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def independent_qa(row, question_index):
    """Discard prior text/answers, retaining only image items and this QA."""
    question = row['questions'][question_index]
    result = copy.deepcopy(row)
    result['id'] = result['session_id'] = row['id'] + '__indq' + str(question_index + 1)
    result['questions'], result['turns'] = [question], 1
    images = [copy.deepcopy(item) for item in row.get('content', []) if item.get('type') == 'image']
    if not images:
        images = [dict(type='image') for _ in row.get('remote_images', row.get('images', []))]
    if not images:
        raise ValueError('Independent QA requires an image')
    result['content'] = images + [dict(type='text', text=question)]
    result['parent_group_id'] = row['id']
    result['independent_question_index'] = question_index + 1
    result['first_turn_scored'] = True
    return result


def select_sources(data, count, balance=True):
    if not 1 <= count <= 16:
        raise ValueError('Use one to sixteen source images')
    if balance:
        groups = defaultdict(list)
        for row in data:
            groups[row.get('dataset', 'unknown')].append(row)
        chosen = []
        index = 0
        while len(chosen) < count:
            added = False
            for name in sorted(groups):
                if index < len(groups[name]) and len(chosen) < count:
                    chosen.append(groups[name][index]); added = True
            if not added:
                break
            index += 1
    else:
        chosen = data[:count]
    identities = [source_identity(row) for row in chosen]
    if len(chosen) != count or len(set(identities)) != count:
        raise ValueError('Missing or repeated original source images')
    return chosen


def canonical_reference(refs, group_id, question_index):
    entry = refs[group_id]
    if entry.get('references'):
        found = [item for item in entry['references'] if int(item['turn']) == question_index + 1]
        if len(found) != 1:
            raise ValueError('Missing/duplicate canonical GT question reference')
        value = found[0].get('answer', found[0].get('answers'))
    else:
        if question_index != 0:
            raise ValueError('Ungrouped references cannot supply a later QA')
        value = entry.get('answer', entry.get('answers'))
    if isinstance(value, list):
        value = value[0] if value else ''
    if not isinstance(value, str):
        raise ValueError('Canonical GT must be the declared first reference string')
    return value


class MarginsHook(AbstractContextManager):
    def __init__(self, executor, consumer):
        self.executor, self.consumer = executor, consumer
        self.original = None

    def __enter__(self):
        if self.consumer is None:
            return self
        self.original = self.executor.margins

        def observe(logits):
            self.consumer(logits)
            return self.original(logits)

        self.executor.margins = observe
        return self

    def __exit__(self, *exception):
        if self.original is not None:
            self.executor.margins = self.original


class PhysicalForwards(AbstractContextManager):
    """Count outer VLM invocations, including prefill/replay/closure, not layers."""
    def __init__(self, model):
        self.model, self.calls, self.handle = model, [], None

    def __enter__(self):
        def observe(module, args, kwargs):
            ids = kwargs.get('input_ids')
            if ids is None and args and torch.is_tensor(args[0]):
                ids = args[0]
            self.calls.append(dict(input_tokens=int(ids.shape[-1]) if ids is not None else None,
                pixel_values_present=kwargs.get('pixel_values') is not None,
                past_key_values_present=kwargs.get('past_key_values') is not None))
        self.handle = self.model.register_forward_pre_hook(observe, with_kwargs=True)
        return self

    def __exit__(self, *exception):
        self.handle.remove()


def logit_vector(logits):
    if logits.ndim == 2 and logits.shape[0] == 1:
        logits = logits[0]
    if logits.ndim != 1:
        raise ValueError('Expected one actual next-token vocabulary vector')
    value = logits.detach().float().cpu().contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError('Nonfinite next-token logits')
    return value


def reference_position(logits, token):
    value = logit_vector(logits)
    winner = int(value.argmax())
    other = value.clone(); other[winner] = -torch.inf
    competitor = int(other.argmax())
    lp = value.double().log_softmax(-1)
    return dict(value=value, logp=lp, probability=lp.exp(), winner=winner, competitor=competitor,
                margin=float(value[winner] - value[competitor]), target_token=int(token))


def compare_position(reference, logits):
    value = logit_vector(logits)
    if value.shape != reference['value'].shape:
        raise ValueError('Teacher/student vocabulary differs')
    winner, competitor, target = reference['winner'], reference['competitor'], reference['target_token']
    other = value.clone(); other[winner] = -torch.inf
    best_other = int(other.argmax())
    signed = float(value[winner] - value[best_other])
    contrast = float(value[winner] - value[competitor])
    lp = value.double().log_softmax(-1)
    kl = float((reference['probability'] * (reference['logp'] - lp)).sum())
    return dict(teacher_winner_id=winner, teacher_fixed_competitor_id=competitor,
        student_argmax_id=int(value.argmax()), student_best_other_id=best_other,
        teacher_reference_margin=reference['margin'], teacher_winner_signed_margin_student=signed,
        teacher_winner_signed_margin_delta=signed - reference['margin'],
        fixed_teacher_top2_contrast_student=contrast, fixed_teacher_top2_delta=contrast - reference['margin'],
        KL_teacher_to_student=kl, target_token_id=target,
        target_NLL_teacher=float(-reference['logp'][target]), target_NLL_student=float(-lp[target]),
        target_NLL_delta=float(reference['logp'][target] - lp[target]),
        teacher_winner_flipped=int(value.argmax()) != winner)


def question_summary(steps):
    if not steps:
        raise ValueError('Empty diagnostic target')
    mean = lambda key: statistics.mean(row[key] for row in steps)
    return dict(KL_teacher_to_student_mean=mean('KL_teacher_to_student'),
        target_NLL_delta_mean=mean('target_NLL_delta'),
        teacher_winner_signed_margin_delta_mean=mean('teacher_winner_signed_margin_delta'),
        fixed_teacher_top2_delta_mean=mean('fixed_teacher_top2_delta'),
        fixed_teacher_top2_delta_RMS=statistics.mean(row['fixed_teacher_top2_delta'] ** 2 for row in steps) ** .5,
        teacher_winner_flip_rate=statistics.mean(row['teacher_winner_flipped'] for row in steps))


def aggregate(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[(row['target_kind'], row['variant'], row['source_id'], row['dataset'])].append(row)
    per_source = []
    for (target, variant, source, dataset), qs in sorted(grouped.items()):
        per_source.append(dict(target_kind=target, variant=variant, source_id=source, dataset=dataset,
            questions=len(qs), metrics={key: statistics.mean(q['summary'][key] for q in qs) for key in METRICS}))
    macro = []
    pairs = sorted({(row['target_kind'], row['variant']) for row in per_source})
    for target, variant in pairs:
        chosen = [r for r in per_source if (r['target_kind'], r['variant']) == (target, variant)]
        macro.append(dict(target_kind=target, variant=variant, sources=len(chosen),
            source_macro={key: statistics.mean(r['metrics'][key] for r in chosen) for key in METRICS}))
    return per_source, macro


def paired_sources(per_source):
    result = []
    targets = sorted({row['target_kind'] for row in per_source})
    for target in targets:
        names = sorted({row['variant'] for row in per_source if row['target_kind'] == target})
        lookup = {(row['variant'], row['source_id']): row for row in per_source if row['target_kind'] == target}
        for index, left in enumerate(names):
            for right in names[index + 1:]:
                ls = {source for name, source in lookup if name == left}
                rs = {source for name, source in lookup if name == right}
                if ls != rs:
                    raise ValueError('Candidate source coverage is not paired')
                for source in sorted(ls):
                    result.append(dict(target_kind=target, left=left, right=right, source_id=source,
                        difference_left_minus_right={key: lookup[left, source]['metrics'][key] - lookup[right, source]['metrics'][key] for key in METRICS}))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--references', type=Path)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    variants = config['variants']
    if not 1 <= len(variants) <= 4 or len({v['name'] for v in variants}) != len(variants):
        raise ValueError('Declare one to four distinct diagnostic candidates')
    a = SimpleNamespace(**config['arguments'])
    sources = select_sources(json.loads(Path(a.samples).read_text(encoding='utf-8-sig')),
        int(config.get('count', 8)), bool(config.get('balance_datasets', True)))
    max_questions = int(config.get('questions_per_source', 2))
    horizon = int(config.get('target_tokens', 8))
    if max_questions not in (1, 2) or not 1 <= horizon <= 8:
        raise ValueError('Use at most two independent QAs/source and eight target tokens')
    caps = {int(v.get('cap', getattr(a, 'max_patches', 8))) for v in variants}
    if len(caps) != 1:
        raise ValueError('Teacher and candidates must use one common image cap')
    cases = []
    for row in sources:
        if any(key in row for key in ('answer', 'answers', 'references', 'ground_truth')):
            raise ValueError('Inference input rows must be answer-free')
        for qi in range(min(max_questions, len(row['questions']))):
            cases.append((row, qi, independent_qa(row, qi)))
    if not cases:
        raise ValueError('No independent QAs selected')
    reference_path = args.references or (Path(config['diagnostic_references']) if config.get('diagnostic_references') else None)
    refs = None
    if reference_path is not None:
        raw = json.loads(reference_path.read_text(encoding='utf-8-sig'))
        refs = {row['id']: row for row in raw} if isinstance(raw, list) else raw
        for row, qi, case in cases:
            canonical_reference(refs, row['id'], qi)
    plan = dict(sources=len(sources), independent_QAs=len(cases), candidates=len(variants),
        teacher_target_horizon=horizon, teacher_passes_per_QA=2,
        optional_GT_separate_passes_per_QA=1 + len(variants) if refs is not None else 0,
        planned_outer_sessions=len(cases) * (2 + len(variants) + (1 + len(variants) if refs is not None else 0)),
        primary_quality_eligible=False, end_to_end_timing=False)
    if args.plan_only:
        print(json.dumps(plan), flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('Actual fixed-prefix VLM diagnostic requires CUDA')
    args.output.mkdir(parents=True, exist_ok=False)
    import round8_runner  # Must register before importing the factory callable.
    from factory import bank_class
    from bank import install_legacy
    install_legacy(a.legacy_scripts)
    import session_boundary as executor
    import run_p175_modern_replication as oldbase
    from transformers import AutoProcessor, AutoModelForImageTextToText
    for key, value in dict(turns=1, max_tokens=horizon, description_tokens=horizon, max_patches=next(iter(caps)),
        margin=0., max_replays=0, save_states=False, read_budget_bytes=None, gain_table_data=None).items():
        setattr(a, key, value)
    if getattr(a, 'native_question_history', False):
        raise ValueError('Complete question must be replayed under each codec')
    torch.manual_seed(1212); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    external = {v[key]: sha(v[key]) for v in variants for key in
        ('direction_book_path', 'encoder_book_path', 'metadata_reference_book_path', 'metadata_factor_path', 'value_joint_book_path') if v.get(key)}
    names = ('round8_continuous.py', 'round8_runner.py', 'round8_bank.py', 'round8_codec.py', 'round8_attention.py',
             'round8_joint_gain.py', 'session_boundary.py', 'factory.py', 'masked_bank.py')
    image_inputs = []
    for row in sources:
        hashes = [sha(path) for path in row.get('remote_images', row['images'])]
        if row.get('hashes') and hashes != row['hashes']:
            raise ValueError('Actual image bytes differ from source metadata')
        image_inputs.append(dict(source_id=source_identity(row), image_sha256=hashes))
    manifest = dict(schema='trajkv_round8_fixed_Full_prefix_diagnostic_v1', config=config, plan=plan,
        config_sha256=sha(args.config), samples_sha256=sha(a.samples),
        references_sha256=sha(reference_path) if reference_path else None,
        selected_source_ids=[source_identity(row) for row in sources], image_inputs=image_inputs,
        selected_QAs=[dict(case_id=case['id'], parent_group_id=row['id'], question_index=qi + 1,
            question_sha256=json_hash(row['questions'][qi])) for row, qi, case in cases],
        external_codec_inputs=external, sources={name: sha(Path(__file__).with_name(name)) for name in names},
        model_config_sha256=sha(Path(a.model) / 'config.json'), labels_used_only_for_optional_GT_pass=refs is not None,
        primary_quality_eligible=False, task_accuracy_eligible=False, end_to_end_timing=False,
        interpretation='All target positions included, no common-prefix/survival filtering. Same token history does not mean same compressed state. GT and Full-self targets are separate.',
        forward_counter_scope='Actual outer VLM model invocations, including native prefill, question replay, forced token and canonical closure calls; not layers/kernels/equal-cost units.')
    write_json(args.output / 'manifest.json', manifest)
    pass_rows, records, controls, ref_rows = [], [], [], []
    original_bank = executor.ProgressiveBank
    full = dict(name='__Full_reference__', method='full', cap=next(iter(caps)), selector='none', store_selector_meta=False)
    try:
        processor = AutoProcessor.from_pretrained(a.model, local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(a.model, local_files_only=True,
            dtype=torch.float16, device_map='cuda:0', attn_implementation='sdpa').eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        stop_ids = oldbase.stop_ids(model, processor)
        canonical_eos = processor.tokenizer.eos_token_id

        def run_pass(case, variant, role, target, consumer, stream):
            cls = bank_class(variant['method']); previous_settings = cls.settings
            cls.settings = variant; executor.ProgressiveBank = cls
            counter = PhysicalForwards(model); result = None; failed = None
            try:
                with counter, MarginsHook(executor, consumer):
                    result, _ = executor.run_session(model, processor, case, a, variant['method'], 'base',
                        forced=None if target is None else [dict(token_ids=target)], keep_reference=False)
                if target is not None and result['turns'][0]['token_ids'] != target:
                    raise RuntimeError('Forced target history changed')
                return result
            except BaseException as error:
                failed = type(error).__name__ + ': ' + str(error)
                raise
            finally:
                cls.settings = previous_settings; executor.ProgressiveBank = original_bank
                item = dict(pass_id=len(pass_rows), case_id=case['id'], variant=variant['name'], method=variant['method'],
                    role=role, status='FAILED' if failed else 'COMPLETE', exception=failed,
                    physical_model_forwards=len(counter.calls), forward_calls=counter.calls,
                    forced_target_tokens=len(target) if target is not None else None,
                    generated_tokens=len(result['turns'][0]['token_ids']) if result is not None else None,
                    canonical_closure_after_target=result['turns'][0]['forced_chat_closure'] if result is not None else None,
                    first_question_suffix_replay_tokens=result.get('first_question_suffix_replay_tokens') if result else None,
                    primary_quality_eligible=False, end_to_end_timing=False)
                pass_rows.append(item); stream.write(json.dumps(item) + '\n'); stream.flush()

        with torch.inference_mode(), (args.output / 'passes.jsonl').open('x', encoding='utf-8') as pass_stream, \
             (args.output / 'records.jsonl').open('x', encoding='utf-8') as output_stream:
            for source_row, qi, case in cases:
                teacher = run_pass(case, full, 'Full_free_teacher_target', None, None, pass_stream)
                target = list(teacher['turns'][0]['token_ids'])
                if not target or len(target) > horizon:
                    raise RuntimeError('Full teacher target empty or beyond declared horizon')
                target_sets = [('Full_self', target, dict(natural_stop=target[-1] in stop_ids,
                    horizon_truncated=len(target) == horizon and target[-1] not in stop_ids))]
                if refs is not None:
                    text = canonical_reference(refs, source_row['id'], qi)
                    ids = processor.tokenizer.encode(text, add_special_tokens=False) + [canonical_eos]
                    target_sets.append(('canonical_GT', ids[:horizon], dict(natural_stop=len(ids) <= horizon,
                        horizon_truncated=len(ids) > horizon, canonical_reference_text_sha256=json_hash(text))))
                for target_kind, ids, flags in target_sets:
                    reference = []
                    def collect(logits):
                        step = len(reference)
                        if step >= len(ids):
                            raise RuntimeError('Unexpected logits beyond the target horizon')
                        reference.append(reference_position(logits, ids[step]))
                    baseline = run_pass(case, full, target_kind + '_forced_reference', ids, collect, pass_stream)
                    if len(reference) != len(ids):
                        raise RuntimeError('Reference did not expose every target position')
                    control = dict(case_id=case['id'], target_kind=target_kind, target_token_ids=ids,
                        target_sha256=json_hash(ids), target_horizon=len(ids), canonical_eos=canonical_eos,
                        canonical_closure_after_horizon=baseline['turns'][0]['forced_chat_closure'],
                        closure_not_scored=True, **flags)
                    controls.append(control)
                    ref_rows.append(dict(**control, full_reference_logit_sha256=[tensor_hash(r['value']) for r in reference],
                        teacher_winner_ids=[r['winner'] for r in reference], teacher_competitor_ids=[r['competitor'] for r in reference],
                        reference_winner_equals_target=[r['winner'] == token for r, token in zip(reference, ids)]))
                    if config.get('save_reference_logits', False):
                        folder = args.output / 'reference_logits'; folder.mkdir(exist_ok=True)
                        torch.save(dict(case_id=case['id'], target_kind=target_kind, token_ids=ids,
                            logits=[r['value'] for r in reference]), folder / (json_hash([case['id'], target_kind]) + '.pt'))
                    for variant in variants:
                        steps = []
                        def compare(logits):
                            index = len(steps)
                            if index >= len(reference):
                                raise RuntimeError('Candidate exposed extra logits positions')
                            row = compare_position(reference[index], logits)
                            row.update(step=index, forced_history_token_ids=ids[:index],
                                target_is_stop=ids[index] in stop_ids, target_is_canonical_eos=ids[index] == canonical_eos)
                            steps.append(row)
                        result = run_pass(case, variant, target_kind + '_forced_candidate', ids, compare, pass_stream)
                        if len(steps) != len(ids) or result['turns'][0]['logical_end'] != baseline['turns'][0]['logical_end'] or result['turns'][0]['forced_chat_closure'] != baseline['turns'][0]['forced_chat_closure']:
                            raise RuntimeError('Candidate history/horizon/logical positions differ')
                        record = dict(schema=manifest['schema'], protocol='independent_QA_fixed_teacher_history_diagnostic',
                            source_id=source_identity(source_row),
                            dataset=source_row.get('dataset', 'unknown'), parent_group_id=source_row['id'], case_id=case['id'],
                            independent_question_index=qi + 1, variant=variant['name'], method=variant['method'],
                            target_kind=target_kind, target_control_sha256=json_hash(control), target_horizon=len(ids),
                            steps=steps, summary=question_summary(steps), owned_initial=result['owned_initial'],
                            primary_quality_eligible=False, task_accuracy_eligible=False, end_to_end_timing=False,
                            no_task_metric=True, no_survivorship_filter=True)
                        records.append(record); output_stream.write(json.dumps(record, allow_nan=False) + '\n'); output_stream.flush()
                print(json.dumps(dict(done_case=case['id'], completed_candidate_diagnostics=len(records),
                    physical_model_forwards=sum(r['physical_model_forwards'] for r in pass_rows))), flush=True)
        if len(pass_rows) != plan['planned_outer_sessions']:
            raise RuntimeError('Physical session-pass plan did not reconcile')
        per_source, macro = aggregate(records)
        write_json(args.output / 'per_source_paired.json', per_source)
        write_json(args.output / 'candidate_paired_source_differences.json', paired_sources(per_source))
        write_json(args.output / 'summary.json', dict(status='COMPLETE', source_macro=macro,
            weighting='equal target positions within QA, equal QAs within original source, equal original sources; target kinds remain separate',
            source_count=len(sources), independent_QAs=len(cases),
            physical_model_forwards=sum(r['physical_model_forwards'] for r in pass_rows),
            physical_passes_by_role={role: dict(passes=sum(r['role'] == role for r in pass_rows),
                model_forwards=sum(r['physical_model_forwards'] for r in pass_rows if r['role'] == role)) for role in sorted({r['role'] for r in pass_rows})},
            primary_quality_eligible=False, task_accuracy_eligible=False, end_to_end_timing=False, no_task_metric=True))
        write_json(args.output / 'forced_controls.json', controls)
        with (args.output / 'teacher_references.jsonl').open('x', encoding='utf-8') as stream:
            for row in ref_rows:
                stream.write(json.dumps(row) + '\n')
        for path, expected in external.items():
            if sha(path) != expected:
                raise RuntimeError('Codec artifact changed during diagnostics')
        for name, expected in manifest['sources'].items():
            if sha(Path(__file__).with_name(name)) != expected:
                raise RuntimeError('Diagnostic source changed during execution')
        if sha(args.config) != manifest['config_sha256'] or sha(a.samples) != manifest['samples_sha256'] or sha(Path(a.model) / 'config.json') != manifest['model_config_sha256']:
            raise RuntimeError('Diagnostic config/data/model identity changed')
        if reference_path and sha(reference_path) != manifest['references_sha256']:
            raise RuntimeError('Canonical GT reference input changed')
        artifact_hashes = {p.relative_to(args.output).as_posix(): sha(p) for p in sorted(args.output.rglob('*')) if p.is_file()}
        write_json(args.output / 'complete.json', dict(status='COMPLETE', schema=manifest['schema'], records=len(records),
            physical_outer_sessions=len(pass_rows), physical_model_forwards=sum(r['physical_model_forwards'] for r in pass_rows),
            primary_quality_eligible=False, end_to_end_timing=False, artifact_sha256=artifact_hashes))
        write_json(args.output / 'receipt.json', dict(status='COMPLETE', artifact_sha256=artifact_hashes,
            complete_sha256=sha(args.output / 'complete.json')))
    except BaseException as error:
        write_json(args.output / 'failure.json', dict(status='FAILED_NEW_RUN_REQUIRED', exception_type=type(error).__name__,
            message=str(error), completed_records=len(records), physical_outer_sessions=len(pass_rows),
            physical_model_forwards=sum(r['physical_model_forwards'] for r in pass_rows), primary_quality_eligible=False))
        raise
    finally:
        executor.ProgressiveBank = original_bank


if __name__ == '__main__':
    main()
