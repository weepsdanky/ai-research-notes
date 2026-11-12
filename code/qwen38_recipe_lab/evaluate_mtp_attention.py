"""Measure MTP acceptance using an independent serial oracle for each attention arm."""
import argparse
from datetime import datetime, timedelta
import fcntl
import json
import os
from pathlib import Path
import resource
import time

from audit_mtp_coverage import MT_PROFILE, PROMPTS
from evaluate_mtp_draft import load_models as load_sparse_models
from evaluate_mtp_draft import prompt_items, score_item as score_sparse_item
from evaluate_mtp_public import add, empty_totals
from mtp_public_data import PROFILE as EVAL_PROFILE, profile_and_tokenizer, public_items
from qwen35_quality_data import digest, load, write_json
from run_control_quality import LOCK, memory, normal
from run_dense_text_guarded import native, pressure
from train_mtp_dense_draft import PROFILE as TRAIN_PROFILE


def prompts(job, profile, tokenizer, completed):
    if job == 'mt-bench':
        manifest = load(MT_PROFILE)
        assert manifest['tokenizer_sha256'] == profile['tokenizer_sha256']
        assert manifest['new_tokens'] == profile['new_tokens']
        assert all(digest(path) == sha for path, sha in manifest['source_hashes'].items())
        count = manifest['conversation_count'] * manifest['turns_per_conversation']
        return prompt_items(manifest, tokenizer, completed), count, digest(MT_PROFILE)
    catalog = load(PROMPTS)
    assert catalog['profile_sha256'] == digest(EVAL_PROFILE)
    assert catalog['preparer_sha256'] == digest('mtp_public_data.py')
    items, sha = public_items(profile, job, tokenizer)
    assert catalog['jobs'][job]['content_sha256'] == sha
    assert catalog['jobs'][job]['items'] == len(items)
    return iter(items), len(items), sha


def training_binding(attention='full'):
    profile = TRAIN_PROFILE if attention == 'full' else load(EVAL_PROFILE)['draft_training_profile']
    training = load(profile)
    if attention == 'full':
        assert training['stage'] == 'pretrain'
    output = Path(training['output'])
    checkpoint = output / 'last.pt'
    state_path = output / 'state.json'
    state = load(state_path)
    assert state['reason'] == 'complete' and state['step'] == state['target_steps'] == training['steps']
    assert state['final_validation'] is not None
    flags = ['rng_replayed', 'frozen_target_unchanged']
    if attention == 'full':
        flags += ['frozen_unused_indexer', 'matched_sparse_sampling_offsets']
    assert all(state[key] for key in flags)
    assert all(digest(path) == sha for path, sha in training['files'].items())
    assert state['checkpoint_sha256'] == digest(checkpoint)
    return training, checkpoint, state_path


def load_models(training, checkpoint):
    import torch

    from model import Config, Model
    from mtp import MultiTokenPredictor

    torch.set_num_threads(1)
    draft_state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    saved = torch.load(training['target'], map_location='cpu', mmap=True, weights_only=True)
    identity = draft_state['identity']
    assert identity['profile_sha256'] == digest(TRAIN_PROFILE) and identity['profile'] == training
    assert identity['runner_sha256'] == digest('train_mtp_dense_draft.py')
    assert draft_state['step'] == training['steps'] and draft_state['final_validation'] is not None
    assert saved['config'] == identity['config'] and saved['args']['stage'] == 'pretrain'
    config = Config(**saved['config'])
    target = Model(config)
    target.load_state_dict(saved['model'], assign=True)
    target.eval().requires_grad_(False)
    torch.manual_seed(training['seed'])
    initial = MultiTokenPredictor(config).eval()
    trained = MultiTokenPredictor(config).eval()
    trained.load_state_dict(draft_state['draft'])
    return target, initial, trained


def score_item(models, item, profile):
    import torch

    from inference import Inference
    from serial_mtp_verification import SerialTargetMTP

    target, initial, trained = models
    assert (target.config.eos, target.config.vocab_size) == (profile['eos'], profile['vocab_size'])
    prompt = torch.tensor([item['input_ids']], dtype=torch.long)
    limit = profile['context_limit']
    assert 0 < prompt.shape[1] <= limit - profile['new_tokens']
    engine = Inference(target, 'pretrain', grouped_gqa=True)
    current, cache, teacher = prompt, {}, []
    start = time.monotonic()
    with torch.no_grad():
        for _ in range(profile['new_tokens']):
            hidden = engine.hidden(current, cache)
            token = int(target.head(target.readout(hidden[:, -1:]))[0, 0].argmax())
            teacher.append(token)
            if token == profile['eos']:
                break
            current = prompt.new_tensor([[token]])
    teacher_seconds = time.monotonic() - start
    results = {}
    for name, draft in [('initial', initial), ('trained', trained)]:
        start = time.monotonic()
        decoder = SerialTargetMTP(target, draft, 'pretrain', context_limit=limit, prefill_last_only=True)
        result = decoder.generate(prompt, profile['new_tokens'], profile['draft_tokens'], {profile['eos']})
        result['seconds'] = time.monotonic() - start
        assert result['output_ids'] == teacher
        assert all(row['mtp_selected_count'] == row['seed_position']
                   for row in result['rounds'] if 'drafted' in row)
        results[name] = result
    return teacher, teacher_seconds, results


def evaluate(args):
    training, checkpoint, state = training_binding(args.attention)
    profile, tokenizer = profile_and_tokenizer()
    completed = []
    items, count, prompts_sha = prompts(args.job, profile, tokenizer, completed)
    trainer = 'train_mtp_dense_draft.py' if args.attention == 'full' else 'train_mtp_draft.py'
    paths = ('evaluate_mtp_attention.py', trainer, 'evaluate_mtp_public.py',
             'evaluate_mtp_draft.py', 'mtp_public_data.py', 'model.py', 'mtp.py', 'inference.py',
             'speculative_mtp.py', 'mtp_inference.py', 'serial_mtp_verification.py')
    identity = dict(job=args.job, attention=args.attention, verification='serial',
                    profile_sha256=digest(EVAL_PROFILE), prompt_content_sha256=prompts_sha,
                    draft_checkpoint_sha256=digest(checkpoint), training_state_sha256=digest(state),
                    target_checkpoint_sha256=digest(training['target']),
                    sources={path: digest(path) for path in paths})
    out = args.out
    meta = Path(f'{out}.meta.json')
    totals = empty_totals('serial')

    def record(row, item):
        add(totals, row, item, tokenizer, profile)
        if args.job == 'mt-bench':
            completed.append(row)

    def save(status):
        write_json(meta, dict(identity=identity, status=status, target=count, totals=totals))

    if args.resume or args.audit_out:
        saved = load(meta)
        assert saved['identity'] == identity
        with out.open() as stream:
            for line in stream:
                record(json.loads(line), next(items))
        if args.audit_out:
            complete = totals['items'] == count
            assert complete == (saved['status'] == 'complete') and totals == saved['totals']
            acceptance = {mode: 1 + row['accepted'] / row['rounds'] if row['rounds'] else None
                          for mode, row in totals['modes'].items()}
            write_json(args.audit_out, dict(identity=identity, complete=complete, target=count, totals=totals,
                                            acceptance=acceptance, output_sha256=digest(out)))
            return
        if saved['status'] == 'running':
            assert totals['items'] >= saved['totals']['items']
        else:
            assert saved['status'] in ('time_budget', 'memory_pressure') and totals == saved['totals']
        if totals['items'] == count:
            save('complete')
            return
    else:
        assert not out.exists() and not meta.exists()
    scorer = score_item
    if args.attention == 'full':
        models = load_models(training, checkpoint)
    else:
        from functools import partial
        from serial_mtp_verification import SerialTargetMTP

        models = load_sparse_models(checkpoint)
        assert models[-1] == training
        scorer = partial(score_sparse_item, prefill_last_only=True, decoder_class=SerialTargetMTP)
    stop = min(args.deadline - timedelta(seconds=600),
               datetime.now().astimezone() + timedelta(seconds=args.seconds))
    status = 'complete'
    with out.open('a' if args.resume else 'x') as stream:
        save('running')
        for item in items:
            if pressure() != 1:
                status = 'memory_pressure'
                break
            if datetime.now().astimezone() >= stop:
                status = 'time_budget'
                break
            teacher, seconds, results = scorer(models, item, profile)
            row = dict(item, output_ids=teacher, text=tokenizer.decode(teacher), teacher_seconds=seconds,
                       speculative=results, pressure=1, swap=native(['sysctl', 'vm.swapusage']).strip(),
                       process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            record(row, item)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            save('running')
    assert (totals['items'] == count) == (status == 'complete')
    save(status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True, choices=('mt-bench', *load(EVAL_PROFILE)['jobs']))
    parser.add_argument('--attention', choices=('qsa', 'full'), default='full')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--deadline', type=datetime.fromisoformat)
    parser.add_argument('--seconds', type=int, default=900)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--resume', action='store_true')
    mode.add_argument('--audit-out', type=Path)
    args = parser.parse_args()
    if args.attention == 'qsa' and args.job != 'mt-bench':
        parser.error('Use the existing public evaluator for non-conversation QSA jobs')
    runs = Path('runs').resolve()
    assert args.out.parent.resolve() == runs
    if args.audit_out:
        assert args.audit_out.parent.resolve() == runs and not args.audit_out.exists()
    else:
        now = datetime.now().astimezone()
        assert args.deadline and now + timedelta(seconds=660) < args.deadline <= now + timedelta(hours=2)
        assert 30 <= args.seconds <= 1100
    assert not any(key in os.environ for key in ('PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'PYTORCH_ENABLE_MPS_FALLBACK'))
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert pressure() == 1 if args.audit_out else normal(memory())
        evaluate(args)


if __name__ == '__main__':
    main()
