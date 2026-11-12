"""Compare trained and seeded MTP acceptance on identical locked Mac prompts."""
import argparse
from datetime import datetime, timedelta
import fcntl
import json
from pathlib import Path
import resource
import time

from qwen35_quality_data import digest, load, write_json
from run_dense_text_guarded import native, pressure


def prompt_items(manifest, tokenizer, completed):
    if 'conversation_source' not in manifest:
        yield from manifest['items']
        return
    with Path(manifest['conversation_source']).open() as stream:
        questions = [json.loads(line) for line in stream]
    assert len(questions) == manifest['conversation_count']
    assert len({question['question_id'] for question in questions}) == len(questions)
    for index, question in enumerate(questions):
        turns = question['turns']
        assert len(turns) == manifest['turns_per_conversation'] == 2
        prefix = f'User: {turns[0]}\nAssistant:'
        yield dict(group='mt-bench', source_id=question['question_id'], turn=1,
                   category=question['category'], input_ids=tokenizer.encode(prefix, add_special_tokens=False).ids)
        # Use the actual verified target answer, not a reference or a second independent prompt.
        answer = completed[2*index]
        assert answer['turn'] == 1 and answer['source_id'] == question['question_id']
        prefix += f" {answer['text']}\nUser: {turns[1]}\nAssistant:"
        yield dict(group='mt-bench', source_id=question['question_id'], turn=2,
                   category=question['category'], input_ids=tokenizer.encode(prefix, add_special_tokens=False).ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--prompts', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--deadline', required=True, type=datetime.fromisoformat)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prefill-last-only', action='store_true')
    args = parser.parse_args()
    path = Path(args.out)
    if path.exists() != args.resume or path.parent.resolve() != Path('runs').resolve():
        parser.error('Choose a new path under runs/, or use --resume for a partial report')
    now = datetime.now().astimezone()
    assert now + timedelta(seconds=660) < args.deadline <= now + timedelta(hours=2)
    with Path('runs/hf-mmmlu-global.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert pressure() == 1
        evaluate(args)


def load_models(checkpoint):
    import torch
    from model import Config, Model
    from mtp import MultiTokenPredictor

    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    profile = saved['identity']['profile']
    assert saved['step'] == profile['steps'] and saved['final_validation'] is not None
    assert all(digest(file) == sha for file, sha in profile['files'].items())
    assert digest('train_mtp_draft.py') == saved['identity']['runner_sha256']
    target_checkpoint = profile['target']
    config = Config(**saved['identity']['config'])
    base = torch.load(target_checkpoint, map_location='cpu', weights_only=True, mmap=True)
    assert base['config'] == vars(config)
    # PLE keeps deterministic tensors outside the checkpoint state dictionary.
    target = Model(config)
    target.load_state_dict(base['model'], assign=True)
    del base
    target.eval().requires_grad_(False)
    torch.manual_seed(profile['seed'])
    initial = MultiTokenPredictor(config).eval()
    trained = MultiTokenPredictor(config).eval()
    trained.load_state_dict(saved['draft'])
    return target, initial, trained, config, profile


def score_item(models, item, manifest, prefill_last_only, *, decoder_class=None):
    import torch
    from inference import Inference
    from speculative_mtp import GreedyMTP

    decoder_class = decoder_class or GreedyMTP
    target, initial, trained, config, _ = models
    eos = {config.eos}
    prompt = torch.tensor([item['input_ids']], dtype=torch.long)
    limit = manifest.get('context_limit', 8192)
    assert 0 < prompt.shape[1] <= limit-manifest['new_tokens']
    with torch.no_grad():
        tick = time.monotonic()
        engine = Inference(target, 'sparse', grouped_gqa=True)
        cache = {}
        current = prompt
        teacher = []
        for _ in range(manifest['new_tokens']):
            hidden = engine.hidden(current, cache)
            token = int(target.head(target.readout(hidden[:, -1:]))[0, 0].argmax())
            teacher.append(token)
            if token in eos:
                break
            current = prompt.new_tensor([[token]])
        teacher_seconds = time.monotonic()-tick
    results = {}
    for name, draft in [('initial', initial), ('trained', trained)]:
        start = time.monotonic()
        result = decoder_class(target, draft, context_limit=limit, prefill_last_only=prefill_last_only).generate(
            prompt, manifest['new_tokens'], manifest.get('draft_tokens', 4), eos)
        result['seconds'] = time.monotonic()-start
        assert result['output_ids'] == teacher
        results[name] = result
    return teacher, teacher_seconds, results


def evaluate(args):
    from tokenizers import Tokenizer

    models = load_models(args.checkpoint)
    target_checkpoint = models[-1]['target']
    manifest = load(args.prompts)
    assert manifest['tokenizer_sha256'] == digest(manifest['tokenizer'])
    assert all(digest(file) == sha for file, sha in manifest['source_hashes'].items())
    tokenizer = Tokenizer.from_file(manifest['tokenizer'])

    conversations = 'conversation_source' in manifest
    count = manifest['conversation_count'] * manifest['turns_per_conversation'] if conversations else len(manifest['items'])
    identity = dict(checkpoint_sha256=digest(args.checkpoint),
                  target_checkpoint_sha256=digest(target_checkpoint), prompts_sha256=digest(args.prompts),
                  source_hashes={name: digest(name) for name in
                                 ('evaluate_mtp_draft.py', 'speculative_mtp.py',
                                  'mtp_inference.py', 'inference.py')},
                  target_items=count, prefill_last_only=args.prefill_last_only)
    if args.resume:
        report = load(args.out)
        legacy_sha = 'ab1894e001325ee4c47384a1e72a8114c9f547a7b431a71ba5a2480cda0652da'
        if report['source_hashes']['evaluate_mtp_draft.py'] == legacy_sha:
            identity['source_hashes']['evaluate_mtp_draft.py'] = legacy_sha
        assert report['status'] == 'saved_partial' and all(report[key] == value for key, value in identity.items())
        assert 0 <= len(report['items']) < count
    else:
        report = dict(identity, items=[])
    report['status'] = 'running'
    report.pop('stop_reason', None)
    write_json(args.out, report)
    try:
        previous = len(report['items'])
        for index, item in enumerate(prompt_items(manifest, tokenizer, report['items'])):
            if index < previous:
                row = report['items'][index]
                assert all(row[key] == value for key, value in item.items())
                assert all(result['output_ids'] == row['output_ids'] for result in row['speculative'].values())
                assert tokenizer.decode(row['output_ids']) == row['text']
                continue
            level = pressure()
            if level >= 2 or datetime.now().astimezone() >= args.deadline-timedelta(seconds=600):
                report['status'] = 'saved_partial'
                report['stop_reason'] = 'memory_pressure' if level >= 2 else 'time_budget'
                break
            teacher, teacher_seconds, results = score_item(models,item,manifest,args.prefill_last_only)
            report['items'].append(dict(item, output_ids=teacher, text=tokenizer.decode(teacher),
                                        teacher_seconds=teacher_seconds, speculative=results,
                                        pressure=level, swap=native(['sysctl', 'vm.swapusage']).strip(),
                                        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            write_json(args.out, report)
        else:
            assert len(report['items']) == count
            report['status'] = 'complete_mac_mt_bench' if conversations else 'complete_mac_prompt_subset'
    except BaseException:
        report['status'] = 'failed'
        raise
    finally:
        totals = {}
        for item in report['items']:
            for name, result in item['speculative'].items():
                group = totals.setdefault(item['group'], {}).setdefault(name, dict(rounds=0, accepted=0, clipped_rounds=0))
                group['rounds'] += result['num_drafts']
                group['accepted'] += result['num_accepted_tokens']
                group['clipped_rounds'] += sum(row['mtp_selected_count'] < row['seed_position']
                                               for row in result['rounds'] if 'drafted' in row)
        for modes in totals.values():
            for stats in modes.values():
                stats['mean_acceptance_length'] = (1 + stats['accepted']/stats['rounds']
                                                   if stats['rounds'] else None)
        report['totals'] = totals
        report['overall'] = {}
        for mode in ('initial', 'trained'):
            rows = [group[mode] for group in totals.values() if mode in group]
            rounds = sum(row['rounds'] for row in rows)
            accepted = sum(row['accepted'] for row in rows)
            report['overall'][mode] = dict(rounds=rounds, accepted=accepted,
                mean_acceptance_length=1+accepted/rounds if rounds else None,
                clipped_rounds=sum(row['clipped_rounds'] for row in rows))
        write_json(args.out, report)


if __name__ == '__main__':
    main()
