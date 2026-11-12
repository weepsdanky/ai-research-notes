"""Evaluate completed Mac control arms on full locked public dataset jobs."""
import argparse
from collections import OrderedDict, deque
from datetime import datetime, timedelta
import fcntl
import json
import math
import os
from pathlib import Path
import re
import resource
import time

from qwen35_quality_data import (adapter_for, batches, decoded, digest, encoded_hash,
                                group_key, load, prepare_jobs, runtime_versions, write_json)
from run_dense_text_guarded import native, pressure


def check_outputs(path, rows, adapter, tokenizer, stops, job, config, choices=()):
    """Replay source alignment and scoring over every durable prediction."""
    count = scored = correct = 0
    tail = deque(maxlen=job['batch_size'])
    subjects = {}
    if path.exists():
        with path.open() as stream:
            for count, line in enumerate(stream, 1):
                task, index, _, gold, ids = rows[count-1]
                item = json.loads(line)
                assert (item['task'],item['index'],item['gold'],item['input_ids']) == (task.name,index,gold,ids)
                tokens = item['output_ids']
                assert 0 < len(tokens) <= job['tokens'] and all(type(t) is int and 0 <= t < config['vocab_size'] for t in tokens)
                reason = item['stop']
                if choices:
                    scores = item['choice_logits']
                    assert reason == 'choice_argmax' and len(scores) == len(choices)
                    assert all(math.isfinite(score) for score in scores)
                    assert tokens == [choices[max(range(len(scores)),key=scores.__getitem__)]]
                elif reason == 'eos':
                    assert tokens[-1] == config['eos'] and config['eos'] not in tokens[:-1]
                else:
                    assert config['eos'] not in tokens
                    if reason == 'stop_string':
                        assert any(stop in tokenizer.decode(tokens[-64:], skip_special_tokens=False) for stop in stops[task.name])
                    else:
                        assert reason == 'max_new_tokens' and len(tokens) == job['tokens']
                raw, prediction = decoded(adapter, tokenizer, tokens, stops[task.name])
                assert raw == item['raw_prediction'] and prediction == item['prediction']
                result = adapter.correct(prediction, gold, task.kind)
                assert result == item['correct']
                scored += result is not None
                correct += result is True
                if choices:
                    group = subjects.setdefault(group_key(task,index),dict(items=0,correct=0))
                    group['items'] += 1
                    group['correct'] += result is True
                tail.append((count-1, tokens))
    return dict(items=count, scored=scored, correct=correct), dict(tail), subjects


# The Table 7 decoder and row scorer are unchanged from this bounded evaluator.
PLACEMENT_V1_SHA256 = 'e685d12bdf478fa3d48923bb2efca7e28a67905ca7a828eeb3c5d3a9b1fbb64c'
NGRAM_V2_SHA256 = 'e948e5fe079ddf614223ca0bd1cc6372562b51c04eac59e9f7a1a69e4628334f'


def training_input(args):
    if getattr(args, 'experiment', None):
        from control_eval_binding import training_input as control_input
        return control_input(args.experiment, args.arm, args.training_audit)
    audit = load(args.training_audit)
    if args.capacity_family:
        from train_ngram_capacity import FAMILIES, OUTPUT, PROFILE, model_and_identity

        profile = load(PROFILE)
        capacity = load(profile['capacity_profile'])
        profile['config'] = capacity['config']
        entry, = [entry for entry in capacity[FAMILIES[args.capacity_family]]
                  if args.arm == f"s{entry['paper_scale_label']}"]
        assert audit['family'] == args.capacity_family
        assert audit['profile_sha256'] == digest(PROFILE)
        assert audit['runner_sha256'] == digest('train_ngram_capacity.py')
        checkpoint = OUTPUT / args.capacity_family / args.arm / 'last.pt'

        def construct():
            return model_and_identity(profile, capacity, args.capacity_family, args.arm, entry)
    else:
        from train_ngram_placement import checked_model, identity as training_identity

        profile_path = 'profiles/ngram_table7_mac.json'
        profile = load(profile_path)
        assert args.arm in profile['arms']
        assert audit['profile_sha256'] == digest(profile_path)
        assert audit['runner_sha256'] == digest('train_ngram_placement.py')
        checkpoint = Path(profile['output']) / args.arm / 'last.pt'

        def construct():
            model = checked_model(profile, args.arm)
            return model, training_identity(profile_path, model)

    assert all(digest(path) == sha for path, sha in profile['source_hashes'].items())
    arm = audit['arms'][args.arm]
    assert arm['complete'] and arm['step'] == profile['steps'] and arm['data_rng_replayed']
    assert arm['frozen_indexer_verified'] and arm['checkpoint_sha256'] == digest(checkpoint)
    return profile, arm, checkpoint, construct


def evaluate(args, requested):
    from tokenizers import Tokenizer

    profile, arm, checkpoint, construct = training_input(args)
    experiment = getattr(args, 'experiment', None)
    constrained = args.job in ('ceval','cmmlu')
    protocol_path = 'profiles/chinese_eval_mac.json' if constrained else 'profiles/qwen35_quality_protocol.json'
    protocol = load(protocol_path)
    job, = [item for item in protocol['jobs'] if item['name'] == args.job]
    if constrained:
        import eval_chinese_tasks as adapter
        tasks = [adapter.load_named(args.job)]
        assert protocol['source_lock_sha256'] == digest('chinese_eval_sources.lock.json')
        assert len(set(tasks[0].prefix_groups)) == job['subjects']
    else:
        adapter, tasks = adapter_for(job)
    tokenizer_path = '.cache/references/tokenizer.json'
    tokenizer_sha = digest(tokenizer_path)
    if constrained:
        assert tokenizer_sha == protocol['tokenizer_sha256']
    assert all(load(Path(path)/'meta.json')['tokenizer_sha256'] == tokenizer_sha for path in profile['data'])
    tokenizer = Tokenizer.from_file(tokenizer_path)
    choices = []
    if constrained:
        for label in 'ABCD':
            token, = tokenizer.encode(label,add_special_tokens=False).ids
            assert tokenizer.decode([token]) == label
            choices.append(token)
    rows, content, prefixes = prepare_jobs(tasks, tokenizer, job)
    if constrained:
        assert max(len(row[4]) for row in rows) == job['max_prompt_tokens']
        rows.sort(key=lambda row: (group_key(row[0],row[1]),len(row[4]),row[1]))
    stops = {task.name: adapter.stopping(task.name) if hasattr(adapter, 'stopping') else [] for task in tasks}
    files = ('evaluate_ngram_placement.py', 'qwen35_quality_data.py', 'evaluate_batched.py',
             'inference.py', 'prefix_inference.py', 'bounded_dense_inference.py', 'prefill_inference.py',
             'reference.py', '.cache/references/modeling_qwen4_exp.py', adapter.__file__, 'eval_tasks.py')
    if experiment:
        files += ('control_eval_binding.py',)
        if experiment == 'table5':
            files += ('residual_inference.py', 'residual_controls.py', 'mhc_control.py')
    identity = dict(arm=args.arm, job=args.job, job_spec=job, checkpoint_sha256=arm['checkpoint_sha256'],
                    training_audit_sha256=digest(args.training_audit), protocol_sha256=digest(protocol_path),
                    tokenizer_sha256=tokenizer_sha, content_sha256=content, prefix_sha256=encoded_hash(prefixes),
                    order_sha256=encoded_hash([(task.name,index) for task,index,*_ in rows]),
                    decoding='choice_argmax' if constrained else 'greedy', choice_token_ids=choices,
                    stop_strings=stops, eos=profile['config']['eos'], training_context=profile['seq'],
                    backend='bounded dense grouped GQA and public GDN chunk prefill',
                    sources={name: digest(name) for name in files}, versions=runtime_versions(),
                    scope='full public dataset job, Mac CPU greedy dense pretraining checkpoint')
    if experiment or args.capacity_family:
        identity['family'] = experiment or args.capacity_family
    out = args.out
    meta = Path(str(out)+'.meta.json')
    previous = {}
    if args.audit or args.resume:
        previous_identity = load(meta)['identity']
        evaluator = 'evaluate_ngram_placement.py'
        old_source = previous_identity['sources'][evaluator]
        if not experiment and (old_source == NGRAM_V2_SHA256 or
                               (not args.capacity_family and old_source == PLACEMENT_V1_SHA256)):
            identity['sources'][evaluator] = old_source
        assert previous_identity == identity
        counts, previous, subjects = check_outputs(out, rows, adapter, tokenizer, stops, job, profile['config'], choices)
        if args.audit:
            complete = counts['items']==len(rows)
            write_json(args.audit, dict(identity=identity, output_sha256=digest(out), complete=complete,
                                        target=len(rows), subjects=subjects,
                                        subject_macro=(sum(s['correct']/s['items'] for s in subjects.values())/len(subjects)
                                                       if complete and subjects else None), **counts))
            return
        assert counts['items'] < len(rows)
    else:
        assert not meta.exists()
        out.touch(exist_ok=False)
        counts = dict(items=0, scored=0, correct=0)
        write_json(meta, dict(identity=identity, status='loading', target=len(rows), **counts))
    import torch

    from bounded_dense_inference import BoundedDenseInference
    from prefix_inference import PrefixInference

    torch.set_num_threads(1)
    model, training_identity = construct()
    saved = torch.load(checkpoint, weights_only=True, map_location='cpu', mmap=True)
    assert saved['identity'] == training_identity
    step = len(saved['history']) if experiment == 'table10' else saved['step']
    assert step == profile['steps'] and saved['final_validation'] == arm['final_validation']
    model.load_state_dict(saved['model'], assign=True)
    del saved
    model.eval()
    if experiment == 'table5':
        from residual_inference import ResidualInference

        runner = ResidualInference(model, prefill_chunk=job['prefill_chunk'])
    else:
        runner = BoundedDenseInference(model, prefill_chunk=job['prefill_chunk'])
    caches = OrderedDict()
    status = 'complete'
    cutoff = args.deadline - timedelta(seconds=600)
    tick = time.monotonic()
    first = counts['items']
    position = 0

    def progress(reason):
        write_json(meta, dict(identity=identity, status=reason, target=len(rows), **counts,
                              process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                              swap=native(['sysctl','vm.swapusage']).strip()))

    progress('running')
    with out.open('a') as stream, torch.no_grad():
        for group in batches(rows, job['batch_size']):
            next_position = position+len(group)
            if next_position <= first:
                position = next_position
                continue
            level = pressure()
            if requested or level >= 2:
                status = 'interrupted_signal' if requested else 'memory_pressure'
                break
            if datetime.now().astimezone() >= cutoff or time.monotonic()-tick >= args.seconds:
                status = 'time_budget'
                break
            batch_start = time.perf_counter()
            key = group_key(group[0][0],group[0][1])
            prefill = runner
            if prefixes[key] and all(group_key(task,index)==key for task,index,*_ in group):
                prefill = caches.pop(key,None)
                if prefill is None:
                    prefill = PrefixInference(runner,prefixes[key])
                caches[key] = prefill
                if len(caches)>4:
                    caches.popitem(last=False)
            cache = {}
            logits = prefill(torch.tensor([row[4] for row in group]),cache)
            generated = [[] for row in group]
            reasons = ['max_new_tokens']*len(group)
            done = [False]*len(group)
            choice_scores = None
            if constrained:
                choice_scores = logits[:,choices]
                generated = [[choices[i]] for i in choice_scores.argmax(-1).tolist()]
                reasons = ['choice_argmax']*len(group)
            for step in range(0 if constrained else job['tokens']):
                selected = logits.argmax(-1)
                for index, token in enumerate(selected.tolist()):
                    if done[index]:
                        continue
                    generated[index].append(token)
                    if token == model.config.eos:
                        reasons[index], done[index] = 'eos', True
                    elif stops[group[index][0].name] and any(
                        stop in tokenizer.decode(generated[index][-64:], skip_special_tokens=False)
                        for stop in stops[group[index][0].name]):
                        reasons[index], done[index] = 'stop_string', True
                if all(done) or step+1 == job['tokens']:
                    break
                logits = runner(selected[:,None],cache)
            batch_seconds = time.perf_counter()-batch_start
            for index, (task, item_index, _, gold, ids) in enumerate(group):
                if position+index < first:
                    assert generated[index] == previous[position+index], 'Partial-batch replay differs'
                    continue
                raw, prediction = decoded(adapter,tokenizer,generated[index],stops[task.name])
                result = adapter.correct(prediction,gold,task.kind)
                row = dict(task=task.name,index=item_index,gold=gold,input_ids=ids,
                           output_ids=generated[index],raw_prediction=raw,prediction=prediction,
                           correct=result,stop=reasons[index],batch_seconds=batch_seconds,batch_items=len(group))
                if constrained:
                    row['choice_logits'] = choice_scores[index].tolist()
                stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                counts['items'] += 1
                counts['scored'] += result is not None
                counts['correct'] += result is True
            position = next_position
            stream.flush()
            os.fsync(stream.fileno())
            progress('running')
    assert (counts['items']==len(rows)) == (status=='complete')
    progress(status)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',required=True)
    family = parser.add_mutually_exclusive_group()
    family.add_argument('--capacity-family', choices=('table8','table9'))
    family.add_argument('--experiment', choices=('table5','table10'))
    parser.add_argument('--job',required=True)
    parser.add_argument('--training-audit',required=True)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--audit',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--seconds',type=int,default=900)
    parser.add_argument('--deadline',type=datetime.fromisoformat)
    args=parser.parse_args()
    assert args.out.parent.resolve()==Path('runs').resolve()
    if args.audit:
        assert not args.resume and not args.audit.exists()
        assert args.audit.parent.resolve()==Path('runs').resolve()
    else:
        now=datetime.now().astimezone()
        assert args.deadline and now+timedelta(seconds=660)<args.deadline<=now+timedelta(hours=2)
        assert 30<=args.seconds<=1100 and args.out.exists()==args.resume
    assert not any(name in os.environ for name in ('PYTORCH_MPS_HIGH_WATERMARK_RATIO','PYTORCH_ENABLE_MPS_FALLBACK'))
    with Path('runs/hf-mmmlu-global.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.audit:
            assert pressure() == 1
            evaluate(args, [])
            return
        free = int(re.search(r'free percentage:\s*(\d+)%', native(['memory_pressure','-Q']))[1])
        assert pressure() == 1 and free >= 55
        from evaluate_batched import stop_requests
        with stop_requests() as requested:
            evaluate(args, requested)


if __name__=='__main__':
    main()
