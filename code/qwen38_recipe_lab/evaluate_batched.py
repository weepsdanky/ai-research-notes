"""Resumable equal-length batches: no padding, no prompt truncation.

Separate from the frozen sequential evaluator. Code generation is unscored
until the pinned official EvalPlus harness has actually judged it.
"""
import argparse
import contextlib
import ctypes
import fcntl
import fnmatch
import hashlib
import importlib
import json
import platform
import re
import resource
import signal
import sys
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer
from evaluate import digest
from evaluate_suite import now, save_meta
from inference import Inference
from model import Model, Config
from train import device_for


@contextlib.contextmanager
def stop_requests():
    requested=[]
    def request(signum,frame):requested.append(signal.Signals(signum).name)
    previous={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        for s in previous:signal.signal(s,request)
        yield requested
    finally:
        for s,handler in previous.items():signal.signal(s,handler)


def system_pressure():
    # Native read avoids forking a subprocess from the large evaluator process.
    libc=ctypes.CDLL(None,use_errno=True);value=ctypes.c_int();size=ctypes.c_size_t(ctypes.sizeof(value))
    call=libc.sysctlbyname;call.argtypes=[ctypes.c_char_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.c_void_p,ctypes.c_size_t]
    call.restype=ctypes.c_int
    if call(b'kern.memorystatus_vm_pressure_level',ctypes.byref(value),ctypes.byref(size),None,0) or size.value!=ctypes.sizeof(value):
        raise OSError(ctypes.get_errno(),'Unable to read Mac memory pressure')
    return value.value


def batches(jobs, size):
    group = []
    for job in jobs:
        if group and (len(group) == size or len(group[0][-1]) != len(job[-1])):
            yield group
            group = []
        group.append(job)
    if group:
        yield group


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint');p.add_argument('--out', required=True)
    p.add_argument('--tasks', nargs='+', required=True)
    p.add_argument('--suite', default='core', help='core/extra/code, or a named eval_<suite>_tasks adapter')
    scope = p.add_mutually_exclusive_group();scope.add_argument('--full', action='store_true');scope.add_argument('--limit', type=int)
    p.add_argument('--tokens', type=int, default=64);p.add_argument('--context', type=int, default=4096)
    p.add_argument('--seconds', type=int, default=1200);p.add_argument('--resume', action='store_true')
    p.add_argument('--batch-size', type=int, choices=[1, 2, 4, 8], default=8)
    p.add_argument('--prefill-chunk', type=int, default=128)
    p.add_argument('--gdn-scan', choices=['recurrent', 'chunk'], default='recurrent', help='Public GDN chunk reference for prefill only')
    p.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    p.add_argument('--prefix-cache', action='store_true', help='Reuse chunk-aligned per-task prefixes with four LRU slots')
    p.add_argument('--grouped-gqa', action='store_true', help='Contract sparse shared KV heads without repeating them per query head')
    p.add_argument('--stop-pressure',type=int,choices=[0,2,4],default=0,help='Mac pressure level that stops after a durable batch; zero disables this extra guard')
    a = p.parse_args()
    if min(a.tokens, a.context, a.seconds, a.limit if a.limit is not None else 1, a.prefill_chunk) < 1 or a.seconds > 1200 or a.prefill_chunk > 512:
        p.error('Positive budgets required; seconds <=1200, chunk <=512')
    if not re.fullmatch('[a-z][a-z0-9_]*', a.suite):p.error('Invalid adapter name')
    if a.stop_pressure and sys.platform!='darwin':p.error('Memory pressure guard requires macOS')
    torch.set_num_threads(1);torch.manual_seed(42)
    out = Path(a.out);meta_path = Path(a.out+'.meta.json');out.parent.mkdir(parents=True, exist_ok=True)
    with Path(a.out+'.lock').open('a') as lock,stop_requests() as requested:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        module = {'core': 'eval_tasks', 'extra': 'eval_extra_tasks', 'code': 'eval_code_tasks'}.get(a.suite, f'eval_{a.suite}_tasks')
        adapter = importlib.import_module(module)
        tasks = [t for t in adapter.load_tasks() if any(fnmatch.fnmatchcase(t.name, pat) for pat in a.tasks)]
        if not tasks or any(not any(fnmatch.fnmatchcase(t.name, pat) for t in tasks) for pat in a.tasks):
            p.error('Every task pattern must match')
        if hasattr(adapter, 'validate_selection'):adapter.validate_selection(tasks)
        tokenizer = Tokenizer.from_file('.cache/references/tokenizer.json')
        limit = None if a.full else (a.limit or 2)
        token_pool = {}
        def encode(prompt):
            return [token_pool.setdefault(token, token) for token in tokenizer.encode(prompt, add_special_tokens=False).ids]
        jobs = [(t, i, prompt, gold, encode(prompt))
                for t in tasks for i, (prompt, gold) in enumerate(t.rows[:limit])]
        jobs.sort(key=lambda j: (len(j[-1]), j[0].name, j[1]))
        content = hashlib.sha256()
        for t, i, prompt, gold, ids in jobs:
            content.update(json.dumps((t.name, i, t.shots, t.cot, t.kind, prompt, gold, ids), ensure_ascii=False).encode())
        core = {name: digest(name) for name in ('model.py', 'optim.py', 'polar_coefficients.json', 'train.py')}
        stops = {t.name: adapter.stopping(t.name) if hasattr(adapter, 'stopping') else [] for t in tasks}
        identity = dict(checkpoint_sha256=digest(a.checkpoint), content_sha256=content.hexdigest(),
            source_hashes={name: digest(name) for name in (__file__, adapter.__file__, 'eval_tasks.py', 'evaluate.py', 'evaluate_suite.py', 'inference.py')},
            runtime_model_sources=core, device=a.device, tokens=a.tokens, context=a.context,
            batch_size=a.batch_size, prefill_chunk=a.prefill_chunk, stop_strings=stops, suite=a.suite,
            tokenizer_sha256=digest('.cache/references/tokenizer.json'), tasks=[t.name for t in tasks],
            scope='full selected datasets' if a.full else f'pilot limit {limit} per task')
        identity['gdn_scan'] = a.gdn_scan
        identity['grouped_gqa'] = a.grouped_gqa
        identity['stop_pressure'] = a.stop_pressure
        if a.gdn_scan == 'chunk':
            identity['chunk_sources'] = {f: digest(f) for f in ('prefill_inference.py', 'reference.py', '.cache/references/modeling_qwen4_exp.py')}
        def cache_key(job):
            groups=getattr(job[0],'prefix_groups',None)
            return job[0].name if groups is None else job[0].name+'/'+str(groups[job[1]])
        prefixes = {}
        if a.prefix_cache:
            from prefix_inference import PrefixInference, common_prefix
            prefix_rows={}
            for job in jobs:prefix_rows.setdefault(cache_key(job),[]).append(job[-1])
            prefixes = {key:common_prefix(rows,a.prefill_chunk) for key,rows in prefix_rows.items()}
        identity['prefix_cache'] = dict(slots=4, source_sha256=digest('prefix_inference.py'),
            plans={name: dict(tokens=len(ids), sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()) for name, ids in prefixes.items()}) if a.prefix_cache else None
        counts = {t.name: dict(n=0, correct=0, scored=0, capacity_failures=0, total_dataset=len(t.rows), target=len(t.rows[:limit]),
                               shots=t.shots, cot=t.cot, kind=t.kind) for t in tasks}
        existing = [];completed = 0

        def count(record):
            c = counts[record['task']];c['n'] += 1;c['capacity_failures'] += int(record['stop'] == 'context_capacity')
            if record['correct'] is not None:
                c['scored'] += 1;c['correct'] += int(record['correct'])

        if a.resume:
            metadata = json.loads(meta_path.read_text())
            if metadata['identity'] != identity:
                raise ValueError('Resume identity mismatch')
            with out.open() as stream:
                for line in stream:
                    r = json.loads(line)
                    if completed >= len(jobs):raise ValueError('Extra prediction')
                    t, i, prompt, gold, ids = jobs[completed]
                    if (r['task'], r['index'], r['prompt'], r['input_ids'], r['gold']) != (t.name, i, prompt, ids, gold):
                        raise ValueError('Stored input prefix mismatch')
                    if r['correct'] != adapter.correct(r['prediction'], gold, t.kind):raise ValueError('Stored score mismatch')
                    existing.append(dict(output_ids=r['output_ids'], prediction=r['prediction']));count(r);completed += 1
        else:
            if out.exists() or meta_path.exists():raise FileExistsError('Use a new path or explicit --resume')
            out.touch(exist_ok=False)
            metadata = dict(identity=identity, started=now(), events=[], environment=dict(python=platform.python_version(), torch=str(torch.__version__), platform=platform.platform()),
                            protocol='greedy; exact-length batches in stable length/task/index order; no padding/truncation; code correctness pending official harness')
        saved = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
        if saved['step'] != saved['args']['steps'] or saved['source_hashes'] != core:
            raise ValueError('Completed checkpoint and exact training sources required')
        saved.pop('optimizers', None);device = device_for(a.device)
        model = Model(Config(**saved['config'])).to(device).eval();model.load_state_dict(saved.pop('model'))
        stage = 'sparse' if saved['args']['stage'] == 'sparse' else 'pretrain'
        metadata.update(checkpoint_step=saved['step'], config=saved['config'], training_context=saved['args']['seq']);del saved
        if a.gdn_scan == 'chunk':
            from prefill_inference import PrefillInference
            runner = PrefillInference(model, stage, prefill_chunk=a.prefill_chunk,grouped_gqa=a.grouped_gqa)
        else:
            runner = Inference(model, stage, prefill_chunk=a.prefill_chunk,grouped_gqa=a.grouped_gqa)
        prefix_runners = {}
        metadata['events'].append(dict(start=now(), resume=a.resume, starting_items=completed, budget_seconds=a.seconds))
        metadata.update(status='running', completed_items=completed, tasks=counts);save_meta(meta_path, metadata)
        started = time.monotonic();position = 0;stop_reason='time_budget'
        with out.open('a') as stream:
            for group in batches(jobs, a.batch_size):
                if position+len(group) <= completed:
                    position += len(group);continue
                tick = time.perf_counter();generated = [[] for _ in group];reasons = ['max_new_tokens']*len(group)
                if len(group[0][-1])+a.tokens > a.context:
                    reasons = ['context_capacity']*len(group)
                else:
                    prefill_runner = runner;name = cache_key(group[0])
                    if a.prefix_cache and prefixes[name] and all(cache_key(j) == name for j in group):
                        prefill_runner = prefix_runners.pop(name, None)
                        if prefill_runner is None:prefill_runner = PrefixInference(runner, prefixes[name])
                        prefix_runners[name] = prefill_runner
                        if len(prefix_runners) > 4:prefix_runners.pop(next(iter(prefix_runners)))
                    cache = {};last = prefill_runner(torch.tensor([j[-1] for j in group], device=device), cache)
                    done = [False]*len(group)
                    for step in range(a.tokens):
                        tokens = last.argmax(-1)
                        for j, token in enumerate(tokens.cpu().tolist()):
                            if done[j]:continue
                            generated[j].append(token)
                            if token == model.config.eos:
                                done[j] = True;reasons[j] = 'eos'
                            elif stops[group[j][0].name]:
                                tail = tokenizer.decode(generated[j][-64:], skip_special_tokens=False)
                                if any(s in tail for s in stops[group[j][0].name]):
                                    done[j] = True;reasons[j] = 'stop_string'
                        if all(done) or step+1 == a.tokens:break
                        last = runner(tokens[:, None], cache)
                elapsed = time.perf_counter()-tick
                for j, (t, i, prompt, gold, ids) in enumerate(group):
                    raw = tokenizer.decode(generated[j], skip_special_tokens=False)
                    text = adapter.completion(tokenizer.decode(generated[j], skip_special_tokens=True), stops[t.name]) if hasattr(adapter, 'completion') else raw
                    r = dict(task=t.name, index=i, shots=t.shots, cot=t.cot, answer_kind=t.kind, prompt=prompt, input_ids=ids,
                             prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), gold=gold, prompt_tokens=len(ids),
                             context_extrapolation=len(ids)>metadata['training_context'], output_ids=generated[j], raw_prediction=raw,
                             prediction=text, correct=adapter.correct(text, gold, t.kind), stop=reasons[j],
                             unused_token_ids=[v for v in generated[j] if tokenizer.id_to_token(v) is None],
                             batch_start=position, batch_items=len(group), batch_seconds=elapsed,
                             seconds=elapsed/len(group), timing_note='batch elapsed divided by items, not isolated item latency')
                    if position+j < completed:
                        old = existing[position+j]
                        if old['output_ids'] != r['output_ids'] or old['prediction'] != r['prediction']:
                            raise ValueError('Partial-batch replay differs; retain original records')
                        continue
                    stream.write(json.dumps(r, ensure_ascii=False)+'\n');stream.flush();count(r);completed += 1
                position += len(group)
                measured = metadata.setdefault('resources', {})
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
                measured['process_peak_rss_bytes'] = max(measured.get('process_peak_rss_bytes', 0), rss)
                if device == 'mps':
                    measured['mps_driver_max_sampled_bytes'] = max(measured.get('mps_driver_max_sampled_bytes', 0), torch.mps.driver_allocated_memory())
                metadata.update(completed_items=completed, tasks=counts);save_meta(meta_path, metadata)
                print(now(), completed, len(jobs), flush=True)
                if requested:stop_reason='interrupted_signal';break
                if a.stop_pressure:
                    try:pressure=system_pressure()
                    except (OSError,ValueError):stop_reason='resource_monitor_failed';break
                    measured['max_sampled_system_pressure']=max(measured.get('max_sampled_system_pressure',0),pressure)
                    if pressure>=a.stop_pressure:stop_reason='memory_pressure';break
                if time.monotonic()-started >= a.seconds:break
        for c in counts.values():c['accuracy'] = c['correct']/c['scored'] if c['scored'] else None
        metadata.update(status='complete' if completed == len(jobs) else stop_reason, completed_items=completed, tasks=counts,
                        full_dataset_attempted=completed == len(jobs) and a.full,
                        full_dataset_generated=completed == len(jobs) and a.full and not any(c['capacity_failures'] for c in counts.values()))
        scored = sum(c['scored'] for c in counts.values())
        metadata.update(micro_accuracy=sum(c['correct'] for c in counts.values())/scored if scored else None,
                        scoring_status='pending official code harness' if any(t.kind == 'code_pending' for t in tasks) else 'scored')
        metadata['events'][-1].update(end=now(), ending_items=completed,stop_reason=None if completed==len(jobs) else stop_reason,signals=requested);save_meta(meta_path, metadata)
        print(metadata['status'], completed, len(jobs), flush=True)


if __name__ == '__main__':
    main()
