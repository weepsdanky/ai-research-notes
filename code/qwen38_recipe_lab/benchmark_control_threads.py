"""Measure CPU thread counts against exact completed Mac MATH predictions."""
import argparse
from collections import OrderedDict
from datetime import datetime, timedelta
import fcntl
import json
import os
from pathlib import Path
import resource
from time import perf_counter

from control_eval_binding import training_input
from qwen35_quality_data import (PROTOCOL, adapter_for, batches, digest, encoded_hash,
                                group_key, load, prepare_jobs, runtime_versions, write_json)
from run_control_quality import LOCK, memory, normal, now
from run_dense_text_guarded import native

RUNS = Path('runs')
EXAMPLES = {'table5': 'gr', 'table10': 'center'}


def reference(experiment, group_count):
    from tokenizers import Tokenizer

    arm = EXAMPLES[experiment]
    raw = RUNS / f'{experiment}-public-{arm}-math.jsonl'
    audit_path = raw.with_name(raw.stem + '-audit.json')
    training_audit = RUNS / f'round38-{experiment}-audit.json'
    audit = load(audit_path)
    identity = audit['identity']
    assert identity['family'] == experiment and identity['arm'] == arm and identity['job'] == 'math'
    assert identity['decoding'] == 'greedy' and not any(identity['stop_strings'].values())
    assert audit['complete'] and audit['items'] == audit['target'] == 5000
    assert digest(raw) == audit['output_sha256'] and digest(training_audit) == identity['training_audit_sha256']
    training = training_input(experiment, arm, training_audit)
    assert training[1]['checkpoint_sha256'] == identity['checkpoint_sha256']
    assert digest(PROTOCOL) == identity['protocol_sha256'] and runtime_versions() == identity['versions']
    assert all(digest(path) == sha for path, sha in identity['sources'].items())
    job, = [job for job in load(PROTOCOL)['jobs'] if job['name'] == 'math']
    assert job == identity['job_spec']
    tokenizer_path = '.cache/references/tokenizer.json'
    assert digest(tokenizer_path) == identity['tokenizer_sha256']
    _, tasks = adapter_for(job)
    rows, content, prefixes = prepare_jobs(tasks, Tokenizer.from_file(tokenizer_path), job)
    assert content == identity['content_sha256'] and encoded_hash(prefixes) == identity['prefix_sha256']
    assert encoded_hash([(task.name, index) for task, index, *_ in rows]) == identity['order_sha256']
    groups = list(batches(rows, job['batch_size']))
    assert 2 <= group_count <= len(groups)
    selection = {i * (len(groups) - 1) // (group_count - 1) for i in range(group_count)}
    chosen = []
    position = 0
    with raw.open() as stream:
        for group_index, group in enumerate(groups):
            lines = [next(stream) for _ in group]
            if group_index in selection:
                expected = []
                for line, (task, index, _, gold, ids) in zip(lines, group, strict=True):
                    row = json.loads(line)
                    assert (row['task'], row['index'], row['gold'], row['input_ids']) == (task.name, index, gold, ids)
                    expected.append((row['output_ids'], row['stop']))
                chosen.append((position, group, expected))
            position += len(group)
        assert next(stream, None) is None
    evidence = dict(experiment=experiment, arm=arm, job='math', raw_sha256=audit['output_sha256'],
                    audit_sha256=digest(audit_path), checkpoint_sha256=identity['checkpoint_sha256'],
                    training_audit_sha256=identity['training_audit_sha256'],
                    selected_batches=[dict(first_row=i, width=len(group), prompt_tokens=len(group[0][4]))
                                      for i, group, _ in chosen],
                    selected_rows=sum(len(group) for _, group, _ in chosen),
                    selected_reference_sha256=encoded_hash([(i, expected) for i, _, expected in chosen]))
    return training, job, chosen, prefixes, evidence


def measure(experiment, prepared, threads, deadline):
    import torch

    training, job, groups, prefixes, evidence = prepared
    torch.set_num_threads(threads)
    profile, binding, checkpoint, construct = training
    model, identity = construct()
    saved = torch.load(checkpoint, weights_only=True, map_location='cpu', mmap=True)
    step = len(saved['history']) if experiment == 'table10' else saved['step']
    assert saved['identity'] == identity and step == profile['steps']
    assert saved['final_validation'] == binding['final_validation']
    model.load_state_dict(saved['model'], assign=True)
    del saved
    model.eval()

    from prefix_inference import PrefixInference
    if experiment == 'table5':
        from residual_inference import ResidualInference as Runner
    else:
        from bounded_dense_inference import BoundedDenseInference as Runner

    runner = Runner(model, prefill_chunk=job['prefill_chunk'])
    prefix_cache = OrderedDict()
    seconds = tokens = processed = 0
    status = 'complete'
    mismatches = []
    swap_before = native(['sysctl', 'vm.swapusage']).strip()
    samples = []
    with torch.no_grad():
        for position, group, expected in groups:
            sample = memory()
            samples.append(sample)
            if not normal(sample):
                status = 'memory_pressure'
                break
            if now() >= deadline - timedelta(seconds=600):
                status = 'time_budget'
                break
            tick = perf_counter()
            keys = {group_key(task, index) for task, index, *_ in group}
            prefill = runner
            if len(keys) == 1:
                key, = keys
                if prefixes[key]:
                    prefill = prefix_cache.pop(key, None)
                    if prefill is None:
                        prefill = PrefixInference(runner, prefixes[key])
                    prefix_cache[key] = prefill
                    if len(prefix_cache) > 4:
                        prefix_cache.popitem(last=False)
            cache = {}
            logits = prefill(torch.tensor([row[4] for row in group]), cache)
            output = [[] for _ in group]
            reasons = ['max_new_tokens'] * len(group)
            eos = model.config.eos
            for step in range(job['tokens']):
                selected = logits.argmax(-1)
                for index, token in enumerate(selected.tolist()):
                    if reasons[index] != 'eos':
                        output[index].append(token)
                        if token == eos:
                            reasons[index] = 'eos'
                if all(reason == 'eos' for reason in reasons) or step + 1 == job['tokens']:
                    break
                logits = runner(selected[:, None], cache)
            seconds += perf_counter() - tick
            tokens += sum(map(len, output))
            processed += len(group)
            candidates = zip(output, reasons, strict=True)
            mismatches.extend(position + i for i, (actual, gold) in enumerate(zip(candidates, expected, strict=True))
                              if actual != gold)
    evidence.update(status=status, threads=torch.get_num_threads(), processed=processed, generated_tokens=tokens,
                    seconds=seconds, items_per_second=processed / seconds if seconds else None,
                    mismatch_rows=mismatches, exact_on_selection=processed == evidence['selected_rows'] and not mismatches,
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    swap_before=swap_before, swap_after=native(['sysctl', 'vm.swapusage']).strip(), memory_samples=samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=EXAMPLES, required=True)
    parser.add_argument('--groups', type=int, default=16)
    parser.add_argument('--threads', type=int, choices=(1, 2, 4))
    parser.add_argument('--deadline', type=datetime.fromisoformat)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert args.out.parent.resolve() == RUNS.resolve() and not args.out.exists()
    assert (args.threads is None) == (args.deadline is None)
    if args.threads is None:
        prepared = reference(args.experiment, args.groups)
    else:
        assert not {'PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'PYTORCH_ENABLE_MPS_FALLBACK'}.intersection(os.environ)
        with LOCK.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert normal(memory()) and now() + timedelta(seconds=690) < args.deadline <= now() + timedelta(hours=2)
            prepared = reference(args.experiment, args.groups)
            assert normal(memory()) and now() + timedelta(seconds=660) < args.deadline
            measure(args.experiment, prepared, args.threads, args.deadline)
    evidence = prepared[-1]
    evidence['benchmark_sha256'] = digest(__file__)
    write_json(args.out, evidence)


if __name__ == '__main__':
    main()
