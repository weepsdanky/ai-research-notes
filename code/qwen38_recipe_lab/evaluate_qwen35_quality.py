"""Generate one complete declared quality job from an audited800+20 control.

Only the owning guard may launch this worker. Every raw model ID is retained,
including unused padded vocabulary IDs. Code outputs require separate judging.
"""
import argparse
import hashlib
import os
import signal
import time
from pathlib import Path
from datetime import datetime
from qwen35_quality_data import (PROTOCOL, load, digest, selection, training_binding,
                                check_header, adapter_for, prepare_jobs, batches,
                                group_key, decoded, encoded_hash, runtime_versions, write_json)


def runtime_sources():
    names = ['evaluate_qwen35_quality.py', 'audit_qwen35_quality.py',
             'qwen35_quality_data.py', 'qwen35_quality_generation.py',
             'qwen35_inference.py', 'qwen35_controls.py', 'model.py',
             'qwen35_formal_contract.py', 'run_qwen35_quality_guarded.py',
             'eval_tasks.py', 'eval_extra_tasks.py', 'eval_mmmlu_tasks.py',
             'eval_code_tasks.py', 'eval_multipl_tasks.py']
    return {name: digest(name) for name in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['arm', 'job', 'training-profile', 'checkpoint', 'audit', 'deadline']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--seconds', type=int, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    assert os.getppid() == int(os.environ['QWEN35_QUALITY_GUARD_PID'])
    assert 30 <= args.seconds <= 1140
    assert all(os.environ.get(key) is None for key in ['PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'PYTORCH_ENABLE_MPS_FALLBACK'])
    binding = training_binding(args.training_profile, args.audit, args.checkpoint, args.arm)
    protocol = load(PROTOCOL);job = selection(protocol, args.arm, args.job)
    path = Path(job['output_template'].format(arm=args.arm))
    assert path.exists() == args.resume and Path(str(path) + '.meta.json').exists() == args.resume
    import torch
    from tokenizers import Tokenizer
    from qwen35_controls import Qwen35Control, ControlConfig
    from qwen35_inference import ControlInference
    from qwen35_quality_generation import ControlGreedy
    from run_dense_text_guarded import pressure
    torch.set_num_threads(1)
    tokenizer = Tokenizer.from_file(protocol['tokenizer']['file'])
    adapter, tasks = adapter_for(job)
    rows, content, prefixes = prepare_jobs(tasks, tokenizer, job)
    profile = load(args.training_profile)
    identity = dict(protocol_sha256=digest(PROTOCOL), job=args.job, job_spec=job,
                    binding=binding, tokenizer_sha256=protocol['tokenizer']['sha256'],
                    content_sha256=content, runtime_sources=runtime_sources(),
                    vocab_size=profile['config']['vocab_size'],
                    runtime_versions=runtime_versions(),
                    prefix_plan_sha256=encoded_hash(prefixes), generation=protocol['generation'])
    counts = {task.name: dict(n=0, scored=0, correct=0, unknown_tokenizer_items=0) for task in tasks}
    previous = []
    if args.resume:
        from audit_qwen35_quality import audit
        audit(path, allow_partial=True)
        meta = load(str(path) + '.meta.json');assert meta['identity'] == identity
        assert not meta['full_dataset_generated']
        with path.open() as stream:
            previous = [__import__('json').loads(line) for line in stream]
        counts = meta['tasks']
    # Inference-only assign: no optimizer construction or parameter-piece update.
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    check_header(saved, profile, binding)
    with torch.device('meta'):
        model = Qwen35Control(ControlConfig(**saved['config']), saved['arm'], seed=profile['seed'])
    model.load_state_dict(saved['model'], strict=True, assign=True);del saved
    model.eval();engine = ControlInference(model, query_chunk=16, max_context=job['context'])
    generate = ControlGreedy(engine, protocol['eos'], job['prefill_chunk'], 4)
    requested = []
    for signum in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(signum, lambda number, frame: requested.append(number))
    start = time.monotonic();completed = len(previous);status = 'running'
    def metadata():
        write_json(str(path) + '.meta.json', dict(identity=identity, status=status,
                   completed_items=completed, target_items=job['target_items'],
                   full_dataset_generated=completed == job['target_items'], tasks=counts,
                   updated_at=datetime.now().astimezone().isoformat(),
                   invocation_seconds=time.monotonic()-start))
    metadata()
    with path.open('a' if args.resume else 'x') as stream, torch.no_grad():
        offset = 0
        for batch in batches(rows, job['batch_size']):
            end = offset + len(batch)
            if end <= len(previous):offset = end;continue
            if pressure() >= 2:status = 'memory_pressure';break
            if requested:status = 'interrupted_signal';break
            if time.monotonic()-start >= args.seconds or datetime.now().astimezone() >= datetime.fromisoformat(args.deadline):
                status = 'time_budget';break
            stops = [adapter.stopping(row[0].name) if hasattr(adapter, 'stopping') else [] for row in batch]
            def stopping(index, ids):
                tail = tokenizer.decode(ids[-64:], skip_special_tokens=False)
                return any(stop in tail for stop in stops[index])
            keys = [group_key(row[0], row[1]) for row in batch]
            prefix = prefixes[keys[0]] if len(set(keys)) == 1 else []
            tick = time.monotonic()
            outputs, reasons = generate.generate([row[4] for row in batch], job['tokens'], prefix=prefix, stopping=stopping)
            elapsed = time.monotonic()-tick
            for position, ((task, index, prompt, gold, ids), output, reason) in enumerate(zip(batch, outputs, reasons)):
                raw, prediction = decoded(adapter, tokenizer, output, stops[position])
                record = dict(task=task.name, index=index, prompt=prompt, gold=gold,
                              input_ids=ids, answer_kind=task.kind, shots=task.shots, cot=task.cot,
                              prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                              output_ids=output, raw_prediction=raw, prediction=prediction,
                              unused_token_ids=[token for token in output if tokenizer.id_to_token(token) is None],
                              stop=reason, correct=adapter.correct(prediction, gold, task.kind),
                              batch_seconds=elapsed, actual_batch_size=len(batch))
                if offset + position < len(previous):
                    old = previous[offset+position]
                    assert all(record[key] == value for key, value in old.items() if key not in ['batch_seconds'])
                    continue
                stream.write(__import__('json').dumps(record, ensure_ascii=False) + '\n')
                count = counts[task.name];count['n'] += 1
                count['unknown_tokenizer_items'] += bool(record['unused_token_ids'])
                if record['correct'] is not None:
                    count['scored'] += 1;count['correct'] += int(record['correct'])
                completed += 1
            stream.flush();os.fsync(stream.fileno());metadata();offset = end
        else:status = 'complete'
    metadata()
    assert digest(args.checkpoint) == binding['checkpoint_sha256']


if __name__ == '__main__':
    main()
