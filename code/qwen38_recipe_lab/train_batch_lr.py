"""Train and audit the five equal-token Mac controls for Qwen Table 10."""
import argparse
from datetime import datetime, timedelta
import fcntl
import math
import os
from pathlib import Path
import re
import resource
import signal
import statistics
import time

from qwen35_quality_data import digest, encoded_hash, load, write_json
from run_dense_text_guarded import native, pressure
from train_ngram_placement import checked_model, fingerprint

PROFILE = Path('profiles/table10_mac_batch_lr.json')
RUNS = Path('runs')
OUTPUT = RUNS / 'table10-batch-lr-mac'


def learning_rate(profile, arm, tokens):
    warmup = profile['warmup_tokens']
    if tokens <= warmup:
        return arm['peak_lr'] * (tokens / warmup)
    total = profile['total_windows'] * profile['sequence_tokens']
    progress = (tokens - warmup) / (total - warmup)
    floor = profile['final_lr_fraction']
    return arm['peak_lr'] * (floor + (1-floor) * (1 + math.cos(math.pi * progress)) / 2)


def sample_plan(profile, corpora):
    import torch

    rng = torch.Generator().manual_seed(profile['seed'])
    seq = profile['sequence_tokens']
    plan = []
    for _ in range(profile['total_windows']):
        corpus = int(torch.rand((), generator=rng).item() < profile['mix_probability'])
        offset = torch.randint(len(corpora[corpus].arrays['train'])-seq, (), generator=rng).item()
        plan.append((corpus, offset))
    return plan


def settings(profile, placement, plan):
    model = checked_model(placement, profile['placement_arm'])
    binding = dict(profile_sha256=digest(PROFILE), runner_sha256=digest(__file__),
                   data_plan_sha256=encoded_hash(plan), initialization_sha256=fingerprint(model, lambda _: True),
                   frozen_indexer_sha256=fingerprint(model, lambda name: '.indexer.' in name))
    validation = dict(seq=profile['sequence_tokens'], batch=profile['validation_batch'],
                      eval_batches=profile['validation_batches'], head_chunk=profile['head_chunk'])
    return model, binding, validation


def batch(corpora, windows, seq):
    import numpy as np
    import torch

    ids = np.stack([corpora[corpus].arrays['train'][offset:offset+seq+1] for corpus,offset in windows])
    ids = torch.from_numpy(ids.astype(np.int64))
    return ids[:, :-1], ids[:, 1:]


def train(args, profile, corpora, plan, model, binding, validation):
    import torch

    from audit_qwen35_training import finite
    from chunked_lm_loss import chunked_cross_entropy
    from optim import RecipeOptimizer, ownership
    from train_qwen35_controls import balance_loss, evaluate

    arm = profile['arms'][args.arm]
    binding = dict(binding, arm=args.arm, torch_version=str(torch.__version__))
    width, seq = arm['batch'], profile['sequence_tokens']
    steps = len(plan) // width
    out = OUTPUT / args.arm
    checkpoint, state = out / 'last.pt', out / 'state.json'
    optimizer = RecipeOptimizer(ownership(model), arm['peak_lr'])
    final = None
    if args.resume:
        if state.exists():
            previous = load(state)
            assert previous['status'] != 'complete'
            if previous['status'] != 'running':
                assert digest(checkpoint) == previous['checkpoint_sha256']
        saved = torch.load(checkpoint, weights_only=True, map_location='cpu', mmap=True)
        assert saved['identity'] == binding
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        history, initial, final = saved['history'], saved['initial_validation'], saved['final_validation']
        del saved
    else:
        assert not checkpoint.exists() and not state.exists()
        history, initial = [], evaluate(model, corpora, validation, 'cpu')
        out.mkdir(parents=True, exist_ok=True)
    assert len(history) <= steps

    def record(status):
        write_json(state, dict(identity=binding, status=status, pid=os.getpid(), step=len(history),
                               tokens=len(history)*width*seq, target_steps=steps, initial_validation=initial,
                               final_validation=final, checkpoint_sha256=digest(checkpoint)))

    def save(status):
        temp = checkpoint.with_suffix('.tmp')
        torch.save(dict(identity=binding, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        history=history, initial_validation=initial, final_validation=final), temp)
        temp.replace(checkpoint)
        record(status)

    if final is not None:
        assert len(history) == steps
        record('complete')
        return
    requests = []
    for signum in (signal.SIGINT,signal.SIGTERM):
        signal.signal(signum, lambda number, frame: requests.append(number))
    cutoff = args.deadline - timedelta(seconds=600)
    stop = time.monotonic() + args.seconds

    def stop_reason():
        if requests:
            return 'interrupted'
        if pressure() >= 2:
            return 'memory_pressure'
        if datetime.now().astimezone() >= cutoff or time.monotonic() >= stop:
            return 'time_budget'
        return None

    save('running')
    reason = None
    while len(history) < steps:
        reason = stop_reason()
        if reason:
            break
        tick = time.monotonic()
        start = len(history) * width
        windows = plan[start:start+width]
        inputs, labels = batch(corpora, windows, seq)
        lr = learning_rate(profile, arm, (start+width)*seq)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        hidden, routers = model.hidden(inputs)
        ce = chunked_cross_entropy(hidden, model.head.weight, labels, profile['head_chunk'])
        balance = balance_loss(routers, model.config.experts, model.config.top_k)
        (ce + profile['moe_balance_weight']*balance).backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), profile['clip'], error_if_nonfinite=True)
        optimizer.step()
        history.append(dict(ce=ce.item(), balance=balance.item(), grad_norm=norm.item(), lr=lr,
                            windows_sha256=encoded_hash(windows), seconds=time.monotonic()-tick,
                            rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        if len(history) % 8 == 0:
            save('running')
            print(f"{args.arm} {len(history)}/{steps} CE {ce.item():.5f}", flush=True)

    reason = reason or stop_reason()
    if reason is None:
        final = evaluate(model, corpora, validation, 'cpu')
        reason = 'complete'
    finite(model.state_dict())
    finite(optimizer.state_dict())
    assert fingerprint(model, lambda name: '.indexer.' in name) == binding['frozen_indexer_sha256']
    save(reason)


def audit(profile, corpora, plan, model, binding, validation):
    import torch

    from audit_qwen35_training import finite
    from optim import ownership
    from train_qwen35_controls import evaluate

    results = {}
    initial = evaluate(model, corpora, validation, 'cpu')
    seq = profile['sequence_tokens']
    for name, arm in profile['arms'].items():
        out = OUTPUT / name
        if not out.exists():
            continue
        assert pressure() == 1
        checkpoint = out / 'last.pt'
        state = load(out / 'state.json')
        assert state['status'] != 'running' and digest(checkpoint) == state['checkpoint_sha256']
        saved = torch.load(checkpoint, weights_only=True, map_location='cpu', mmap=True)
        expected = dict(binding, arm=name, torch_version=str(torch.__version__))
        assert state['identity'] == saved['identity'] == expected
        model.load_state_dict(saved['model'])
        assert fingerprint(model, lambda key: '.indexer.' in key) == binding['frozen_indexer_sha256']
        finite(saved['model'])
        finite(saved['optimizer'])
        groups = saved['optimizer']['param_groups']
        assert [(name,pieces) for name,_,pieces in ownership(model)] == [
            (group['name'],group['pieces']) for group in groups]
        width, rows = arm['batch'], saved['history']
        assert len(rows) == state['step'] <= len(plan)//width
        assert state['tokens'] == len(rows)*width*seq
        for index, row in enumerate(rows):
            assert row['windows_sha256'] == encoded_hash(plan[index*width:(index+1)*width])
            assert row['lr'] == learning_rate(profile,arm,(index+1)*width*seq)
            assert all(math.isfinite(row[key]) for key in ('ce','balance','grad_norm','seconds'))
        if rows:
            assert all(group['lr'] == rows[-1]['lr'] for group in groups)
        assert state['initial_validation'] == saved['initial_validation'] == initial
        final = saved['final_validation']
        assert state['final_validation'] == final
        complete = state['status'] == 'complete'
        assert complete == (final is not None)
        if complete:
            assert state['tokens'] == len(plan)*seq and evaluate(model,corpora,validation,'cpu') == final
        ce = [row['ce'] for row in rows]
        radius = profile['spike_local_radius']
        spikes = [index+1 for index in range(radius,len(rows)-radius)
                  if ce[index]-statistics.median(ce[index-radius:index]+ce[index+1:index+radius+1])
                  > profile['spike_excess']]
        norms = [row['grad_norm'] for row in rows]
        clipped = [i+1 for i,norm in enumerate(norms) if norm > profile['clip']]
        p999 = statistics.quantiles(norms, n=1000, method='inclusive')[-1] if len(norms) > 1 else None
        warmup = profile['warmup_tokens']
        results[name] = dict(state, complete=complete, batch=width, peak_lr=arm['peak_lr'],
                             loss_spike_steps=spikes, loss_spike_eligible=max(0,len(rows)-2*radius),
                             max_preclip_grad_norm=max(norms,default=None), p999_preclip_grad_norm=p999,
                             clipped_steps=clipped,
                             clipped_after_warmup=sum(i*width*seq > warmup for i in clipped),
                             median_update_seconds=statistics.median(row['seconds'] for row in rows) if rows else None,
                             max_rss_bytes=max((row['rss_bytes'] for row in rows),default=0))
        del saved
    return dict(binding=binding, arms=results,
                complete=len(results)==len(profile['arms']) and all(row['complete'] for row in results.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--arm')
    operation.add_argument('--audit-out', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--deadline', type=datetime.fromisoformat)
    parser.add_argument('--seconds', type=int, default=900)
    args = parser.parse_args()
    profile = load(PROFILE)
    if args.arm:
        now = datetime.now().astimezone()
        assert args.arm in profile['arms'] and 30 <= args.seconds <= 1100
        assert args.deadline and now+timedelta(seconds=660) < args.deadline <= now+timedelta(hours=2)
    else:
        assert not args.resume and not args.audit_out.exists() and args.audit_out.parent.resolve()==RUNS.resolve()
    assert not any(key in os.environ for key in ('PYTORCH_MPS_HIGH_WATERMARK_RATIO','PYTORCH_ENABLE_MPS_FALLBACK'))
    assert all(digest(path)==sha for path,sha in profile['source_hashes'].items())
    assert all(profile['total_windows']%arm['batch']==0 for arm in profile['arms'].values())
    placement = load(profile['placement_profile'])
    with (RUNS / 'hf-mmmlu-global.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        free = int(re.search(r'free percentage:\s*(\d+)%',native(['memory_pressure','-Q']))[1])
        assert pressure()==1 and free>=55
        import torch
        from train import Corpus

        torch.set_num_threads(1)
        corpora = [Corpus(path,profile['sequence_tokens']) for path in placement['data']]
        plan = sample_plan(profile,corpora)
        model,binding,validation = settings(profile,placement,plan)
        if args.arm:
            train(args,profile,corpora,plan,model,binding,validation)
        else:
            write_json(args.audit_out,audit(profile,corpora,plan,model,binding,validation))


if __name__ == '__main__':
    main()
