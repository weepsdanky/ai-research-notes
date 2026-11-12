"""Train and audit fixed-capacity Mac N-gram placement controls."""
import argparse
from datetime import datetime, timedelta
import fcntl
import hashlib
import math
import os
from pathlib import Path
import re
import signal
import statistics
import time

from qwen35_quality_data import digest, load, write_json
from run_dense_text_guarded import native, pressure


def fingerprint(model, keep):
    checksum = hashlib.sha256()
    for name, value in model.state_dict().items():
        if keep(name):
            checksum.update(name.encode())
            checksum.update(value.numpy().tobytes())
    return checksum.hexdigest()


def checked_model(profile, arm):
    from model import Config
    from ngram_placement import NGramPlacement

    model = NGramPlacement(Config(**profile['config']), arm, profile['seed'])
    expected = 0 if arm == 'none' else profile['ngram_parameters_per_enabled_arm']
    assert model.ngram_parameters() == expected
    return model


def identity(profile_path, model):
    return dict(profile_sha256=digest(profile_path), runner_sha256=digest(__file__),
                arm=model.arm,
                shared_initial_sha256=fingerprint(model, lambda name: '.ple.' not in name),
                frozen_indexer_sha256=fingerprint(model, lambda name: '.indexer.' in name))


def train(args, profile, corpora):
    import resource
    import torch

    from audit_qwen35_training import finite
    from chunked_lm_loss import chunked_cross_entropy
    from optim import RecipeOptimizer, ownership
    from train_qwen35_controls import balance_loss, evaluate

    model = checked_model(profile, args.arm)
    optimizer = RecipeOptimizer(ownership(model), profile['lr'])
    rng = torch.Generator().manual_seed(profile['seed'])
    out = Path(profile['output'])/args.arm
    checkpoint = out/'last.pt'
    state = out/'state.json'
    binding = identity(args.profile, model)
    if args.resume:
        saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
        assert saved['identity'] == binding and saved['final_validation'] is None
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.set_state(saved['rng'])
        history, initial = saved['history'], saved['initial_validation']
        assert saved['step'] == len(history) <= profile['steps']
        del saved
    else:
        out.mkdir(parents=True, exist_ok=False)
        history, initial = [], evaluate(model, corpora, profile, 'cpu')
    first = len(history)

    def save(final=None):
        temp = checkpoint.with_suffix('.tmp')
        torch.save(dict(identity=binding, step=len(history), model=model.state_dict(),
                        optimizer=optimizer.state_dict(), rng=rng.get_state(),
                        history=history, initial_validation=initial, final_validation=final), temp)
        temp.replace(checkpoint)

    def progress(status):
        write_json(state, dict(status=status, pid=os.getpid(), step=len(history), **binding))

    requests = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda number, frame: requests.append(number))
    start = time.monotonic()
    cutoff = args.deadline - timedelta(seconds=600)
    reason = 'complete'
    if not args.resume:
        save()
    progress('running')
    while len(history) < profile['steps']:
        level = pressure()
        if requests or level >= 2:
            reason = 'interrupted' if requests else 'memory_pressure'
            break
        if datetime.now().astimezone() >= cutoff or time.monotonic()-start >= args.seconds:
            reason = 'time_budget'
            break
        tick = time.monotonic()
        corpus = int(torch.rand((), generator=rng).item() < profile['mix_probability'])
        inputs, labels = corpora[corpus].batch('train', profile['batch'], profile['seq'], rng, 'cpu')
        optimizer.zero_grad(set_to_none=True)
        hidden, routers = model.hidden(inputs)
        ce = chunked_cross_entropy(hidden, model.head.weight, labels, profile['head_chunk'])
        balance = balance_loss(routers, model.config.experts, model.config.top_k)
        (ce + profile['balance_weight']*balance).backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), profile['clip'], error_if_nonfinite=True)
        optimizer.step()
        history.append(dict(step=len(history)+1, corpus=corpus, ce=ce.item(), balance=balance.item(),
                            grad_norm=norm.item(), seconds=time.monotonic()-tick, pressure=level,
                            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        if len(history) % 16 == 0:
            save()
            progress('running')
    final = None
    if reason == 'complete':
        if pressure() == 1 and not requests and datetime.now().astimezone() < cutoff:
            final = evaluate(model, corpora, profile, 'cpu')
        else:
            reason = 'validation_pending'
    finite(model.state_dict())
    finite(optimizer.state_dict())
    assert fingerprint(model, lambda name: '.indexer.' in name) == binding['frozen_indexer_sha256']
    save(final)
    write_json(state, dict(status=reason, pid=os.getpid(), step=len(history),
                           new_updates=len(history)-first, target_steps=profile['steps'],
                           tokens=len(history)*profile['seq']*profile['batch'],
                           checkpoint_sha256=digest(checkpoint),
                           ngram_parameters=model.ngram_parameters(),
                           parameters=sum(parameter.numel() for parameter in model.parameters()),
                           initial_validation=initial, final_validation=final,
                           swap=native(['sysctl','vm.swapusage']).strip(), **binding))


def audit(args, profile, corpora):
    import torch

    from audit_qwen35_training import finite
    from optim import ownership
    from train_qwen35_controls import evaluate

    results = {}
    shared = None
    for arm in profile['arms']:
        out = Path(profile['output'])/arm
        if not out.exists():
            continue
        assert pressure() == 1
        checkpoint = out/'last.pt'
        state = load(out/'state.json')
        assert state['status'] != 'running' and digest(checkpoint) == state['checkpoint_sha256']
        saved = torch.load(checkpoint, weights_only=True, map_location='cpu', mmap=True)
        model = checked_model(profile, arm)
        binding = identity(args.profile, model)
        assert binding == saved['identity'] and all(state[key] == value for key, value in binding.items())
        shared = shared or binding['shared_initial_sha256']
        assert shared == binding['shared_initial_sha256']
        model.load_state_dict(saved['model'])
        assert fingerprint(model, lambda name: '.indexer.' in name) == binding['frozen_indexer_sha256']
        finite(saved['model'])
        finite(saved['optimizer'])
        assert [(name, pieces) for name, _, pieces in ownership(model)] == [
            (group['name'], group['pieces']) for group in saved['optimizer']['param_groups']]
        rng = torch.Generator().manual_seed(profile['seed'])
        step = saved['step']
        assert 0 <= step == state['step'] == len(saved['history']) <= profile['steps']
        for index, row in enumerate(saved['history'], 1):
            corpus = int(torch.rand((), generator=rng).item() < profile['mix_probability'])
            assert row['step'] == index and row['corpus'] == corpus
            torch.randint(len(corpora[corpus].arrays['train'])-profile['seq'], (profile['batch'],), generator=rng)
        assert torch.equal(saved['rng'], rng.get_state())
        assert state['tokens'] == step*profile['seq']*profile['batch']
        assert state['ngram_parameters'] == model.ngram_parameters()
        for key in ('initial_validation', 'final_validation'):
            values = state[key]
            assert values == saved[key]
            if values is not None:
                assert set(values) == {f'val{i}/lm' for i in range(len(corpora))}
                assert all(math.isfinite(value) for value in values.values())
        complete = step == profile['steps'] and state['status'] == 'complete'
        assert complete == (saved['final_validation'] is not None)
        if complete:
            assert evaluate(model, corpora, profile, 'cpu') == state['final_validation']
        history = saved['history']
        assert all(math.isfinite(row[key]) for row in history for key in ('ce','balance','grad_norm','seconds'))
        results[arm] = dict(state, complete=complete, frozen_indexer_verified=True, data_rng_replayed=True,
                            last_recorded_training_ce=history[-1]['ce'] if history else None,
                            last_32_training_ce_mean=statistics.mean(row['ce'] for row in history[-32:]) if history else None,
                            median_update_seconds=statistics.median(row['seconds'] for row in history) if history else None,
                            max_process_rss_bytes=max((row['process_peak_rss_bytes'] for row in history), default=0))
        del model, saved
    complete = len(results) == len(profile['arms']) and all(state['complete'] for state in results.values())
    write_json(args.audit_out, dict(status='complete_mac_placement_ablation' if complete else 'partial_arms',
                                  profile_sha256=digest(args.profile), runner_sha256=digest(__file__),
                                  shared_initial_sha256=shared, arms=results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--arm')
    operation.add_argument('--audit-out', type=Path)
    parser.add_argument('--deadline', type=datetime.fromisoformat)
    parser.add_argument('--seconds', type=int, default=900)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.arm:
        now = datetime.now().astimezone()
        assert args.deadline and now+timedelta(seconds=660) < args.deadline <= now+timedelta(hours=2)
        assert 30 <= args.seconds <= 1100
    else:
        assert not args.resume and not args.audit_out.exists()
        assert args.audit_out.parent.resolve() == Path('runs').resolve()
    assert not any(key in os.environ for key in ('PYTORCH_MPS_HIGH_WATERMARK_RATIO','PYTORCH_ENABLE_MPS_FALLBACK'))
    with Path('runs/hf-mmmlu-global.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        free = int(re.search(r'free percentage:\s*(\d+)%', native(['memory_pressure','-Q']))[1])
        assert pressure() == 1 and free >= 55
        import torch
        from train import Corpus
        from ngram_placement import PLACEMENTS

        torch.set_num_threads(1)
        profile = load(args.profile)
        assert tuple(profile['arms']) == tuple(PLACEMENTS)
        assert Path(profile['output']).parent.resolve() == Path('runs').resolve()
        assert all(digest(path) == sha for path, sha in profile['source_hashes'].items())
        corpora = [Corpus(path, profile['seq']) for path in profile['data']]
        if args.arm:
            train(args, profile, corpora)
        else:
            audit(args, profile, corpora)


if __name__ == '__main__':
    main()
