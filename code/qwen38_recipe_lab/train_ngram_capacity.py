"""Train or audit the fixed-total and growing-total Mac N-gram controls."""
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

from qwen35_quality_data import digest, load, write_json
from run_dense_text_guarded import native, pressure
from train_ngram_placement import fingerprint

FAMILIES = {'table8': 'table8_fixed_total', 'table9': 'table9_growing_total'}
PROFILE = Path('profiles/ngram_table8_9_mac_training.json')
RUNS = Path('runs')
OUTPUT = RUNS / 'ngram-capacity-mac'


def model_and_identity(profile, capacity, family, arm, entry):
    import torch

    from model import Config
    from ngram_scaling import NGramScaling

    model = NGramScaling(Config(**capacity['config']), arm, entry, capacity['position'], profile['seed'])
    expected = (profile['baseline_parameters']
                + (entry['experts'] - capacity['config']['experts']) * capacity['expert_increment_including_router']
                + entry['table']['parameters'] + entry['added_ple_adapter_parameters'])
    if family == 'table8':
        assert expected == profile['table8_parameters']
    assert sum(parameter.numel() for parameter in model.parameters()) == expected

    def shared(name):
        return '.ple.' not in name and (family == 'table9' or not any(
            part in name for part in ('.moe.experts.', '.moe.router.')))

    identity = dict(family=family, arm=arm, profile_sha256=digest(PROFILE), runner_sha256=digest(__file__),
                    device='cpu', precision='float32', torch_version=str(torch.__version__),
                    parameters=expected, experts=entry['experts'], table_rows=entry['table']['rows'],
                    ngram_parameters=model.ngram_parameters(), shared_initial_sha256=fingerprint(model, shared),
                    frozen_indexer_sha256=fingerprint(model, lambda name: '.indexer.' in name))
    return model, identity


def train(args, profile, corpora, model, identity):
    import torch

    from audit_qwen35_training import finite
    from chunked_lm_loss import chunked_cross_entropy
    from optim import RecipeOptimizer, ownership
    from train_qwen35_controls import balance_loss, evaluate

    out = OUTPUT / args.family / args.arm
    checkpoint, state_path = out / 'last.pt', out / 'state.json'
    optimizer = RecipeOptimizer(ownership(model), profile['lr'])
    rng = torch.Generator().manual_seed(profile['seed'])
    recovered_final = None
    if args.resume:
        state = load(state_path) if state_path.exists() else {'status': 'running'}
        assert state['status'] != 'complete'
        if state['status'] != 'running':
            assert digest(checkpoint) == state['checkpoint_sha256']
        saved = torch.load(checkpoint, weights_only=True, map_location='cpu', mmap=True)
        assert saved['identity'] == identity
        recovered_final = saved['final_validation']
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.set_state(saved['rng'])
        history, initial = saved['history'], saved['initial_validation']
        assert 0 <= saved['step'] == len(history) <= profile['steps']
        del saved
    else:
        assert not checkpoint.exists() and not state_path.exists()
        history, initial = [], evaluate(model, corpora, profile, 'cpu')
        out.mkdir(parents=True, exist_ok=True)
    first = len(history)

    def record(status, final=None):
        write_json(state_path, dict(identity, status=status, pid=os.getpid(), step=len(history),
                                   target_steps=profile['steps'], new_updates=len(history)-first,
                                   tokens=len(history)*profile['seq']*profile['batch'],
                                   checkpoint_sha256=digest(checkpoint), initial_validation=initial,
                                   final_validation=final, swap=native(['sysctl','vm.swapusage']).strip()))

    def save(status, final=None):
        temp = checkpoint.with_suffix('.tmp')
        torch.save(dict(identity=identity, step=len(history), model=model.state_dict(),
                        optimizer=optimizer.state_dict(), rng=rng.get_state(), history=history,
                        initial_validation=initial, final_validation=final), temp)
        temp.replace(checkpoint)
        record(status, final)

    if recovered_final is not None:
        assert first == profile['steps']
        record('complete', recovered_final)
        return

    requests = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda number, frame: requests.append(number))
    stop = time.monotonic() + args.seconds
    cutoff = args.deadline - timedelta(seconds=600)

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
    while len(history) < profile['steps']:
        reason = stop_reason()
        if reason:
            break
        tick = time.monotonic()
        corpus = int(torch.rand((), generator=rng).item() < profile['mix_probability'])
        inputs, labels = corpora[corpus].batch('train', profile['batch'], profile['seq'], rng, 'cpu')
        optimizer.zero_grad(set_to_none=True)
        hidden, routers = model.hidden(inputs)
        ce = chunked_cross_entropy(hidden, model.head.weight, labels, profile['head_chunk'])
        balance = balance_loss(routers, model.config.experts, model.config.top_k)
        (ce + profile['balance_weight'] * balance).backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), profile['clip'], error_if_nonfinite=True)
        optimizer.step()
        history.append(dict(step=len(history)+1, corpus=corpus, ce=ce.item(), balance=balance.item(),
                            grad_norm=norm.item(), seconds=time.monotonic()-tick,
                            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        if len(history) % profile['checkpoint_every'] == 0:
            save('running')
            print(f"{args.family}/{args.arm} step {len(history)}/{profile['steps']} loss {ce.item():.5f}", flush=True)

    reason = reason or stop_reason()
    final = None
    if reason is None:
        final = evaluate(model, corpora, profile, 'cpu')
        reason = 'complete'
    finite(model.state_dict())
    finite(optimizer.state_dict())
    assert fingerprint(model, lambda name: '.indexer.' in name) == identity['frozen_indexer_sha256']
    save(reason, final)


def audit_arm(profile, corpora, model, identity, out):
    import torch

    from audit_qwen35_training import finite
    from optim import ownership
    from train_qwen35_controls import evaluate

    state = load(out / 'state.json')
    checkpoint = out / 'last.pt'
    assert state['status'] != 'running' and digest(checkpoint) == state['checkpoint_sha256']
    saved = torch.load(checkpoint, weights_only=True, map_location='cpu', mmap=True)
    assert saved['identity'] == identity and all(state[key] == value for key, value in identity.items())
    model.load_state_dict(saved['model'])
    assert fingerprint(model, lambda name: '.indexer.' in name) == identity['frozen_indexer_sha256']
    finite(saved['model'])
    finite(saved['optimizer'])
    assert [(name, pieces) for name, _, pieces in ownership(model)] == [
        (group['name'], group['pieces']) for group in saved['optimizer']['param_groups']]
    rng = torch.Generator().manual_seed(profile['seed'])
    step, history = saved['step'], saved['history']
    assert 0 <= step == state['step'] == len(history) <= profile['steps']
    for index, row in enumerate(history, 1):
        corpus = int(torch.rand((), generator=rng).item() < profile['mix_probability'])
        assert row['step'] == index and row['corpus'] == corpus
        assert all(math.isfinite(row[key]) for key in ('ce', 'balance', 'grad_norm', 'seconds'))
        torch.randint(len(corpora[corpus].arrays['train'])-profile['seq'], (profile['batch'],), generator=rng)
    assert torch.equal(saved['rng'], rng.get_state())
    assert state['tokens'] == step*profile['seq']*profile['batch']
    for key in ('initial_validation', 'final_validation'):
        values = state[key]
        assert values == saved[key]
        if values is not None:
            assert set(values) == {f'val{i}/lm' for i in range(len(corpora))}
            assert all(math.isfinite(value) for value in values.values())
    complete = state['status'] == 'complete'
    assert complete == (saved['final_validation'] is not None)
    if complete:
        assert step == profile['steps']
        assert evaluate(model, corpora, profile, 'cpu') == state['final_validation']
    return dict(state, complete=complete, data_rng_replayed=True, frozen_indexer_verified=True,
                median_update_seconds=statistics.median(row['seconds'] for row in history) if history else None,
                last_32_training_ce_mean=statistics.mean(row['ce'] for row in history[-32:]) if history else None,
                max_process_rss_bytes=max((row['process_peak_rss_bytes'] for row in history), default=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=FAMILIES, required=True)
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
        assert args.audit_out.parent.resolve() == RUNS.resolve()
    assert not any(key in os.environ for key in ('PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'PYTORCH_ENABLE_MPS_FALLBACK'))

    with (RUNS / 'hf-mmmlu-global.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        free = int(re.search(r'free percentage:\s*(\d+)%', native(['memory_pressure', '-Q']))[1])
        assert pressure() == 1 and free >= 55
        import torch
        from train import Corpus

        torch.set_num_threads(1)
        profile = load(PROFILE)
        assert all(digest(path) == sha for path, sha in profile['source_hashes'].items())
        capacity = load(profile['capacity_profile'])
        entries = {f"s{entry['paper_scale_label']}": entry for entry in capacity[FAMILIES[args.family]]}
        corpora = [Corpus(path, profile['seq']) for path in profile['data']]
        if args.arm:
            if args.arm not in entries:
                parser.error(f"arms for {args.family}: {', '.join(entries)}")
            model, identity = model_and_identity(profile, capacity, args.family, args.arm, entries[args.arm])
            train(args, profile, corpora, model, identity)
            return
        results, shared = {}, None
        for arm, entry in entries.items():
            out = OUTPUT / args.family / arm
            if not out.exists():
                continue
            assert pressure() == 1
            model, identity = model_and_identity(profile, capacity, args.family, arm, entry)
            shared = shared or identity['shared_initial_sha256']
            assert shared == identity['shared_initial_sha256']
            results[arm] = audit_arm(profile, corpora, model, identity, out)
            del model
        complete = len(results) == len(entries) and all(row['complete'] for row in results.values())
        write_json(args.audit_out, dict(status='complete_mac_capacity_training' if complete else 'partial_arms',
                                       family=args.family, profile_sha256=digest(PROFILE), runner_sha256=digest(__file__),
                                       shared_initial_sha256=shared, arms=results))


if __name__ == '__main__':
    main()
