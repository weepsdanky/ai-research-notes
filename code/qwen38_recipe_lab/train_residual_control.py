"""Run one locked Mac residual arm with shared data and initialization."""
import argparse
from datetime import datetime, timedelta
import fcntl
import hashlib
import os
from pathlib import Path
import re
import signal
import time

from qwen35_quality_data import digest, load, write_json
from run_dense_text_guarded import native, pressure


TABLE5_ARMS = ('prenorm', 'mhc_static', 'mhc_dynamic', 'gr')
TABLE6_PROFILE_SHA256 = '4183b0c39c721c723bc2e2be0773d82673ec07afc83294ca7c23eae4a7f988c3'
TABLE6_TRAINER_SHA256 = 'ed8104bbf044b1d1a1ac40f3cd9e8d0e61c34bb517aabf0bccae070891d671bf'


def load_profile(path):
    profile = load(path)
    assert all(digest(name) == sha for name,sha in profile['source_hashes'].items())
    reuse = profile.get('reuse')
    if reuse:
        assert tuple(profile['arms']) == TABLE5_ARMS and set(reuse['arms']) == {'prenorm', 'gr'}
        assert digest(reuse['profile']) == TABLE6_PROFILE_SHA256
        original = load(reuse['profile'])
        for name in ('config','seed','steps','seq','batch','save_every','head_chunk','eval_batches',
                     'lr','balance_weight','clip','mix_probability','data'):
            assert profile[name] == original[name], name
    else:
        from residual_controls import VARIANTS

        assert tuple(profile['arms']) == tuple(VARIANTS)
    return profile


def make_model(profile, arm):
    from residual_controls import ResidualConfig, ResidualControl

    config = ResidualConfig(**profile['config'])
    if arm in ('mhc_static', 'mhc_dynamic'):
        from mhc_control import MHCResidualControl

        return MHCResidualControl(config, arm, profile['seed'])
    return ResidualControl(config, arm, profile['seed'])


def shared_initial_sha256(model):
    shared = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if name.startswith(('embedding.', 'head.')) or '.mix.' in name or '.moe.' in name:
            shared.update(name.encode())
            shared.update(tensor.numpy().tobytes())
    return shared.hexdigest()


def accepted_trainer_sha(identity):
    return identity['trainer_sha256'] == digest(__file__) or (
        identity['profile_sha256'] == TABLE6_PROFILE_SHA256
        and identity['trainer_sha256'] == TABLE6_TRAINER_SHA256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--deadline', required=True, type=datetime.fromisoformat)
    parser.add_argument('--seconds', type=int, default=900)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    now = datetime.now().astimezone()
    assert now + timedelta(seconds=660) < args.deadline <= now + timedelta(hours=2)
    assert 30 <= args.seconds <= 1100
    assert not any(key in os.environ for key in ('PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'PYTORCH_ENABLE_MPS_FALLBACK'))
    with Path('runs/hf-mmmlu-global.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        free = int(re.search(r'free percentage:\s*(\d+)%', native(['memory_pressure', '-Q']))[1])
        assert pressure() == 1 and free >= 55
        train(args)


def train(args):
    import resource
    import torch

    from audit_qwen35_training import finite
    from chunked_lm_loss import chunked_cross_entropy
    from optim import RecipeOptimizer, ownership
    from train import Corpus
    from train_qwen35_controls import balance_loss, evaluate

    profile = load_profile(args.profile)
    assert args.arm in profile['arms'] and args.arm not in profile.get('reuse', {}).get('arms', {})
    torch.set_num_threads(1)
    out = Path(profile['output'])/args.arm
    assert out.parent.parent.resolve() == Path('runs').resolve()
    checkpoint = out/'last.pt'
    status_path = out/'state.json'
    model = make_model(profile, args.arm)
    config = model.config
    optimizer = RecipeOptimizer(ownership(model), profile['lr'])
    rng = torch.Generator().manual_seed(profile['seed'])
    corpora = [Corpus(path, profile['seq']) for path in profile['data']]
    identity = dict(profile_sha256=digest(args.profile), trainer_sha256=digest(__file__),
                    shared_initial_sha256=shared_initial_sha256(model), arm=args.arm)
    if args.resume:
        if status_path.exists():
            previous = load(status_path)
            if previous['status'] != 'running':
                assert digest(checkpoint) == previous['checkpoint_sha256']
        saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
        source_identity = saved['identity']
        assert accepted_trainer_sha(source_identity)
        identity['trainer_sha256'] = source_identity['trainer_sha256']
        assert source_identity == identity
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.set_state(saved['rng'])
        step, history, initial = saved['step'], saved['history'], saved['initial_validation']
        assert step == len(history) < profile['steps'] or (
            step == len(history) == profile['steps'] and saved['final_validation'] is None)
        del saved
    else:
        out.mkdir(parents=True, exist_ok=False)
        step, history = 0, []
        initial = evaluate(model, corpora, profile, 'cpu')
    first = step

    def save(final=None):
        temp = checkpoint.with_suffix('.tmp')
        torch.save(dict(identity=identity, step=step, model=model.state_dict(),
                        optimizer=optimizer.state_dict(), rng=rng.get_state(),
                        history=history, initial_validation=initial, final_validation=final), temp)
        temp.replace(checkpoint)

    requests = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda number, frame: requests.append(number))
    start = time.monotonic()
    cutoff = args.deadline - timedelta(seconds=600)
    reason = 'complete'
    if not args.resume:
        save()
    write_json(status_path, dict(status='running', pid=os.getpid(), step=step, **identity))
    while step < profile['steps']:
        level = pressure()
        if requests or level >= 2:
            reason = 'interrupted' if requests else 'memory_pressure'
            break
        if datetime.now().astimezone() >= cutoff or time.monotonic()-start >= args.seconds:
            reason = 'time_budget'
            break
        tick = time.monotonic()
        index = int(torch.rand((), generator=rng).item() < profile['mix_probability'])
        inputs, labels = corpora[index].batch('train', profile['batch'], profile['seq'], rng, 'cpu')
        optimizer.zero_grad(set_to_none=True)
        hidden, routers = model.hidden(inputs)
        ce = chunked_cross_entropy(hidden, model.head.weight, labels, profile['head_chunk'])
        balance = balance_loss(routers, config.experts, config.top_k)
        (ce + profile['balance_weight']*balance).backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), profile['clip'], error_if_nonfinite=True)
        optimizer.step()
        step += 1
        history.append(dict(step=step, corpus=index, ce=ce.item(), balance=balance.item(),
                            grad_norm=norm.item(), seconds=time.monotonic()-tick, pressure=level,
                            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        if step % profile['save_every'] == 0:
            save()
            write_json(status_path, dict(status='running', pid=os.getpid(), step=step, **identity))
    final = None
    if reason == 'complete':
        if pressure() == 1 and not requests and datetime.now().astimezone() < cutoff:
            final = evaluate(model, corpora, profile, 'cpu')
        else:
            reason = 'validation_pending'
    finite(model.state_dict())
    finite(optimizer.state_dict())
    replay = torch.Generator().manual_seed(profile['seed'])
    for index, row in enumerate(history, 1):
        corpus = int(torch.rand((), generator=replay).item() < profile['mix_probability'])
        assert row['step'] == index and row['corpus'] == corpus
        torch.randint(len(corpora[corpus].arrays['train']) - profile['seq'], (profile['batch'],), generator=replay)
    assert torch.equal(rng.get_state(), replay.get_state())
    save(final)
    write_json(status_path, dict(status=reason, pid=os.getpid(), step=step, new_updates=step-first,
                                 target_steps=profile['steps'], tokens=step*profile['seq']*profile['batch'],
                                 checkpoint_sha256=digest(checkpoint), parameters=sum(p.numel() for p in model.parameters()),
                                 initial_validation=initial, final_validation=final,
                                 rng_replayed=True, swap=native(['sysctl', 'vm.swapusage']).strip(), **identity))


if __name__ == '__main__':
    main()
