"""Train a shared four-depth MTP draft on a frozen sparse target and local OWT."""
import argparse
from datetime import datetime, timedelta
import fcntl
import hashlib
import os
from pathlib import Path
import signal
import time

from qwen35_quality_data import digest, load, write_json
from run_dense_text_guarded import native, pressure


def losses(target, draft, tokens, profile):
    import torch
    from chunked_lm_loss import chunked_cross_entropy

    context = profile['context']
    base = tokens[:, :context]
    with torch.no_grad():
        hidden = target.embedding(base).repeat(1, 1, target.config.branches)
        for layer in target.layers:
            hidden, _, _ = layer(hidden, base, 'sparse')
    selection = None
    depth_losses = []
    for depth in range(1, profile['depths'] + 1):
        embedding = target.embedding(tokens[:, depth:context + depth])
        output, hidden, selection, kl, _ = draft(embedding, hidden, 'sparse', selection)
        labels = tokens[:, depth + 1:context + depth + 1]
        depth_losses.append(chunked_cross_entropy(output, target.head.weight, labels))
        if depth == 1:
            assert 0 < int((selection[:, -1] >= 0).sum()) < context
            index_loss = kl
    return torch.stack(depth_losses), index_loss


def run(profile_path, output, deadline, seconds, resume, updates):
    import numpy as np
    import resource
    import torch
    from model import Config, Model
    from mtp import MultiTokenPredictor
    from optim import RecipeOptimizer, ownership

    profile = load(profile_path)
    for name, sha in profile['files'].items():
        if digest(name) != sha:
            raise ValueError(f'Input changed: {name}')
    torch.set_num_threads(1)
    saved = torch.load(profile['target'], map_location='cpu', weights_only=True, mmap=True)
    assert saved['step'] == 8000 and saved['args']['stage'] == 'sparse'
    config = Config(**saved['config'])
    assert profile['context'] > config.budget + config.block
    # PLE keeps deterministic tensors outside the checkpoint state dictionary.
    target = Model(config)
    target.load_state_dict(saved['model'], assign=True)
    target.eval().requires_grad_(False)
    del saved

    def target_digest():
        fingerprint = hashlib.sha256()
        for name, parameter in target.named_parameters():
            fingerprint.update(name.encode())
            fingerprint.update(parameter.detach().numpy().tobytes())
        return fingerprint.hexdigest()

    frozen = target_digest()
    torch.manual_seed(profile['seed'])
    draft = MultiTokenPredictor(config)
    rng = torch.Generator().manual_seed(profile['seed'])
    optimizer = RecipeOptimizer(ownership(draft), profile['lr'])
    width = profile['context'] + profile['depths'] + 1
    arrays = {split: np.memmap(f'data/owt/{split}.bin', dtype='<u4', mode='r')
              for split in ('train', 'val')}

    def batch(split, offset):
        return torch.from_numpy(arrays[split][offset:offset+width].astype('int64')).unsqueeze(0)

    @torch.no_grad()
    def validation():
        draft.eval()
        values = [losses(target, draft, batch('val', offset), profile) for offset in profile['validation_offsets']]
        draft.train()
        return dict(depth_ce=torch.stack([ce for ce, _ in values]).mean(0).tolist(),
                    index_kl=torch.stack([kl for _, kl in values]).mean().item())

    checkpoint = output/'last.pt'
    status = output/'state.json'
    identity = dict(profile_sha256=digest(profile_path), profile=profile, config=vars(config),
                    frozen_target_parameters_sha256=frozen,
                    runner_sha256=digest(__file__))
    step = 0
    metrics = []
    if resume:
        prior = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert prior['identity'] == identity
        draft.load_state_dict(prior['draft'])
        optimizer.load_state_dict(prior['optimizer'])
        rng.set_state(prior['rng'])
        step, initial = prior['step'], prior['initial_validation']
        metrics = prior['metrics']
        assert len(metrics) == step
    else:
        output.mkdir(exist_ok=False)
        initial = validation()
    interrupted = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda number, frame: interrupted.append(number))
    start = time.monotonic()
    end = deadline - timedelta(seconds=600)

    def save(final_validation=None):
        temporary = output/'last.tmp'
        torch.save(dict(identity=identity, step=step, draft=draft.state_dict(),
                        optimizer=optimizer.state_dict(), rng=rng.get_state(),
                        initial_validation=initial, final_validation=final_validation,
                        metrics=metrics), temporary)
        temporary.replace(checkpoint)

    if not resume:
        save()
    reason = 'complete'
    first_step = step
    while step < profile['steps']:
        level = pressure()
        if interrupted or level >= 2:
            reason = 'interrupted' if interrupted else 'memory_pressure'
            break
        if datetime.now().astimezone() >= end or time.monotonic()-start >= seconds:
            reason = 'time_budget'
            break
        if updates is not None and step - first_step >= updates:
            reason = 'update_budget'
            break
        tick = time.monotonic()
        offset = int(torch.randint(len(arrays['train']) - width + 1, (), generator=rng))
        optimizer.zero_grad(set_to_none=True)
        ce, kl = losses(target, draft, batch('train', offset), profile)
        loss = ce.mean() + profile['index_kl_weight'] * kl
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(draft.parameters(), profile['clip'], error_if_nonfinite=True)
        optimizer.step()
        step += 1
        row = dict(step=step, offset=offset, depth_ce=ce.tolist(), index_kl=kl.item(),
                   grad_norm=norm.item(), seconds=time.monotonic()-tick,
                   pressure=level, swap=native(['sysctl', 'vm.swapusage']).strip(),
                   process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        metrics.append(row)
        save()
    final = validation() if reason == 'complete' and pressure() == 1 and datetime.now().astimezone() < end else None
    if final is not None:
        save(final)
    replay = torch.Generator().manual_seed(profile['seed'])
    assert all(int(torch.randint(len(arrays['train']) - width + 1, (), generator=replay)) == row['offset']
               for row in metrics)
    assert torch.equal(rng.get_state(), replay.get_state())
    assert target_digest() == frozen and all(parameter.grad is None for parameter in target.parameters())
    assert digest(profile['target']) == profile['files'][profile['target']]
    write_json(output/'metrics.json', metrics)
    write_json(status, dict(reason=reason, step=step, target_steps=profile['steps'],
                           initial_validation=initial, final_validation=final,
                           checkpoint_sha256=digest(checkpoint), elapsed=time.monotonic()-start,
                           rng_replayed=True, frozen_target_unchanged=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--deadline', type=datetime.fromisoformat, required=True)
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--updates', type=int, help='Stop after this many new updates and keep the declared target')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    now = datetime.now().astimezone()
    assert now + timedelta(seconds=660) < args.deadline <= now + timedelta(hours=2)
    assert 30 <= args.seconds <= 1100
    assert args.updates is None or args.updates > 0
    assert not any(name in os.environ for name in ('PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'PYTORCH_ENABLE_MPS_FALLBACK'))
    output = Path(load(args.profile)['output'])
    assert output.parent.resolve() == Path('runs').resolve()
    with Path('runs/hf-mmmlu-global.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        free = int(native(['memory_pressure', '-Q']).split('free percentage:')[-1].strip().rstrip('%'))
        if pressure() != 1 or free < 55:
            raise RuntimeError('Mac resource admission failed')
        run(args.profile, output, args.deadline, args.seconds, args.resume, args.updates)


if __name__ == '__main__':
    main()
