"""Train the frozen Full Attention arm for the Mac Table 4 comparison."""
import argparse
from datetime import datetime, timedelta
import fcntl
import hashlib
import os
from pathlib import Path
import resource
import signal
import time

from qwen35_quality_data import digest, load, write_json
from run_control_quality import LOCK, memory, normal
from run_dense_text_guarded import native, pressure

PROFILE = Path('profiles/mtp_frozen_dense_draft_mac.json')


def preflight():
    profile = load(PROFILE)
    sparse = load(profile['paired_sparse_profile'])
    assert digest(profile['paired_sparse_profile']) == profile['paired_sparse_profile_sha256']
    keys = ('context', 'depths', 'steps', 'seed', 'lr', 'clip', 'validation_offsets')
    assert all(profile[key] == sparse[key] for key in keys)
    assert profile['stage'] == 'pretrain' and profile['index_kl_weight'] == 0
    assert all(digest(path) == sha for path, sha in profile['files'].items())
    assert digest(profile['target_audit']) == profile['target_audit_sha256']
    target = load(profile['target_audit'])
    assert target['checkpoint'] == profile['target'] and target['checkpoint_sha256'] == profile['files'][profile['target']]
    assert target['phase_complete'] and target['step'] == target['target_steps'] == 8000
    for key in ('paired_sparse_metrics', 'paired_sparse_draft'):
        assert digest(profile[key]) == profile[f'{key}_sha256']
    sparse_rows = load(profile['paired_sparse_metrics'])
    assert [row['step'] for row in sparse_rows] == list(range(1, profile['steps'] + 1))
    return profile, [row['offset'] for row in sparse_rows]


def train(profile, offsets, args):
    import numpy as np
    import torch

    from chunked_lm_loss import chunked_cross_entropy
    from model import Config, Model
    from mtp import MultiTokenPredictor
    from optim import RecipeOptimizer, ownership

    torch.set_num_threads(1)
    target_path = profile['target']
    saved = torch.load(target_path, map_location='cpu', weights_only=True, mmap=True)
    assert saved['step'] == 8000 and saved['args']['stage'] == 'pretrain'
    sparse_draft = torch.load(profile['paired_sparse_draft'], map_location='cpu', weights_only=True)
    assert saved['config'] == sparse_draft['identity']['config']
    assert sparse_draft['step'] == profile['steps']
    config = Config(**saved['config'])
    target = Model(config)
    target.load_state_dict(saved['model'], assign=True)
    target.eval().requires_grad_(False)
    del saved, sparse_draft

    def fingerprint():
        h = hashlib.sha256()
        for name, parameter in target.named_parameters():
            h.update(name.encode())
            h.update(parameter.detach().numpy().tobytes())
        return h.hexdigest()

    frozen = fingerprint()
    torch.manual_seed(profile['seed'])
    draft = MultiTokenPredictor(config)
    optimizer = RecipeOptimizer(ownership(draft), profile['lr'])
    context = profile['context']
    width = context + profile['depths'] + 1
    arrays = {split: np.memmap(f'data/owt/{split}.bin', dtype='<u4', mode='r') for split in ('train', 'val')}
    rng = torch.Generator().manual_seed(profile['seed'])
    indexer = tuple(draft.layer.mix.indexer.parameters())
    initial_indexer = [p.detach().clone() for p in indexer]

    def losses(split, offset):
        tokens = torch.from_numpy(arrays[split][offset:offset + width].astype('int64')).unsqueeze(0)
        base = tokens[:, :context]
        with torch.no_grad():
            hidden = target.embedding(base).repeat(1, 1, config.branches)
            for layer in target.layers:
                hidden, _, _ = layer(hidden, base, 'pretrain')
        values = []
        for depth in range(1, profile['depths'] + 1):
            embedding = target.embedding(tokens[:, depth:context + depth])
            output, hidden, selection, kl, _ = draft(embedding, hidden, 'pretrain')
            assert selection is None and kl.item() == 0
            values.append(chunked_cross_entropy(output, target.head.weight, tokens[:, depth + 1:context + depth + 1]))
        return torch.stack(values)

    @torch.no_grad()
    def validate():
        draft.eval()
        values = [losses('val', offset) for offset in profile['validation_offsets']]
        draft.train()
        return dict(depth_ce=torch.stack(values).mean(0).tolist(), index_kl=0.0)

    output = Path(profile['output'])
    checkpoint = output / 'last.pt'
    identity = dict(profile_sha256=digest(PROFILE), profile=profile, config=vars(config),
                    frozen_target_parameters_sha256=frozen, runner_sha256=digest(__file__))
    step, metrics = 0, []
    if args.resume:
        prior = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert prior['identity'] == identity and prior['final_validation'] is None
        draft.load_state_dict(prior['draft'])
        optimizer.load_state_dict(prior['optimizer'])
        rng.set_state(prior['rng'])
        step, initial, metrics = prior['step'], prior['initial_validation'], prior['metrics']
        assert len(metrics) == step and [row['offset'] for row in metrics] == offsets[:step]
        assert all(torch.equal(p, old) for p, old in zip(indexer, initial_indexer, strict=True))
    else:
        assert not output.exists()
        initial = validate()
        output.mkdir(exist_ok=False)

    def save(final=None):
        temporary = output / 'last.tmp'
        torch.save(dict(identity=identity, step=step, draft=draft.state_dict(), optimizer=optimizer.state_dict(),
                        rng=rng.get_state(), initial_validation=initial, final_validation=final,
                        metrics=metrics), temporary)
        temporary.replace(checkpoint)

    if not args.resume:
        save()
    interrupted = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda number, frame: interrupted.append(number))
    start = time.monotonic()
    end = args.deadline - timedelta(seconds=600)
    reason = 'complete'
    while step < profile['steps']:
        level = pressure()
        if interrupted or level != 1:
            reason = 'interrupted' if interrupted else 'memory_pressure'
            break
        if datetime.now().astimezone() >= end or time.monotonic() - start >= args.seconds:
            reason = 'time_budget'
            break
        tick = time.monotonic()
        offset = int(torch.randint(len(arrays['train']) - width + 1, (), generator=rng))
        assert offset == offsets[step]
        optimizer.zero_grad(set_to_none=True)
        ce = losses('train', offset)
        ce.mean().backward()
        assert all(p.grad is None for p in indexer)
        norm = torch.nn.utils.clip_grad_norm_(draft.parameters(), profile['clip'], error_if_nonfinite=True)
        optimizer.step()
        step += 1
        metrics.append(dict(step=step, offset=offset, depth_ce=ce.tolist(), index_kl=0.0,
                            grad_norm=norm.item(), seconds=time.monotonic() - tick,
                            pressure=level, swap=native(['sysctl', 'vm.swapusage']).strip(),
                            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        save()
    final = None
    if reason == 'complete':
        if pressure() != 1:
            reason = 'memory_pressure'
        elif datetime.now().astimezone() >= end:
            reason = 'time_budget'
        else:
            final = validate()
            save(final)
    replay = torch.Generator().manual_seed(profile['seed'])
    assert all(int(torch.randint(len(arrays['train']) - width + 1, (), generator=replay)) == row['offset']
               for row in metrics)
    assert torch.equal(rng.get_state(), replay.get_state())
    assert all(torch.equal(p, old) for p, old in zip(indexer, initial_indexer, strict=True))
    assert fingerprint() == frozen and all(p.grad is None for p in target.parameters())
    assert digest(target_path) == profile['files'][target_path]
    write_json(output / 'metrics.json', metrics)
    write_json(output / 'state.json', dict(reason=reason, step=step, target_steps=profile['steps'],
                                           initial_validation=initial, final_validation=final,
                                           checkpoint_sha256=digest(checkpoint), elapsed=time.monotonic() - start,
                                           rng_replayed=True, frozen_target_unchanged=True,
                                           frozen_unused_indexer=True, matched_sparse_sampling_offsets=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-out', type=Path)
    parser.add_argument('--deadline', type=datetime.fromisoformat)
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    profile, offsets = preflight()
    if args.prepare_out:
        assert not args.resume and args.deadline is None and not args.prepare_out.exists()
        assert args.prepare_out.parent.resolve() == Path('runs').resolve()
        write_json(args.prepare_out, dict(status='verified_full_attention_training_inputs',
                                           profile_sha256=digest(PROFILE), runner_sha256=digest(__file__),
                                           scheduled_offsets=len(offsets), training_started=False))
        return
    now = datetime.now().astimezone()
    assert args.deadline and now + timedelta(seconds=660) < args.deadline <= now + timedelta(hours=2)
    assert 30 <= args.seconds <= 1100
    assert Path(profile['output']).parent.resolve() == Path('runs').resolve()
    assert not any(name in os.environ for name in ('PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'PYTORCH_ENABLE_MPS_FALLBACK'))
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert normal(memory())
        train(profile, offsets, args)


if __name__ == '__main__':
    main()
