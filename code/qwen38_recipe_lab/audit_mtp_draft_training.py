"""Audit a Mac MTP draft's saved training state without loading its backbone."""
import argparse
from collections import Counter
import math
from pathlib import Path

from qwen35_quality_data import digest, load, write_json
from run_control_quality import memory, normal

PROFILES = {
    'qsa': Path('profiles/mtp_frozen_draft_mac.json'),
    'full': Path('profiles/mtp_frozen_dense_draft_mac.json'),
}


def audit(attention, allow_partial=False, snapshot=None):
    profile_path = PROFILES[attention]
    profile = load(profile_path)
    out = Path(snapshot or profile['output'])
    checkpoint, state_path, metrics_path = (out / name for name in ('last.pt', 'state.json', 'metrics.json'))
    if not state_path.exists():
        raise ValueError(f'{attention} has no saved draft training state')
    state = load(state_path)
    complete = state['reason'] == 'complete'
    assert 0 <= state['step'] <= state['target_steps'] == profile['steps']
    if complete:
        assert state['step'] == state['target_steps'] and state['final_validation'] is not None
    else:
        assert allow_partial and state['reason'] in ('time_budget', 'memory_pressure', 'interrupted')
        assert state['final_validation'] is None
    assert state['checkpoint_sha256'] == digest(checkpoint)
    assert state['rng_replayed'] and state['frozen_target_unchanged']
    assert all(digest(path) == sha for path, sha in profile['files'].items())
    runner = 'train_mtp_dense_draft.py' if attention == 'full' else 'train_mtp_draft.py'
    before = memory()
    assert normal(before)

    import torch
    from model import Config
    from mtp import MultiTokenPredictor
    from optim import ownership, view

    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    ident = saved['identity']
    assert ident['profile_sha256'] == digest(profile_path) and ident['profile'] == profile
    assert ident['runner_sha256'] == digest(runner)
    assert saved['step'] == state['step']
    for key in ('initial_validation', 'final_validation'):
        validation = saved[key]
        assert validation == state[key]
        if validation is None:
            assert key == 'final_validation' and not complete
            continue
        assert len(validation['depth_ce']) == profile['depths']
        assert all(math.isfinite(loss) and loss >= 0 for loss in (*validation['depth_ce'], validation['index_kl']))
        if attention == 'full':
            assert validation['index_kl'] == 0
    metrics = load(metrics_path)
    assert saved['metrics'] == metrics
    assert [row['step'] for row in metrics] == list(range(1, saved['step'] + 1))
    assert all(len(row['depth_ce']) == profile['depths'] and
               all(math.isfinite(x) and x >= 0 for x in (*row['depth_ce'], row['index_kl'], row['grad_norm']))
               for row in metrics)
    width = profile['context'] + profile['depths'] + 1
    offsets = [row['offset'] for row in metrics]
    train_bytes = Path('data/owt/train.bin').stat().st_size
    assert train_bytes == 4 * load('data/owt/meta.json')['files']['train']['tokens']
    rng = torch.Generator().manual_seed(profile['seed'])
    assert all(int(torch.randint(train_bytes // 4 - width + 1, (), generator=rng)) == offset for offset in offsets)
    assert torch.equal(rng.get_state(), saved['rng'])
    if attention == 'full':
        assert state['frozen_unused_indexer'] and state['matched_sparse_sampling_offsets']
        assert digest(profile['paired_sparse_metrics']) == profile['paired_sparse_metrics_sha256']
        assert offsets == [row['offset'] for row in load(profile['paired_sparse_metrics'])[:saved['step']]]
        assert all(row['index_kl'] == 0 for row in metrics)

    torch.manual_seed(profile['seed'])
    initial = MultiTokenPredictor(Config(**ident['config']))
    stored = saved['draft']
    assert set(stored) == set(initial.state_dict())
    assert all(old.shape == stored[name].shape and old.dtype == stored[name].dtype and
               torch.isfinite(stored[name]).all() for name, old in initial.state_dict().items())
    changed = {name for name, old in initial.named_parameters() if not torch.equal(old, stored[name])}
    indexer = {name for name, _ in initial.named_parameters() if '.indexer.' in name}
    assert indexer and bool(changed - indexer) == bool(saved['step'])
    assert bool(changed & indexer) == (attention == 'qsa' and saved['step'] > 0)
    plans = ownership(initial)
    optimizer = saved['optimizer']
    groups = optimizer['param_groups']
    initialized, owned = set(), set()
    kinds = Counter()
    for (name, param, pieces), group in zip(plans, groups, strict=True):
        key, = group['params']
        assert key not in owned and group['name'] == name and group['pieces'] == pieces
        assert group['lr'] == profile['lr']
        owned.add(key)
        if key not in optimizer['state']:
            continue
        initialized.add(name)
        for spec, buffers in zip(pieces, optimizer['state'][key]['pieces'], strict=True):
            keys = {'momentum'} if spec['kind'] == 'muon' else {'m', 'v', 'step'}
            assert buffers.keys() == keys
            if 'step' in keys:
                assert 1 <= buffers['step'] <= saved['step']
            expected_shape = view(param, spec).shape
            assert all(buffers[k].shape == expected_shape and torch.isfinite(buffers[k]).all()
                       for k in keys - {'step'})
            kinds[spec['kind']] += 1
    assert set(optimizer['state']) <= owned and changed <= initialized
    assert bool(indexer & initialized) == (attention == 'qsa' and saved['step'] > 0)

    assert state == load(state_path) and state['checkpoint_sha256'] == digest(checkpoint)
    return dict(attention=attention, steps=saved['step'], target_steps=state['target_steps'],
                complete=complete, stopped=state['reason'], checkpoint=str(checkpoint),
                checkpoint_sha256=state['checkpoint_sha256'], state_sha256=digest(state_path),
                metrics_sha256=digest(metrics_path), profile_sha256=digest(profile_path),
                target_checkpoint_sha256=profile['files'][profile['target']],
                changed_parameter_tensors=len(changed), initialized_optimizer_tensors=len(initialized),
                optimizer_pieces=kinds, changed_indexer=sorted(changed & indexer),
                initial_validation=state['initial_validation'], final_validation=state['final_validation'],
                exact_rng_replay=True, resource_samples=[before, memory()], auditor_sha256=digest(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attention', required=True, choices=PROFILES)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    assert args.out.parent.resolve() == Path('runs').resolve() and not args.out.exists()
    write_json(args.out, audit(args.attention, args.allow_partial, args.snapshot))
