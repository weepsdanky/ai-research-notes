"""Independently check saved residual arms and compare complete paired budgets."""
import argparse
import fcntl
import math
from pathlib import Path
import statistics

from qwen35_quality_data import digest, load, write_json
from run_dense_text_guarded import pressure
from train_residual_control import accepted_trainer_sha, load_profile, make_model, shared_initial_sha256


def audit(profile_path):
    import torch
    from audit_qwen35_training import finite
    from optim import ownership
    from train import Corpus
    from train_qwen35_controls import evaluate

    torch.set_num_threads(1)
    profile = load_profile(profile_path)
    root = Path(profile['output'])
    reuse = profile.get('reuse', {})
    reused_arms = reuse.get('arms', {})
    reference_path = reuse.get('profile')
    reference = load_profile(reference_path) if reference_path else None
    corpora = [Corpus(path, profile['seq']) for path in profile['data']]
    token_counts = [len(corpus.arrays['train']) for corpus in corpora]
    rng = torch.Generator().manual_seed(profile['seed'])
    corpus_choices = []
    states = [rng.get_state()]
    for _ in range(profile['steps']):
        corpus = int(torch.rand((), generator=rng).item() < profile['mix_probability'])
        corpus_choices.append(corpus)
        torch.randint(token_counts[corpus]-profile['seq'], (profile['batch'],), generator=rng)
        states.append(rng.get_state())

    arms = {}
    shared_initial = None
    for arm in profile['arms']:
        reused = arm in reused_arms
        path = (Path(reference['output']) if reused else root)/arm
        if not path.exists() and not reused:
            continue
        assert pressure() == 1
        checkpoint = path/'last.pt'
        state = load(path/'state.json')
        assert state['status'] != 'running'
        assert digest(checkpoint) == state['checkpoint_sha256']
        if reused:
            assert state['checkpoint_sha256'] == reused_arms[arm]
        saved = torch.load(checkpoint, weights_only=True, map_location='cpu', mmap=True)
        identity = saved['identity']
        assert identity['arm'] == arm and identity['profile_sha256'] == digest(reference_path if reused else profile_path)
        assert accepted_trainer_sha(identity)
        assert all(state[key] == value for key, value in identity.items())
        step = saved['step']
        assert 0 <= step == state['step'] == len(saved['history']) <= profile['steps']
        assert [row['step'] for row in saved['history']] == list(range(1, step+1))
        assert [row['corpus'] for row in saved['history']] == corpus_choices[:step]
        assert torch.equal(states[step], saved['rng'])
        assert state['tokens'] == step*profile['seq']*profile['batch']
        for key in ('initial_validation','final_validation'):
            values = state[key]
            assert saved[key] == values
            if values is not None:
                assert set(values) == {f'val{i}/lm' for i in range(len(profile['data']))}
                assert all(math.isfinite(value) for value in values.values())
        model = make_model(profile, arm)
        shared = shared_initial_sha256(model)
        assert shared == identity['shared_initial_sha256']
        if shared_initial is None:
            shared_initial = shared
        assert shared_initial == shared
        model.load_state_dict(saved['model'], strict=True)
        finite(saved['model'])
        finite(saved['optimizer'])
        plans = [(name, pieces) for name, _, pieces in ownership(model)]
        groups = [(group['name'], group['pieces']) for group in saved['optimizer']['param_groups']]
        assert plans == groups
        history = saved['history']
        assert all(math.isfinite(row[key]) for row in history for key in ('ce','balance','grad_norm','seconds'))
        last = history[-1] if history else None
        complete = step == profile['steps'] and state['status'] == 'complete'
        assert complete == (saved['final_validation'] is not None)
        assert complete or not reused
        if complete:
            replayed = evaluate(model, corpora, profile, 'cpu')
            assert replayed == saved['final_validation']
        arms[arm] = dict(step=step, complete=complete, tokens=state['tokens'],
                         checkpoint_sha256=state['checkpoint_sha256'],
                         initial_validation=state['initial_validation'], final_validation=state['final_validation'],
                         last_recorded_training_ce=last['ce'] if last else None,
                         last_32_training_ce_mean=statistics.mean(row['ce'] for row in history[-32:]) if history else None,
                         median_update_seconds=statistics.median(row['seconds'] for row in history) if history else None,
                         max_process_rss_bytes=max((row['process_peak_rss_bytes'] for row in history), default=0))
        if reuse:
            arms[arm].update(checkpoint=str(checkpoint), reused_table6=reused)
        del model,saved
    complete = len(arms) == len(profile['arms']) and all(arm['complete'] for arm in arms.values())
    return dict(status='complete_mac_ablation' if complete else 'partial_arms',
                profile_sha256=digest(profile_path), shared_initial_sha256=shared_initial,
                auditor_sha256=digest(__file__), arms=arms)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists() and args.out.parent.resolve() == Path('runs').resolve()
    with Path('runs/hf-mmmlu-global.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(args.out, audit(args.profile))
