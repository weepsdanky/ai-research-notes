"""Resolve audited Mac Table 5 and Table 10 checkpoints for public evaluation."""
from pathlib import Path

from qwen35_quality_data import digest, load

RESIDUAL = Path('profiles/residual_table5_mac.json')
BATCH_LR = Path('profiles/table10_mac_batch_lr.json')
EXPERIMENTS = {'table5': RESIDUAL, 'table10': BATCH_LR}


def checked_sources(profile):
    assert all(digest(path) == sha for path, sha in profile['source_hashes'].items())


def residual_binding(arm_name, audit_path):
    from train_residual_control import accepted_trainer_sha, make_model, shared_initial_sha256

    profile, audit = load(RESIDUAL), load(audit_path)
    checked_sources(profile)
    assert arm_name in profile['arms'] and audit['profile_sha256'] == digest(RESIDUAL)
    assert audit['auditor_sha256'] == digest('audit_residual_controls.py')
    arm = audit['arms'][arm_name]
    reused = profile['reuse']['arms']
    source_path = profile['reuse']['profile'] if arm_name in reused else RESIDUAL
    source = load(source_path)
    checked_sources(source)
    checkpoint = Path(source['output']) / arm_name / 'last.pt'
    assert arm['checkpoint'] == str(checkpoint) and arm['reused_table6'] == (arm_name in reused)
    state = load(checkpoint.with_name('state.json'))
    keys = ('profile_sha256','trainer_sha256','shared_initial_sha256','arm')
    identity = {key: state[key] for key in keys}
    assert accepted_trainer_sha(identity)
    assert identity['profile_sha256'] == digest(source_path) and identity['arm'] == arm_name
    assert identity['shared_initial_sha256'] == audit['shared_initial_sha256']
    assert state['status'] == 'complete' and state['step'] == arm['step'] == profile['steps']
    assert state['tokens'] == arm['tokens'] == profile['steps']*profile['seq']*profile['batch']
    assert state['final_validation'] == arm['final_validation']
    assert state['checkpoint_sha256'] == arm['checkpoint_sha256'] == digest(checkpoint)
    if arm_name in reused:
        assert arm['checkpoint_sha256'] == reused[arm_name]

    def construct():
        model = make_model(profile, arm_name)
        assert shared_initial_sha256(model) == identity['shared_initial_sha256']
        return model, identity

    return profile, arm, checkpoint, construct


def batch_lr_binding(arm_name, audit_path):
    from train_batch_lr import OUTPUT, sample_plan, settings

    schedule, audit = load(BATCH_LR), load(audit_path)
    checked_sources(schedule)
    profile = load(schedule['placement_profile'])
    checked_sources(profile)
    assert arm_name in schedule['arms']
    seq = schedule['sequence_tokens']
    width = schedule['arms'][arm_name]['batch']
    assert schedule['total_windows'] % width == 0
    steps = schedule['total_windows'] // width
    profile = dict(profile, seq=seq, steps=steps)
    binding, arm = audit['binding'], audit['arms'][arm_name]
    assert binding['profile_sha256'] == digest(BATCH_LR)
    assert binding['runner_sha256'] == digest('train_batch_lr.py')
    assert arm['status'] == 'complete' and arm['step'] == steps and arm['tokens'] == steps*width*seq
    checkpoint = OUTPUT / arm_name / 'last.pt'
    assert arm['checkpoint_sha256'] == digest(checkpoint)

    def construct():
        import torch
        from train import Corpus

        corpora = [Corpus(path, seq) for path in profile['data']]
        plan = sample_plan(schedule, corpora)
        model, expected, _ = settings(schedule, profile, plan)
        assert expected == binding
        identity = dict(expected, arm=arm_name, torch_version=str(torch.__version__))
        assert identity == arm['identity']
        return model, identity

    return profile, arm, checkpoint, construct


def training_input(experiment, arm_name, audit_path):
    assert experiment in EXPERIMENTS
    binder = residual_binding if experiment == 'table5' else batch_lr_binding
    profile, arm, checkpoint, construct = binder(arm_name, audit_path)
    assert arm['complete'] and arm['final_validation'] is not None
    return profile, arm, checkpoint, construct
