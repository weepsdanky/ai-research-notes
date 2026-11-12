"""Bind the five Mac Table 4 jobs to their saved trace audits."""
import argparse
import hashlib
import subprocess
from pathlib import Path

from mtp_public_data import PROFILE
from qwen35_quality_data import digest, load, write_json

RUNS = Path('runs')
DRAFT = RUNS / 'mtp-frozen-draft-mac/last.pt'
TRAINING_AUDIT = RUNS / 'round36-mtp-draft-004-state.json'
PROMPTS = RUNS / 'round39-mtp-full-public-prompts-v2-check.json'
REPLAY = RUNS / 'round39-mtp-public-trace-replay-check.json'
MT_AUDIT = RUNS / 'round36-mtp-mt-bench-audit.json'
MT_PROFILE = Path('profiles/mtp_acceptance_mac_mt_bench.json')
ARCHIVE = '58ba0f53318a3f7424ceb3c044a0eb7dc6fa76ce'
ARCHIVED_SOURCES = {'evaluate_mtp_public.py', 'evaluate_mtp_draft.py'}


def archived_digest(path):
    prefix = subprocess.check_output(['git', 'rev-parse', '--show-prefix'], text=True).strip()
    source = subprocess.check_output(['git', 'show', f'{ARCHIVE}:{prefix}{path}'])
    return hashlib.sha256(source).hexdigest()


def mt_bench(draft_sha, target_sha):
    audit = load(MT_AUDIT)
    manifest = load(MT_PROFILE)
    raw = Path(audit['predictions'])
    predictions = load(raw)
    replay = load(REPLAY)
    verified = replay['sources'][str(raw)]
    assert audit['status'] == 'verified_all_mt_bench_turns_mac_acceptance'
    assert replay['profile_sha256'] == digest(PROFILE)
    assert replay['auditor_sha256'] == archived_digest('evaluate_mtp_public.py')
    assert digest(raw) == audit['predictions_sha256'] == verified['sha256']
    assert digest(MT_PROFILE) == audit['manifest_sha256'] == predictions['prompts_sha256']
    assert digest(manifest['tokenizer']) == manifest['tokenizer_sha256']
    assert all(digest(path) == sha for path, sha in manifest['source_hashes'].items())
    assert predictions['checkpoint_sha256'] == draft_sha
    assert predictions['target_checkpoint_sha256'] == target_sha
    assert audit['conversations'] == manifest['conversation_count']
    target = manifest['conversation_count'] * manifest['turns_per_conversation']
    assert audit['turns'] == target == len(predictions['items']) == verified['items']
    assert predictions['target_items'] == verified['totals']['items'] == target
    assert predictions['status'] == 'complete_mac_mt_bench'
    totals = verified['totals']
    acceptance = {}
    for mode, counts in totals['modes'].items():
        scored = audit['acceptance']['all'][mode]
        assert all(scored[key] == counts[key] for key in ('rounds', 'accepted', 'clipped_rounds'))
        acceptance[mode] = 1 + counts['accepted'] / counts['rounds']
        assert acceptance[mode] == scored['mean_acceptance_length']
    return dict(status='complete_audited', items=target, target=target, totals=totals, acceptance=acceptance,
                output_sha256=verified['sha256'], audit_sha256=digest(MT_AUDIT),
                replay_sha256=digest(REPLAY), verification='batched_archive')


def coverage(verification='legacy'):
    profile = load(PROFILE)
    profile_sha = digest(PROFILE)
    training = load(profile['draft_training_profile'])
    draft_sha, target_sha = digest(DRAFT), digest(training['target'])
    trained = load(TRAINING_AUDIT)
    assert trained['reason'] == 'complete' and trained['step'] == trained['target_steps'] == training['steps']
    assert trained['checkpoint_sha256'] == draft_sha
    assert trained['rng_replayed'] and trained['frozen_target_unchanged']
    assert all(digest(path) == sha for path, sha in profile['source_hashes'].items())
    prompts = load(PROMPTS)
    assert prompts['profile_sha256'] == profile_sha
    assert prompts['preparer_sha256'] == digest('mtp_public_data.py')
    jobs = {}
    expected = {job: (spec['items'], prompts['jobs'][job]['content_sha256'])
                for job, spec in profile['jobs'].items()}
    for job, (target, _) in expected.items():
        assert prompts['jobs'][job]['items'] == target
    if verification == 'serial':
        mt = load(MT_PROFILE)
        assert mt['tokenizer_sha256'] == profile['tokenizer_sha256']
        assert mt['new_tokens'] == profile['new_tokens']
        assert all(digest(path) == sha for path, sha in mt['source_hashes'].items())
        expected['mt-bench'] = (mt['conversation_count'] * mt['turns_per_conversation'], digest(MT_PROFILE))
    else:
        jobs['mt-bench'] = mt_bench(draft_sha, target_sha)
    prefix = 'mtp-public-serial' if verification == 'serial' else 'mtp-public'
    for job, (target, prompts_sha) in expected.items():
        raw = RUNS / f'{prefix}-{job}.jsonl'
        meta = Path(f'{raw}.meta.json')
        audit = raw.with_name(f'{raw.stem}-audit.json')
        if not meta.exists():
            assert not audit.exists()
            jobs[job] = dict(status='not_started', items=0, target=target)
            continue
        has_audit = audit.exists()
        state = load(meta)
        identity = state['identity']
        assert identity['job'] == job and state['target'] == target
        assert identity['profile_sha256'] == profile_sha
        assert identity['prompt_content_sha256'] == prompts_sha
        assert identity['draft_checkpoint_sha256'] == draft_sha
        assert identity['target_checkpoint_sha256'] == target_sha
        assert identity.get('verification', 'legacy') == verification
        if job == 'mt-bench':
            state_path = DRAFT.with_name('state.json')
            assert identity['attention'] == 'qsa' and identity['training_state_sha256'] == digest(state_path)
            assert load(state_path) == trained
        for path, sha in identity['sources'].items():
            source_sha = archived_digest(path) if verification == 'legacy' and path in ARCHIVED_SOURCES else digest(path)
            assert source_sha == sha
        items = state['totals']['items']
        assert 0 <= items <= target
        if not has_audit:
            status = 'awaiting_audit' if state['status'] == 'complete' else state['status']
            if verification == 'legacy' and job == 'gsm8k':
                stopped = load(RUNS / 'round40-mtp-gsm8k-prefix652-audit.json')
                assert not stopped['complete'] and stopped['totals']['items'] == items
                assert stopped['output_sha256'] == digest(raw)
                status = 'audited_prefix_after_divergence'
            jobs[job] = dict(status=status, items=items, target=target)
            continue
        checked = load(audit)
        assert checked['complete'] and state['status'] == 'complete' and items == target
        assert checked['target'] == target and checked['identity'] == identity
        assert checked['totals'] == state['totals'] and checked['output_sha256'] == digest(raw)
        for mode, counts in checked['totals']['modes'].items():
            length = 1 + counts['accepted'] / counts['rounds'] if counts['rounds'] else None
            assert checked['acceptance'][mode] == length
        jobs[job] = dict(status='complete_audited', items=items, target=target, totals=checked['totals'],
                         acceptance=checked['acceptance'], output_sha256=checked['output_sha256'],
                         audit_sha256=digest(audit), verification=verification)
    return dict(complete=all(record['status'] == 'complete_audited' for record in jobs.values()),
                public_verification=verification, jobs=jobs, profile_sha256=profile_sha, training_audit_sha256=digest(TRAINING_AUDIT),
                draft_checkpoint_sha256=draft_sha, target_checkpoint_sha256=target_sha,
                prompts_audit_sha256=digest(PROMPTS), auditor_sha256=digest(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--verification', choices=('legacy', 'serial'), default='legacy')
    args = parser.parse_args()
    assert args.out.parent.resolve() == RUNS.resolve() and not args.out.exists()
    result = coverage(args.verification)
    write_json(args.out, result)
    if args.require_complete and not result['complete']:
        raise SystemExit('Some Mac Table 4 jobs lack complete trace audits.')


if __name__ == '__main__':
    main()
