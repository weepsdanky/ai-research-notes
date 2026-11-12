"""Pair audited Table 4 attention arms by the locked prompt identities."""
import argparse
from fractions import Fraction
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path

from audit_mtp_coverage import MT_PROFILE, PROMPTS, RUNS
from evaluate_mtp_attention import training_binding
from mtp_public_data import PROFILE
from qwen35_quality_data import digest, load, write_json
from run_mtp_public_jobs import FULL_JOBS

PREFIXES = {'full': 'mtp-dense-serial', 'qsa': 'mtp-public-serial'}


def checked_output(job, attention, expected):
    training, checkpoint, state = training_binding(attention)
    raw = RUNS / f'{PREFIXES[attention]}-{job}.jsonl'
    audit_path = raw.with_name(raw.stem + '-audit.json')
    audit = load(audit_path)
    meta = load(f'{raw}.meta.json')
    identity = audit['identity']
    assert audit['complete'] and audit['totals']['items'] == audit['target'] == expected['items']
    assert meta['status'] == 'complete' and meta['identity'] == identity and meta['totals'] == audit['totals']
    assert identity['job'] == job and identity['verification'] == 'serial'
    assert identity['profile_sha256'] == digest(PROFILE)
    assert identity['prompt_content_sha256'] == expected['prompt_content_sha256']
    assert identity['draft_checkpoint_sha256'] == digest(checkpoint)
    assert identity['target_checkpoint_sha256'] == digest(training['target'])
    if attention == 'full' or job == 'mt-bench':
        assert identity['attention'] == attention and identity['training_state_sha256'] == digest(state)
    assert all(digest(path) == sha for path, sha in identity['sources'].items())
    assert audit['output_sha256'] == digest(raw)
    trained = audit['totals']['modes']['trained']
    acceptance = 1 + trained['accepted'] / trained['rounds'] if trained['rounds'] else None
    assert audit['acceptance']['trained'] == acceptance
    evidence = dict(audit=str(audit_path), audit_sha256=digest(audit_path), output_sha256=audit['output_sha256'],
                    draft_checkpoint_sha256=identity['draft_checkpoint_sha256'],
                    target_checkpoint_sha256=identity['target_checkpoint_sha256'],
                    acceptance_length=acceptance, trained=trained)
    return raw, evidence


def paired(job, full, qsa):
    counts = dict(items=0, same_inputs=0, different_second_turn_inputs=0,
                  same_target_outputs=0, qsa_higher=0, tied=0, full_higher=0, no_rounds=0)
    for left, right in zip(full, qsa, strict=True):
        assert left['source_id'] == right['source_id']
        same = left['input_ids'] == right['input_ids']
        if job == 'mt-bench':
            assert (left['turn'], left['category']) == (right['turn'], right['category'])
            assert left['turn'] == 2 or same
            counts['different_second_turn_inputs'] += not same
        else:
            assert left['group'] == right['group'] == job and left['task'] == right['task']
            assert same
        counts['items'] += 1
        counts['same_inputs'] += same
        counts['same_target_outputs'] += left['output_ids'] == right['output_ids']
        a, b = left['speculative']['trained'], right['speculative']['trained']
        assert a['output_ids'] == left['output_ids'] and b['output_ids'] == right['output_ids']
        if not a['num_drafts'] or not b['num_drafts']:
            counts['no_rounds'] += 1
        else:
            delta = Fraction(b['num_accepted_tokens'], b['num_drafts']) - Fraction(a['num_accepted_tokens'], a['num_drafts'])
            counts['qsa_higher' if delta > 0 else 'full_higher' if delta < 0 else 'tied'] += 1
    return counts


def compare(job, expected):
    full, full_evidence = checked_output(job, 'full', expected)
    qsa, qsa_evidence = checked_output(job, 'qsa', expected)
    assert full_evidence['target_checkpoint_sha256'] != qsa_evidence['target_checkpoint_sha256']
    assert full_evidence['draft_checkpoint_sha256'] != qsa_evidence['draft_checkpoint_sha256']
    with full.open() as a, qsa.open() as b:
        counts = paired(job, map(json.loads, a), map(json.loads, b))
    assert counts['items'] == expected['items']
    return dict(full=full_evidence, qsa=qsa_evidence, **counts)


def compare_prefix(job, expected, audit_path):
    audit = load(audit_path)
    identity = audit['identity']
    assert audit['status'] == 'verified_durable_snapshot'
    count = audit['totals']['items']
    assert 0 < count == audit['items'] < audit['target'] == expected['items']
    assert identity['job'] == job and identity['attention'] == 'full' and identity['verification'] == 'serial'
    assert identity['profile_sha256'] == digest(PROFILE)
    assert identity['prompt_content_sha256'] == expected['prompt_content_sha256']
    training, checkpoint, state = training_binding('full')
    assert identity['target_checkpoint_sha256'] == digest(training['target'])
    assert identity['draft_checkpoint_sha256'] == digest(checkpoint)
    assert identity['training_state_sha256'] == digest(state)
    assert all(digest(path) == sha for path, sha in identity['sources'].items())

    qsa, qsa_evidence = checked_output(job, 'qsa', expected)
    for key in ('target_checkpoint_sha256', 'draft_checkpoint_sha256'):
        assert identity[key] != qsa_evidence[key]
    full = RUNS / f'{PREFIXES["full"]}-{job}.jsonl'
    fingerprint = sha256()

    def full_rows(source):
        for line in islice(source, count):
            assert line.endswith(b'\n')
            fingerprint.update(line)
            yield json.loads(line)

    with full.open('rb') as a, qsa.open('rb') as b:
        counts = paired(job, full_rows(a), map(json.loads, islice(b, count)))
    assert counts['items'] == count and fingerprint.hexdigest() == audit['output_prefix_sha256']
    return dict(target=expected['items'], full_prefix_audit=str(audit_path), full_prefix_audit_sha256=digest(audit_path),
                qsa_full_audit=qsa_evidence['audit'], qsa_full_audit_sha256=qsa_evidence['audit_sha256'], **counts)


def five_job_macro(results):
    if set(results) != set(FULL_JOBS):
        return None
    macro = {}
    for arm in PREFIXES:
        scores = [job[arm]['acceptance_length'] for job in results.values()]
        macro[arm] = None if None in scores else sum(scores) / len(scores)
    return macro


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', nargs='+', choices=FULL_JOBS, default=FULL_JOBS)
    parser.add_argument('--prefix-audit', type=Path, help='Pair one independently audited Full Attention prefix')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert args.out.parent.resolve() == RUNS.resolve() and not args.out.exists()
    assert len(set(args.jobs)) == len(args.jobs)
    source = load(PROMPTS)
    assert source['profile_sha256'] == digest(PROFILE)
    expected = {job: dict(items=row['items'], prompt_content_sha256=row['content_sha256'])
                for job, row in source['jobs'].items()}
    mt = load(MT_PROFILE)
    expected['mt-bench'] = dict(items=mt['conversation_count'] * mt['turns_per_conversation'],
                               prompt_content_sha256=digest(MT_PROFILE))
    if args.prefix_audit:
        if len(args.jobs) != 1:
            parser.error('--prefix-audit requires exactly one job')
        job, = args.jobs
        result = compare_prefix(job, expected[job], args.prefix_audit)
        write_json(args.out, dict(status='paired_audited_prefix', jobs={job: result}, five_job_macro=None,
                                  profile_sha256=digest(PROFILE), auditor_sha256=digest(__file__)))
        return
    results = {job: compare(job, expected[job]) for job in args.jobs}
    write_json(args.out, dict(status='complete_paired_attention_jobs', jobs=results, five_job_macro=five_job_macro(results),
                              profile_sha256=digest(PROFILE), auditor_sha256=digest(__file__)))


if __name__ == '__main__':
    main()
