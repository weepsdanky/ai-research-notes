"""Break down audited serial MTP target forwards on paired evaluation inputs."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median

from compare_mtp_attention import FULL_JOBS, MT_PROFILE, PREFIXES, PROMPTS, checked_output, paired
from mtp_public_data import PROFILE
from qwen35_quality_data import digest, load, write_json

MODES = ('initial', 'trained')


def forward_parts(trace):
    parts = Counter(prefill=1, boundary=0, verify_seed=0, verify_proposals=0, replay=0)
    replayed = 0
    for step in trace['rounds']:
        if step['kind'] == 'target_only_boundary':
            parts['boundary'] += 1
            continue
        parts['verify_seed'] += 1
        parts['verify_proposals'] += step['drafted']
        if step['target_replayed']:
            parts['replay'] += step['target_consumed']
            replayed += 1
    assert sum(parts.values()) == trace['target_forward_calls']
    assert parts['prefill'] + parts['boundary'] + parts['verify_seed'] + replayed == trace['target_calls']
    assert parts['verify_seed'] == trace['num_drafts'] and parts['verify_proposals'] == trace['num_draft_tokens']
    return parts


def cost_rows(raw, audit, result):
    seconds = {name: [] for name in ('teacher', *MODES)}
    teacher_forwards = 0
    parts = {mode: Counter() for mode in MODES}
    with raw.open() as stream:
        for line in stream:
            row = json.loads(line)
            teacher_forwards += len(row['output_ids'])
            seconds['teacher'].append(row['teacher_seconds'])
            for mode in MODES:
                trace = row['speculative'][mode]
                parts[mode].update(forward_parts(trace))
                seconds[mode].append(trace['seconds'])
            yield row
    totals = audit['totals']
    assert len(seconds['teacher']) == totals['items'] and sum(seconds['teacher']) == totals['teacher_seconds']
    for mode in MODES:
        assert sum(seconds[mode]) == totals['modes'][mode]['seconds']
        assert sum(parts[mode].values()) == totals['modes'][mode]['target_forward_calls']
    result.update(teacher_forwards=teacher_forwards, speculative_forwards=parts,
                  total_seconds={mode: sum(values) for mode, values in seconds.items()},
                  median_seconds={mode: median(values) for mode, values in seconds.items()})


def audit_cost(job):
    catalog = load(PROMPTS)
    assert catalog['profile_sha256'] == digest(PROFILE)
    if job == 'mt-bench':
        profile = load(MT_PROFILE)
        count = profile['conversation_count'] * profile['turns_per_conversation']
        expected = dict(items=count, prompt_content_sha256=digest(MT_PROFILE))
    else:
        spec = catalog['jobs'][job]
        expected = dict(items=spec['items'], prompt_content_sha256=spec['content_sha256'])
    arms, rows = {}, {}
    for arm in PREFIXES:
        raw, evidence = checked_output(job, arm, expected)
        audit = load(evidence['audit'])
        arms[arm] = {key: evidence[key] for key in ('audit', 'audit_sha256', 'output_sha256',
                    'draft_checkpoint_sha256', 'target_checkpoint_sha256', 'acceptance_length')}
        rows[arm] = cost_rows(raw, audit, arms[arm])
    for key in ('draft_checkpoint_sha256', 'target_checkpoint_sha256'):
        assert arms['full'][key] != arms['qsa'][key]
    matched = paired(job, rows['full'], rows['qsa'])
    assert matched['items'] == expected['items']
    return dict(job=job, verification='serial', matched=matched, arms=arms,
                profile_sha256=digest(PROFILE), auditor_sha256=digest(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', choices=FULL_JOBS, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert args.out.parent.resolve() == Path('runs').resolve() and not args.out.exists()
    write_json(args.out, audit_cost(args.job))


if __name__ == '__main__':
    main()
