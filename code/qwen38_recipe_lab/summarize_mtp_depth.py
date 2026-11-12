"""Summarize draft depth and paired changes from complete MTP trace audits."""
import argparse
from fractions import Fraction
import json
from pathlib import Path

from audit_mtp_coverage import MT_AUDIT, RUNS, coverage
from mtp_public_data import PROFILE
from qwen35_quality_data import digest, load, write_json

MODES = ('initial', 'trained')


def rows_for(job, verification):
    if verification == 'batched_archive':
        assert job == 'mt-bench'
        yield from load(load(MT_AUDIT)['predictions'])['items']
        return
    prefix = 'mtp-public-serial' if verification == 'serial' else 'mtp-public'
    with (RUNS / f'{prefix}-{job}.jsonl').open() as stream:
        yield from map(json.loads, stream)


def summarize(rows, checked, depth):
    modes = {mode: dict(proposed=[0] * depth, accepted=[0] * depth, rounds=0) for mode in MODES}
    paired = dict(improved=0, tied=0, declined=0, no_rounds=0)
    items = 0
    for row in rows:
        lengths = {}
        for mode in MODES:
            trace = row['speculative'][mode]
            assert trace['output_ids'] == row['output_ids']
            summary = modes[mode]
            for step in trace['rounds']:
                if 'drafted' not in step:
                    assert step['kind'] == 'target_only_boundary'
                    continue
                assert 0 <= step['accepted'] <= step['drafted'] <= depth
                summary['rounds'] += 1
                for position in range(step['drafted']):
                    summary['proposed'][position] += 1
                for position in range(step['accepted']):
                    summary['accepted'][position] += 1
            if trace['num_drafts']:
                lengths[mode] = Fraction(trace['num_accepted_tokens'], trace['num_drafts'])
        if len(lengths) < 2:
            paired['no_rounds'] += 1
        else:
            delta = lengths['trained'] - lengths['initial']
            paired['improved' if delta > 0 else 'declined' if delta < 0 else 'tied'] += 1
        items += 1
    assert items == checked['items'] == checked['target'] == sum(paired.values())
    for mode, summary in modes.items():
        expected = checked['totals']['modes'][mode]
        assert summary['rounds'] == expected['rounds']
        assert sum(summary['proposed']) == expected['draft_tokens']
        assert sum(summary['accepted']) == expected['accepted']
        mean = 1 + expected['accepted'] / expected['rounds'] if expected['rounds'] else None
        assert mean == checked['acceptance'][mode]
        summary['mean_acceptance_length'] = mean
    return dict(items=items, modes=modes, paired_prompt_acceptance=paired,
                output_sha256=checked['output_sha256'], audit_sha256=checked['audit_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    profile = load(PROFILE)
    parser.add_argument('--jobs', nargs='+', required=True, choices=('mt-bench', *profile['jobs']))
    parser.add_argument('--verification', choices=('legacy', 'serial'), default='serial')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    assert args.out.parent.resolve() == Path('runs').resolve() and not args.out.exists()
    assert len(set(args.jobs)) == len(args.jobs)
    checked = coverage(args.verification)['jobs']
    jobs = {}
    for job in args.jobs:
        assert checked[job]['status'] == 'complete_audited', f'{job} requires its complete trace audit'
        verification = checked[job]['verification']
        jobs[job] = summarize(rows_for(job, verification), checked[job], profile['draft_tokens'])
        jobs[job]['verification'] = verification
    write_json(args.out, dict(jobs=jobs, profile_sha256=digest(PROFILE),
                              summarizer_sha256=digest(__file__)))


if __name__ == '__main__':
    main()
