"""Audit durable full-job coverage for the five Mac paper control families."""
import argparse
from collections import Counter
from pathlib import Path

from control_eval_binding import EXPERIMENTS
from qwen35_quality_data import digest, load, write_json
from train_ngram_capacity import FAMILIES, PROFILE as CAPACITY_TRAINING

SCOPE = Path('profiles/paper_control_benchmarks.json')
CODE = {'code','multipl'}


def arms_for(family):
    if family in EXPERIMENTS:
        return list(load(EXPERIMENTS[family])['arms'])
    if family in FAMILIES:
        capacity = load(load(CAPACITY_TRAINING)['capacity_profile'])
        return [f"s{entry['paper_scale_label']}" for entry in capacity[FAMILIES[family]]]
    assert family == 'table7'
    return load('profiles/ngram_table7_mac.json')['arms']


def checked_arm(raw, audit, meta, job):
    row = load(audit)
    assert row['complete'] and row['items'] == row['target'] == job['target_items']
    assert row['output_sha256'] == digest(raw)
    assert meta['status'] == 'complete' and row['identity'] == meta['identity']
    assert row['items'] == meta['items'] and row['correct'] == meta['correct']
    expected_scored = 0 if job['name'] in CODE else job['target_items']
    assert row['scored'] == expected_scored
    return row


def checked_pair(path, job, prefix, suffix, family, arms):
    report = load(path)
    assert report['status']=='complete_paired_public_job' and report['job']==suffix
    assert report['items_per_arm']==job['target_items'] and list(report['arms'])==arms
    identity = report['common_identity']
    expected_family = None if family=='table7' else family
    assert identity['job']==job['name'] and identity.get('family')==expected_family
    assert identity['job_spec']==job
    results = {}
    for arm,record in report['arms'].items():
        raw = Path(f'{prefix}-{arm}-{suffix}.jsonl')
        audit = Path(record['audit'])
        assert audit==raw.with_name(raw.stem+'-audit.json')
        assert digest(audit) == record['audit_sha256']
        row = checked_arm(raw, audit, load(str(raw)+'.meta.json'), job)
        assert row['target'] == record['target'] and row['output_sha256'] == record['output_sha256']
        assert row['identity']['arm']==arm and row['identity']['checkpoint_sha256']==record['checkpoint_sha256']
        assert {key:value for key,value in row['identity'].items() if key not in ('arm','checkpoint_sha256')}==identity
        assert row['scored']==record['scored'] and row['correct']==record['correct']
        results[arm] = dict(items=row['items'], scored=row['scored'], correct=row['correct'],
                            checkpoint_sha256=record['checkpoint_sha256'])
    return dict(status='generated_pending_judge' if job['name'] in CODE else 'scored_full_pair',
                paired=str(path),paired_sha256=digest(path),arms=results)


def observed_arms(job,prefix,suffix,family,arms):
    result = {}
    expected_family = None if family=='table7' else family
    for arm in arms:
        raw = Path(f'{prefix}-{arm}-{suffix}.jsonl')
        meta = Path(str(raw)+'.meta.json')
        if not meta.exists():
            result[arm] = dict(status='awaiting_metadata' if raw.exists() else 'not_started',items=0)
            continue
        row = load(meta)
        assert row['identity']['arm']==arm and row['identity'].get('family')==expected_family
        assert row['identity']['job_spec']==job and row['target']==job['target_items']
        assert 0<=row['items']<=row['target']
        snapshot = dict(status=row['status'], items=row['items'], metadata_snapshot=str(meta))
        audit = raw.with_name(raw.stem+'-audit.json')
        if row['status'] == 'complete' and audit.exists():
            verified = checked_arm(raw, audit, row, job)
            snapshot.update(audit=str(audit), audit_sha256=digest(audit),
                            output_sha256=verified['output_sha256'],
                            scored=verified['scored'], correct=verified['correct'])
        result[arm] = snapshot
    return result


def coverage():
    scope = load(SCOPE)
    assert digest(scope['paper'])==scope['paper_sha256']
    protocols = [load(scope[name]) for name in ('text_protocol','chinese_protocol')]
    jobs = {job['name']:job for protocol in protocols for job in protocol['jobs']}
    languages = [name for name in jobs if name.startswith('mmmlu-')]
    assert len(languages)==14 and all(jobs[name]['target_items']==14042 for name in languages)
    families = {}
    for family,spec in scope['families'].items():
        arms = arms_for(family)
        selected = [*scope['common_jobs'],*languages,*spec['extra_jobs']]
        entries = {}
        for name in selected:
            job = jobs[name]
            suffix = spec.get('aliases',{}).get(name,name)
            prefix = spec['prefix']
            pair = Path(f'{prefix}-{name}-paired.json')
            if pair.exists():
                entries[name] = checked_pair(pair,job,prefix,suffix,family,arms)
            else:
                snapshots = observed_arms(job,prefix,suffix,family,arms)
                entries[name] = dict(status='in_progress' if any(r['status']!='not_started' for r in snapshots.values()) else 'not_started',
                                     arms=snapshots)
        families[family] = dict(arms=arms, jobs=entries, job_counts=dict(Counter(j['status'] for j in entries.values())),
                                target_predictions=len(arms)*sum(jobs[name]['target_items'] for name in selected),
                                unavailable_columns=spec.get('unavailable_columns',[]))
    return dict(scope_sha256=digest(SCOPE), auditor_sha256=digest(__file__), families=families)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    assert args.out.parent.resolve()==Path('runs').resolve() and not args.out.exists()
    write_json(args.out,coverage())
