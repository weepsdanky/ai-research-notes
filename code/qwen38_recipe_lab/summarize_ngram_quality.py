"""Bind full paired Mac benchmark results across identical source rows."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path

from qwen35_quality_data import digest, load, write_json


def summarize(prefix, job, family=None):
    if family in ('table5', 'table10'):
        from control_eval_binding import EXPERIMENTS

        arms = list(load(EXPERIMENTS[family])['arms'])
    elif family:
        from train_ngram_capacity import FAMILIES, PROFILE

        capacity = load(load(PROFILE)['capacity_profile'])
        arms = [f"s{entry['paper_scale_label']}" for entry in capacity[FAMILIES[family]]]
    else:
        arms = load('profiles/ngram_table7_mac.json')['arms']
    baseline = arms[0]
    outcome_key = 'outcome_changes_from_baseline' if family else 'outcome_changes_from_none'
    tokens_key = 'output_token_changes_from_baseline' if family else 'output_token_changes_from_none'
    reports = {}
    common = None
    with ExitStack() as stack:
        streams = {}
        for arm in arms:
            data = Path(f'{prefix}-{arm}-{job}.jsonl')
            audit = Path(f'{prefix}-{arm}-{job}-audit.json')
            report = load(audit)
            assert report['complete'] and report['items'] == report['target']
            assert report['output_sha256'] == digest(data)
            assert report['identity']['arm'] == arm and report['identity'].get('family') == family
            comparable = {key:value for key,value in report['identity'].items()
                          if key not in ('arm','checkpoint_sha256')}
            common = common or comparable
            assert common == comparable
            meta = load(str(data)+'.meta.json')
            assert meta['identity'] == report['identity'] and meta['status'] == 'complete'
            reports[arm] = dict(audit=str(audit),audit_sha256=digest(audit),
                                checkpoint_sha256=report['identity']['checkpoint_sha256'],
                                output_sha256=report['output_sha256'],
                                target=report['target'],correct=report['correct'],scored=report['scored'],
                                process_peak_rss_bytes=meta['process_peak_rss_bytes'],
                                **{outcome_key: 0, tokens_key: 0})
            if report.get('subject_macro') is not None:
                reports[arm]['subject_macro'] = report['subject_macro']
            streams[arm] = stack.enter_context(data.open())
        count = 0
        for lines in zip(*streams.values(),strict=True):
            batch = {arm:json.loads(line) for arm,line in zip(arms,lines,strict=True)}
            base = batch[baseline]
            for arm,row in batch.items():
                assert all(row[key] == base[key] for key in ('task','index','input_ids','gold'))
                reports[arm][outcome_key] += row['correct'] != base['correct']
                reports[arm][tokens_key] += row['output_ids'] != base['output_ids']
            count += 1
        assert all(report['target']==count for report in reports.values())
    summary = dict(status='complete_paired_public_job',job=job,items_per_arm=count,
                   common_identity=common,arms=reports)
    if family:
        summary.update(family=family, baseline=baseline)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--job', required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--capacity-family', choices=('table8','table9'))
    group.add_argument('--experiment', choices=('table5','table10'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert args.out.parent.resolve()==Path('runs').resolve() and not args.out.exists()
    write_json(args.out,summarize(args.prefix,args.job,args.experiment or args.capacity_family))
