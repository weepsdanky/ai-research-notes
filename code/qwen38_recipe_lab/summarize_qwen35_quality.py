"""Assemble evidence for the nine declared public Table 1 quality groups."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

from qwen35_quality_data import PROTOCOL, digest, load, training_binding, write_json

RUNS = Path('runs')
CODE = {'code', 'multipl'}
JUDGE_PROTOCOL = Path('profiles/qwen35_code_judging_protocol.json')


def generation_evidence(path, protocol, jobs, bindings, checked):
    result = load(path)
    identity = result['identity']
    arm, name = identity['binding']['arm'], identity['job']
    assert arm in protocol['arms'] and name in jobs
    job = jobs[name]
    raw = Path(job['output_template'].format(arm=arm))
    meta_path = Path(f'{raw}.meta.json')
    meta = load(meta_path)
    assert result['status'] == 'passed_source_output_audit' and result['coverage'] == 'full'
    assert result['auditor_sha256'] == digest('audit_qwen35_quality.py')
    assert meta['status'] == 'complete' and result['full_dataset_generated'] and meta['full_dataset_generated']
    assert Path(result['predictions']).resolve() == raw.resolve()
    assert result['predictions_sha256'] == digest(raw) and result['meta_sha256'] == digest(meta_path)
    assert identity == meta['identity'] and identity['protocol_sha256'] == digest(PROTOCOL)
    assert identity['job_spec'] == job and identity['generation'] == protocol['generation']
    assert identity['tokenizer_sha256'] == protocol['tokenizer']['sha256']
    binding = identity['binding']
    paths = (binding['training_profile'], binding['training_audit'], binding['checkpoint'], arm)
    if paths not in checked:
        checked[paths] = training_binding(*paths)
    assert binding == checked[paths]
    initial = bindings.setdefault(arm, binding)
    assert all(initial[key] == binding[key] for key in ('training_profile_sha256', 'checkpoint_sha256'))
    counts = result['tasks']
    assert set(counts) == set(job['tasks']) == set(meta['tasks'])
    assert result['items'] == result['target'] == meta['completed_items'] == meta['target_items'] == job['target_items']
    assert sum(row['n'] for row in counts.values()) == result['items']
    assert sum(row['correct'] for row in counts.values()) == result['correct']
    for task, row in counts.items():
        assert row == {key: meta['tasks'][task][key] for key in row}
        assert row['n'] > 0 and 0 <= row['correct'] <= row['scored']
        assert row['scored'] == (0 if name in CODE else row['n'])
    return (arm, name), dict(audit=str(path), audit_sha256=digest(path), identity=identity,
                             predictions=str(raw), predictions_sha256=result['predictions_sha256'],
                             items=result['items'], tasks=counts)


def judge_counts(raw, name):
    group, item, keys = ('dataset', 'task_id', ('base_pass', 'plus_pass')) if name == 'code' else (
        'language', 'name', ('passed',))
    counts, seen = {}, set()
    with raw.open() as stream:
        for line in stream:
            row = json.loads(line)
            identity = row[group], row[item]
            assert identity not in seen and line.endswith('\n')
            seen.add(identity)
            score = counts.setdefault(row[group], dict.fromkeys(('items', *keys), 0))
            score['items'] += 1
            for key in keys:
                assert type(row[key]) is bool
                score[key] += row[key]
    return counts


def judged_evidence(raw, path, generated):
    result = load(path)
    identity = result['identity']
    provenance = identity['generation_provenance']
    binding = provenance['generation_identity']
    arm, name = binding['binding']['arm'], binding['job']
    assert name in CODE
    generation = generated[arm, name]
    assert generation['identity'] == binding
    for key in ('predictions', 'generation_audit'):
        expected = generation['predictions' if key == 'predictions' else 'audit']
        assert Path(provenance[key]).resolve() == Path(expected).resolve()
    assert provenance['predictions_sha256'] == generation['predictions_sha256']
    assert provenance['generation_audit_sha256'] == generation['audit_sha256']
    assert provenance['judging_protocol_sha256'] == digest(JUDGE_PROTOCOL)
    protocol = load(JUDGE_PROTOCOL)
    sources = {**protocol['common_sources'], **protocol['suites'][name]['source_hashes']}
    assert all(digest(source) == sha for source, sha in sources.items())
    meta_path = Path(f'{raw}.meta.json')
    meta = load(meta_path)
    assert result['status'] == 'passed_execution_evidence_audit' and result['full_judged']
    assert result['raw_only'] == (name == 'multipl')
    assert result['judged_sha256'] == digest(raw) and result['meta_sha256'] == digest(meta_path)
    assert meta['identity'] == identity and meta['status'] == 'complete'
    assert result['items'] == meta['completed_items'] == meta['target_items'] == provenance['target_items'] == generation['items']
    assert identity['judge_sha256'] == digest('judge_qwen35_code.py')
    image = 'evalplus' if name == 'code' else 'multipl'
    assert identity['image_id'] == load('code_images.lock.json')[image]['image_id']
    if name == 'multipl':
        assert identity['helper_sha256'] == digest('audit_multipl_results.py')
    counts = result['tasks']
    assert counts == judge_counts(raw, name)
    prefix = 'evalplus-' if name == 'code' else 'multipl-'
    expected = {task.removeprefix(prefix): row['n'] for task, row in generation['tasks'].items()}
    assert {task: row['items'] for task, row in counts.items()} == expected
    keys = ('base_pass', 'plus_pass') if name == 'code' else ('passed',)
    for row in counts.values():
        assert all(0 <= row[key] <= row['items'] for key in keys)
    accuracy = None
    if name == 'code':
        assert len(counts) == 2
        accuracy = mean(row[key] / row['items'] for row in counts.values() for key in keys)
        assert abs(accuracy - result['four_suite_mean']) < 1e-15
    return (arm, name), dict(audit=str(path), audit_sha256=digest(path), judged=str(raw),
                             judged_sha256=result['judged_sha256'], items=result['items'], tasks=counts,
                             accuracy=accuracy, raw_only=result['raw_only'])


def text_scores(group, counts):
    micro = sum(row['correct'] for row in counts.values()) / sum(row['n'] for row in counts.values())
    scores = dict(micro_accuracy=micro)
    if group == 'mmmlu':
        assert len(counts) == 14 and all(row['n'] == 14042 for row in counts.values())
        scores['language_macro_accuracy'] = mean(row['correct'] / row['n'] for row in counts.values())
    if group == 'bbh':
        families = defaultdict(list)
        for task, row in counts.items():
            name = task.removeprefix('bbh-')
            family = next((key for key in ('logical_deduction', 'tracking_shuffled_objects')
                           if name.startswith(key + '_')), name)
            families[family].append(row['correct'] / row['n'])
        assert len(counts) == 27 and len(families) == 23
        scores['configuration_macro_accuracy'] = mean(value for values in families.values() for value in values)
        scores['family_macro_accuracy'] = mean(mean(values) for values in families.values())
    return scores


def summarize(generation_audits, judge_audits):
    protocol = load(PROTOCOL)
    jobs = {job['name']: job for job in protocol['jobs']}
    bindings, generated, judged, checked = {}, {}, {}, {}
    for path in generation_audits:
        key, evidence = generation_evidence(path, protocol, jobs, bindings, checked)
        assert key not in generated
        generated[key] = evidence
    for raw, path in judge_audits:
        key, evidence = judged_evidence(raw, path, generated)
        assert key not in judged
        judged[key] = evidence
    arms = {}
    for arm in protocol['arms']:
        groups = {}
        for group, target in protocol['targets'].items():
            selected = [job for job in jobs.values() if job['group'] == group]
            assert sum(job['target_items'] for job in selected) == target
            present = {job['name']: generated[arm, job['name']] for job in selected if (arm, job['name']) in generated}
            items = sum(result['items'] for result in present.values())
            entry = dict(target=target, audited_generation=items, generation=present, scores=None)
            if items != target:
                entry['status'] = 'incomplete_generation'
            elif group in ('evalplus', 'multipl-e'):
                name = selected[0]['name']
                evidence = judged.get((arm, name))
                entry['judging'] = evidence
                entry['status'] = ('awaiting_judging' if evidence is None else
                                   'raw_fixture_limit' if evidence['raw_only'] else 'complete_scored')
                if evidence and not evidence['raw_only']:
                    entry['scores'] = dict(four_suite_mean=evidence['accuracy'])
            else:
                counts = {name: row for evidence in present.values() for name, row in evidence['tasks'].items()}
                assert sum(row['n'] for row in counts.values()) == target
                entry.update(status='complete_scored', scores=text_scores(group, counts))
            groups[group] = entry
        arms[arm] = dict(binding=bindings.get(arm), groups=groups)
    return dict(protocol_sha256=digest(PROTOCOL), summarizer_sha256=digest(__file__), arms=arms)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation-audit', nargs='+', type=Path, required=True)
    parser.add_argument('--judged', nargs=2, metavar=('PREDICTIONS', 'AUDIT'), action='append', type=Path, default=[])
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    assert args.out.parent.resolve() == RUNS.resolve() and not args.out.exists()
    write_json(args.out, summarize(args.generation_audit, args.judged))
