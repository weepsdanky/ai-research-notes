"""Pair fully audited Table 2 EvalPlus outcomes by their locked task IDs."""
import argparse
from collections import Counter
import json
from pathlib import Path

from qwen35_quality_data import digest, load, write_json

PROFILE = Path('profiles/paper_dense_bounded_evaluation.json')
SPARSE_AUDIT = Path('runs/paper-evalplus-completed-audit.json')
BINDING = Path('runs/paper-dense-evaluation-binding.json')
DATASETS = {'humaneval': 164, 'mbpp': 378}
SUITES = ('base', 'plus')
MATCHED_PROTOCOL = ('content_sha256', 'tokenizer_sha256', 'stop_strings', 'suite', 'tasks',
                    'tokens', 'context', 'batch_size', 'prefill_chunk', 'gdn_scan',
                    'grouped_gqa', 'prefix_cache', 'device', 'scope')


def execution(audit_path, prediction_path, checkpoint_sha):
    audited = load(audit_path)
    judged_path = audited['judged_path']
    assert audited['all_prompts_ids_solutions_driver_bytes_official_results_images_and_cleanups_verified']
    assert Path(audited['predictions_path']) == Path(prediction_path)
    assert audited['predictions_sha256'] == digest(prediction_path)
    assert audited['judged_sha256'] == digest(judged_path)
    generation = load(f'{prediction_path}.meta.json')
    judged = load(f'{judged_path}.meta.json')
    assert generation['status'] == judged['status'] == 'complete' and generation['full_dataset_generated']
    identity = generation['identity']
    judge = judged['identity']
    assert judge['generation_identity'] == identity and judge['predictions_sha256'] == audited['predictions_sha256']
    assert identity['checkpoint_sha256'] == audited['generation_checkpoint_sha256'] == checkpoint_sha
    assert judge['source_hashes'] == audited['source_hashes']
    assert audited['scores'] == judged['tasks'] and audited['four_suite_mean'] == judged['four_suite_mean']
    assert audited['judged_items'] == judged['completed_items'] == generation['completed_items'] == sum(DATASETS.values())
    scores = {}
    with Path(judged_path).open() as stream:
        for line in stream:
            row = json.loads(line)
            key = row['dataset'], row['task_id']
            assert key not in scores and key[0] in DATASETS and row['image_id'] == judge['image_id']
            scores[key] = tuple(row[f'{suite}_pass'] for suite in SUITES)
            assert all(type(flag) is bool for flag in scores[key])
            assert not scores[key][1] or scores[key][0]
    assert Counter(dataset for dataset, _ in scores) == DATASETS
    for dataset, target in DATASETS.items():
        expected = dict(n=target)
        for i, suite in enumerate(SUITES):
            expected[f'{suite}_correct'] = sum(flags[i] for key, flags in scores.items() if key[0] == dataset)
        assert expected == audited['scores'][dataset]
    evidence = dict(audit=str(audit_path), audit_sha256=digest(audit_path),
                    predictions_sha256=audited['predictions_sha256'], judged_sha256=audited['judged_sha256'],
                    checkpoint_sha256=checkpoint_sha, scores=audited['scores'],
                    four_suite_mean=audited['four_suite_mean'])
    return identity, judge, scores, evidence


def paired(sparse, dense):
    assert sparse.keys() == dense.keys()
    counts = {dataset: {suite: dict(both_correct=0, sparse_only_correct=0, dense_only_correct=0,
                                   neither_correct=0) for suite in SUITES} for dataset in DATASETS}
    names = {(True, True): 'both_correct', (True, False): 'sparse_only_correct',
             (False, True): 'dense_only_correct', (False, False): 'neither_correct'}
    for key, sparse_flags in sparse.items():
        for suite, left, right in zip(SUITES, sparse_flags, dense[key], strict=True):
            counts[key[0]][suite][names[left, right]] += 1
    return counts


def compare(dense_audit):
    profile = load(PROFILE)
    job, = [item for item in profile['jobs'] if item['name'] == 'code']
    binding = load(BINDING)
    assert digest('profiles/paper_dense_evaluation.json') == binding['profile_sha256'] == profile['parent_profile_sha256']
    assert job['target_items'] == sum(DATASETS.values())
    assert digest(job['primary_meta']) == job['primary_meta_sha256']
    sparse_path = job['primary_meta'].removesuffix('.meta.json')
    control = load('profiles/paper_dense_control.json')
    si, sj, sparse, sparse_evidence = execution(SPARSE_AUDIT, sparse_path, control['comparison_sha256'])
    di, dj, dense, dense_evidence = execution(dense_audit, job['output'], binding['checkpoint_sha256'])
    assert si['checkpoint_sha256'] != di['checkpoint_sha256']
    assert all(si[key] == di[key] for key in MATCHED_PROTOCOL)
    assert si['content_sha256'] == job['primary_content_sha256']
    assert digest('evaluate_dense_bounded.py') == profile['engine_sha256']
    assert digest('bounded_dense_inference.py') == di['dense_execution_variant']['source_sha256'] == profile['kernel_sha256']
    assert sj['image_id'] == dj['image_id'] and sj['source_hashes'] == dj['source_hashes']
    return dict(status='complete_paired_evalplus', items=job['target_items'], image_id=sj['image_id'],
                matched_protocol={key: si[key] for key in MATCHED_PROTOCOL},
                sparse=sparse_evidence, dense=dense_evidence, paired_correctness=paired(sparse, dense),
                profile_sha256=digest(PROFILE), auditor_sha256=digest(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dense-audit', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    assert args.out.parent.resolve() == Path('runs').resolve() and not args.out.exists()
    write_json(args.out, compare(args.dense_audit))


if __name__ == '__main__':
    main()
