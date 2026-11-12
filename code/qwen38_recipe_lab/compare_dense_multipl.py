"""Pair the audited original and Mac-repaired MultiPL-E outcomes."""
import argparse
from collections import Counter
import json
from pathlib import Path

from audit_generation_sources import verify_generation_sources
from compare_dense_evalplus import BINDING, MATCHED_PROTOCOL, PROFILE
from qwen35_quality_data import digest, load, write_json

REPAIRED_PROFILE = Path('profiles/paper_dense_multipl_repaired_evaluation.json')
OUTCOMES = {(True, True): 'both_correct', (True, False): 'sparse_only_correct',
            (False, True): 'dense_only_correct', (False, False): 'neither_correct'}


def scores(rows, image):
    result = {}
    for row in rows:
        key = row['language'], row['name']
        assert key not in result and type(row['passed']) is bool and row['image_id'] == image
        result[key] = row['passed']
    return result


def execution(audit_path, checkpoint_sha, predictions, repaired_predictions):
    audit = load(audit_path)
    assert audit['all_program_inputs_generated_ids_official_returns_images_limits_driver_bytes_and_cleanups_verified']
    assert audit['generation_checkpoint_sha256'] == checkpoint_sha
    artifacts = audit['artifacts']
    paths = {name: Path(artifact['path']) for name, artifact in artifacts.items()}
    assert all(digest(path) == artifacts[name]['sha256'] for name, path in paths.items())
    assert paths['predictions'] == Path(predictions) and paths['repaired_predictions'] == Path(repaired_predictions)
    for name in ('predictions', 'raw', 'scaffolds', 'repaired_predictions'):
        assert paths[f'{name}_meta'] == Path(f'{paths[name]}.meta.json')
    generation = load(paths['predictions_meta'])
    repair_generation = load(paths['repaired_predictions_meta'])
    raw_meta = load(paths['raw_meta'])
    assert generation['status'] == repair_generation['status'] == raw_meta['status'] == 'complete'
    assert generation['full_dataset_generated'] and repair_generation['full_dataset_generated']
    identity, repair_identity = generation['identity'], repair_generation['identity']
    assert identity['checkpoint_sha256'] == repair_identity['checkpoint_sha256'] == checkpoint_sha
    archive = audit['source_archive']
    assert len(archive) == 40 and all(c in '0123456789abcdef' for c in archive)
    for generation_identity in (identity, repair_identity):
        assert verify_generation_sources(generation_identity, archive)
    judge = raw_meta['identity']
    assert judge['generation_identity'] == identity and judge['raw_only']
    assert judge['predictions_sha256'] == artifacts['predictions']['sha256']
    scaffold_meta = load(paths['scaffolds_meta'])
    assert scaffold_meta['identity']['image_id'] == judge['image_id']
    with paths['raw'].open() as stream:
        raw = scores((json.loads(line) for line in stream), judge['image_id'])
    repaired = scores(load(paths['repaired_scores']), judge['image_id'])
    invalid = set(map(tuple, audit['invalid_original_fixtures']))
    assert repaired.keys() == invalid <= raw.keys() and len(invalid) == audit['repaired_execution_records_verified'] == 3
    summary = load(paths['summary'])
    assert summary['original_full_model_accuracy'] is audit['original_full_model_accuracy'] is None
    assert raw_meta['completed_items'] == summary['original_raw_tested'] == len(raw) == 1277
    original = {key: passed for key, passed in raw.items() if key not in invalid}
    assert len(original) == summary['original_valid_scaffolds'] == 1274
    assert sum(original.values()) == summary['original_valid_correct']
    composite = original | repaired
    counts = Counter(key[0] for key in composite)
    assert len(counts) == 8 and counts == {lang: row['n'] for lang, row in summary['tasks'].items()}
    for language, row in summary['tasks'].items():
        assert row['correct'] == sum(passed for key, passed in composite.items() if key[0] == language)
        assert row['repaired_fixtures'] == sum(key[0] == language for key in invalid)
    macro = sum(row['correct'] / row['n'] for row in summary['tasks'].values()) / len(counts)
    assert macro == summary['mac_repaired_eight_language_macro'] == audit['mac_repaired_macro']
    evidence = dict(audit=str(audit_path), audit_sha256=digest(audit_path), checkpoint_sha256=checkpoint_sha,
                    original_valid_correct=summary['original_valid_correct'], mac_repaired_macro=macro,
                    original_valid_items=len(original), mac_repaired_items=len(composite))
    return identity, repair_identity, judge, artifacts['scaffolds']['sha256'], original, repaired, evidence


def paired(sparse, dense):
    assert sparse.keys() == dense.keys()
    counts = {}
    for key, passed in sparse.items():
        language = key[0]
        counts.setdefault(language, dict.fromkeys(OUTCOMES.values(), 0))[OUTCOMES[passed, dense[key]]] += 1
    return counts


def compare(sparse_audit, dense_audit):
    profile = load(PROFILE)
    job, = [item for item in profile['jobs'] if item['name'] == 'multipl']
    repair_job = load(REPAIRED_PROFILE)
    binding = load(BINDING)
    assert digest('profiles/paper_dense_evaluation.json') == binding['profile_sha256'] == profile['parent_profile_sha256']
    assert digest('evaluate_dense_bounded.py') == profile['engine_sha256']
    assert digest('bounded_dense_inference.py') == profile['kernel_sha256']
    assert digest('multipl_repairs.lock.json') == repair_job['repair_lock_sha256']
    for spec in (job, repair_job):
        assert digest(spec['primary_meta']) == spec['primary_meta_sha256']
    sparse_sha = load('profiles/paper_dense_control.json')['comparison_sha256']
    dense_sha = binding['checkpoint_sha256']
    assert sparse_sha != dense_sha
    si, sri, sj, ss, sparse, sparse_repaired, se = execution(
        sparse_audit, sparse_sha, *(spec['primary_meta'].removesuffix('.meta.json') for spec in (job, repair_job)))
    di, dri, dj, ds, dense, dense_repaired, de = execution(
        dense_audit, dense_sha, job['output'], repair_job['output'])
    assert ss == ds and sj['image_id'] == dj['image_id'] and sj['source_hashes'] == dj['source_hashes']
    for spec, left, right in ((job, si, di), (repair_job, sri, dri)):
        assert all(left[key] == right[key] for key in MATCHED_PROTOCOL)
        assert left['content_sha256'] == spec['primary_content_sha256']
        assert right['dense_execution_variant']['source_sha256'] == profile['kernel_sha256']
    return dict(status='complete_paired_multipl', original_full_model_accuracy=None, image_id=sj['image_id'],
                invalid_original_fixtures=sorted(sparse_repaired), sparse=se, dense=de,
                original_valid=paired(sparse, dense), mac_repaired_fixtures=paired(sparse_repaired, dense_repaired),
                mac_repaired_composite=paired(sparse | sparse_repaired, dense | dense_repaired),
                profile_sha256=digest(PROFILE), repaired_profile_sha256=digest(REPAIRED_PROFILE),
                auditor_sha256=digest(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sparse-audit', required=True, type=Path)
    parser.add_argument('--dense-audit', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    assert args.out.parent.resolve() == Path('runs').resolve() and not args.out.exists()
    write_json(args.out, compare(args.sparse_audit, args.dense_audit))


if __name__ == '__main__':
    main()
