"""Compare two fully audited, matched text-evaluation jobs; never aggregate partial jobs."""
import argparse
import hashlib
import itertools
import json
from datetime import datetime
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--profile', default='profiles/paper_dense_evaluation.json')
    parser.add_argument('--sparse-audit', required=True)
    parser.add_argument('--dense-audit', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    output = Path(args.out)
    assert not output.exists(), 'Preserve existing comparison receipts'
    profile_path = args.profile
    profile = load(profile_path)
    job = next(row for row in profile['jobs'] if row['name'] == args.job)
    assert job['suite'] in ('core', 'extra', 'mmmlu'), 'Code requires its own execution scorer'
    binding = load('runs/paper-dense-evaluation-binding.json')
    assert binding['profile_sha256'] == profile.get('parent_profile_sha256', digest(profile_path))
    if 'engine_sha256' in profile:
        assert profile['engine_sha256'] == digest('evaluate_dense_bounded.py') and profile['kernel_sha256'] == digest('bounded_dense_inference.py')
        assert digest(profile['accepted_batch_probe']) == profile['accepted_batch_probe_sha256'] and load(profile['accepted_batch_probe'])['greedy_matches_saved_dense_batch']
    assert digest(job['primary_meta']) == job['primary_meta_sha256']
    sparse = job['primary_meta'].removesuffix('.meta.json')
    dense = job.get('output', profile['outputs_prefix'] + job['name'] + '-full.jsonl')
    entries = []; identities = []; files = {}
    for filename, audit_path in [(sparse, args.sparse_audit), (dense, args.dense_audit)]:
        audit = load(audit_path); entry = audit['files'][filename]
        assert audit['scorer_source_sha256'] == digest({'core':'eval_tasks.py','extra':'eval_extra_tasks.py','mmmlu':'eval_mmmlu_tasks.py'}[job['suite']])
        assert entry['full_dataset_generated'] and entry['items'] == entry['target_items'] == job['target_items']
        assert entry['all_source_prompts_gold_input_ids_raw_decoding_and_scores_verified']
        assert digest(filename) == entry['file_sha256']
        meta = load(filename + '.meta.json')
        assert meta['full_dataset_generated'] and meta['status'] == 'complete'
        identities.append(meta['identity']); entries.append(entry)
        files[filename] = dict(sha256=entry['file_sha256'], meta_sha256=digest(filename + '.meta.json'), audit=audit_path, audit_sha256=digest(audit_path))
    control = load('profiles/paper_dense_control.json')
    assert identities[0]['checkpoint_sha256'] == control['comparison_sha256']
    assert identities[1]['checkpoint_sha256'] == binding['checkpoint_sha256']
    differences = {key: [identities[0].get(key), identities[1].get(key)] for key in identities[0].keys() | identities[1].keys() if identities[0].get(key) != identities[1].get(key)}
    allowed = {'checkpoint_sha256', 'stop_pressure'}
    if 'engine_sha256' in profile:allowed |= {'source_hashes', 'dense_execution_variant'}
    assert set(differences) <= allowed, differences
    if 'engine_sha256' in profile:
        assert identities[1]['dense_execution_variant']['source_sha256'] == profile['kernel_sha256']
        for identity in identities:
            assert all(digest(path) == sha for path, sha in identity['source_hashes'].items())
    counts = dict(both_correct=0, dense_only_correct=0, sparse_only_correct=0, neither_correct=0)
    matched = 0
    fields = ('task', 'index', 'shots', 'cot', 'answer_kind', 'prompt', 'input_ids', 'prompt_sha256', 'gold', 'prompt_tokens', 'context_extrapolation')
    with Path(sparse).open() as left, Path(dense).open() as right:
        for a, b in itertools.zip_longest(left, right):
            assert a is not None and b is not None
            a, b = json.loads(a), json.loads(b)
            assert all(a[key] == b[key] for key in fields), (a['task'], a['index'])
            key = 'both_correct' if a['correct'] and b['correct'] else 'sparse_only_correct' if a['correct'] else 'dense_only_correct' if b['correct'] else 'neither_correct'
            counts[key] += 1; matched += 1
    assert matched == job['target_items']
    assert counts['both_correct'] + counts['sparse_only_correct'] == entries[0]['correct']
    assert counts['both_correct'] + counts['dense_only_correct'] == entries[1]['correct']
    assert all(digest(path) == record['sha256'] and digest(path + '.meta.json') == record['meta_sha256'] for path, record in files.items())
    result = dict(at=datetime.now().astimezone().isoformat(), source_sha256=digest(__file__), profile_sha256=digest(profile_path), job=args.job, matched_items=matched, files=files, identity_differences=differences, paired_correctness=counts, sparse_accuracy=entries[0]['micro_accuracy'], dense_accuracy=entries[1]['micro_accuracy'], scope='Full matched Mac text job only; no eight-group aggregate, statistical equivalence, official-scale quality or kernel-speed claim')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('job', 'matched_items', 'paired_correctness')}))


if __name__ == '__main__':
    main()
