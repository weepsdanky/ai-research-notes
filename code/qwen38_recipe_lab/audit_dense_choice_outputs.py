"""Diagnose the extracted choices in fully audited Table 2 text pairs."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from eval_tasks import answer
from qwen35_quality_data import digest, load, write_json

PROFILE = Path('profiles/paper_dense_bounded_evaluation.json')


def parser_branch(text):
    if re.search(r'\banswer\s*(?:is|:)\s*\(?([A-Z])\)?\b', text, re.IGNORECASE):
        return 'answer_marker'
    if re.search(r'\(([A-Z])\)', text):
        return 'parenthesized'
    if re.match(r'^\s*([A-Z])[.)]?\s*(?:\n|$)', text):
        return 'leading_bare_letter'
    return 'unparsed'


def audit_arm(file, evidence, target):
    assert digest(evidence['audit']) == evidence['audit_sha256']
    checked = load(evidence['audit'])['files'][file]
    assert checked['full_dataset_generated'] and checked['items'] == target
    assert checked['all_source_prompts_gold_input_ids_raw_decoding_and_scores_verified']
    extracted, correct, gold, stops = Counter(), Counter(), Counter(), Counter()
    branches, correct_by_branch = Counter(), Counter()
    subjects = defaultdict(Counter)
    stream_hash = hashlib.sha256()
    with Path(file).open('rb') as stream:
        for line in stream:
            stream_hash.update(line)
            row = json.loads(line)
            assert row['answer_kind'] == 'choice'
            letter = answer(row['prediction'], 'choice')
            expected = answer(row['gold'], 'choice')
            branch = parser_branch(row['prediction'])
            assert expected and row['correct'] == (letter == expected)
            assert (letter is None) == (branch == 'unparsed')
            extracted[letter or 'no_extracted_letter'] += 1
            gold[expected] += 1
            stops[row['stop']] += 1
            branches[branch] += 1
            correct_by_branch[branch] += row['correct']
            subject = subjects[row['task']]
            subject['items'] += 1
            subject['extracted'] += letter is not None
            subject['correct'] += row['correct']
            if row['correct']:
                correct[letter] += 1
    assert gold.total() == extracted.total() == target
    assert correct.total() == correct_by_branch.total() == checked['correct']
    assert stream_hash.hexdigest() == evidence['sha256'] == checked['file_sha256']
    return dict(extracted=extracted, correct_by_letter=correct, gold=gold, stops=stops,
                subjects=subjects, parser_branches=branches, correct_by_parser_branch=correct_by_branch)


def audit_pair(path, jobs):
    pair = load(path)
    job = jobs[pair['job']]
    target = job['target_items']
    assert pair['matched_items'] == target and digest(job['primary_meta']) == job['primary_meta_sha256']
    files = dict(sparse=job['primary_meta'].removesuffix('.meta.json'), dense=job['output'])
    assert set(files.values()) == pair['files'].keys()
    modes = {mode: audit_arm(file, pair['files'][file], target) for mode, file in files.items()}
    assert modes['sparse']['gold'] == modes['dense']['gold']
    return dict(job=pair['job'], items_per_arm=target, modes=modes, pair=str(path), pair_sha256=digest(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', nargs='+', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    assert args.out.parent.resolve() == Path('runs').resolve() and not args.out.exists()
    jobs = {job['name']: job for job in load(PROFILE)['jobs']
            if job['name'] in ('mmlu-pro', 'supergpqa') or job['suite'] == 'mmmlu'}
    pairs = [audit_pair(path, jobs) for path in args.pairs]
    assert len({pair['job'] for pair in pairs}) == len(pairs)
    write_json(args.out, dict(pairs=pairs, scorer_sha256=digest('eval_tasks.py'),
                              auditor_sha256=digest(__file__), profile_sha256=digest(PROFILE)))


if __name__ == '__main__':
    main()
