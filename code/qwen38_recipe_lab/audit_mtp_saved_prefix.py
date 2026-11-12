"""Independently replay a durable MTP prefix without acquiring the writer lock."""
import argparse
from collections import Counter
import hashlib
from itertools import islice
import json
from pathlib import Path
import resource

from evaluate_mtp_public import add, empty_totals
from mtp_public_data import PROFILE, profile_and_tokenizer, public_items
from qwen35_quality_data import digest, load, write_json
from run_control_quality import memory, normal


def paired_prompt_items(job, target, prompt_sha):
    from compare_mtp_attention import PROMPTS, checked_output

    catalog = load(PROMPTS)
    assert catalog['profile_sha256'] == digest(PROFILE)
    spec = catalog['jobs'][job]
    assert (target, prompt_sha) == (spec['items'], spec['content_sha256'])
    raw, evidence = checked_output(job, 'qsa', dict(items=target, prompt_content_sha256=prompt_sha))

    def rows():
        with raw.open() as stream:
            for line in stream:
                row = json.loads(line)
                yield {key: row[key] for key in ('group', 'task', 'source_id', 'input_ids')}

    return rows(), evidence


def audit(raw, checkpoint, prior_path, paired_qsa_prompts=False):
    metadata = Path(f'{raw}.meta.json').read_bytes()
    snapshot = json.loads(metadata)
    identity = snapshot['identity']
    verification = identity['verification']
    assert verification in ('batched', 'serial')
    assert identity['profile_sha256'] == digest(PROFILE)
    assert all(digest(path) == sha for path, sha in identity['sources'].items())
    assert identity['draft_checkpoint_sha256'] == digest(checkpoint)
    count = snapshot['totals']['items']
    prior = load(prior_path) if prior_path else None
    prior_count = prior['totals']['items'] if prior else 0
    if prior:
        assert prior['identity'] == identity and prior['target'] == snapshot['target']
        assert 0 < prior_count <= count

    job = identity['job']
    if paired_qsa_prompts:
        assert identity.get('attention') == 'full' and verification == 'serial' and job != 'mt-bench'

    # These paths stream prompt rows without rebuilding the source dataset in memory.
    def admitted(sample):
        return sample['level'] == 1 if job == 'mt-bench' or paired_qsa_prompts else normal(sample)

    before = memory()
    assert admitted(before)
    profile, tokenizer = profile_and_tokenizer()
    completed = []
    pairing = None
    if 'attention' in identity:
        from evaluate_mtp_attention import prompts, training_binding

        attention = identity['attention']
        assert attention == 'full' or (attention == 'qsa' and job == 'mt-bench')
        assert verification == 'serial'
        training, bound_checkpoint, state = training_binding(attention)
        assert Path(checkpoint).resolve() == bound_checkpoint.resolve()
        assert identity['training_state_sha256'] == digest(state)
        if paired_qsa_prompts:
            target, prompts_sha = snapshot['target'], identity['prompt_content_sha256']
            items, pairing = paired_prompt_items(job, target, prompts_sha)
        else:
            items, target, prompts_sha = prompts(job, profile, tokenizer, completed)
    else:
        training = load(profile['draft_training_profile'])
        prepared, prompts_sha = public_items(profile, job, tokenizer)
        items, target = iter(prepared), len(prepared)
    assert identity['target_checkpoint_sha256'] == digest(training['target'])
    assert 0 < count <= snapshot['target'] == target
    assert identity['prompt_content_sha256'] == prompts_sha
    assert admitted(memory())

    totals = empty_totals(verification)
    sources = Counter()
    hash_state = hashlib.sha256()
    with raw.open('rb') as stream:
        for index, line in enumerate(islice(stream, count), 1):
            assert line.endswith(b'\n')
            hash_state.update(line)
            item, row = next(items), json.loads(line)
            add(totals, row, item, tokenizer, profile)
            if job == 'mt-bench':
                completed.append(row)
                sources[item['category']] += 1
            else:
                sources[item['task']] += 1
            if index == prior_count:
                expected = prior.get('output_prefix_sha256', prior.get('output_sha256'))
                assert hash_state.hexdigest() == expected and totals == prior['totals']
    assert totals == snapshot['totals']
    result = dict(status='verified_durable_snapshot', identity=identity, items=count, target=target,
                  previous_items=prior_count, new_rows=count - prior_count, tasks=sources,
                  output_prefix_sha256=hash_state.hexdigest(), totals=totals,
                  metadata_snapshot_sha256=hashlib.sha256(metadata).hexdigest(),
                  metadata_status=snapshot['status'], prior_audit_sha256=digest(prior_path) if prior else None,
                  resource_samples=[before, memory()], auditor_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  auditor_sha256=digest(__file__))
    if pairing:
        result['prompt_replay'] = dict(source='audited_qsa_prompt_ids', audit_sha256=pairing['audit_sha256'],
                                       output_sha256=pairing['output_sha256'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path('runs/mtp-frozen-draft-mac/last.pt'))
    parser.add_argument('--prior', type=Path)
    parser.add_argument('--paired-qsa-prompts', action='store_true',
                        help='Stream prompts from the complete QSA audit for non-conversation Full Attention jobs')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    runs = Path('runs').resolve()
    assert args.out.parent.resolve() == args.predictions.parent.resolve() == runs
    assert not args.out.exists()
    write_json(args.out, audit(args.predictions, args.checkpoint, args.prior, args.paired_qsa_prompts))


if __name__ == '__main__':
    main()
