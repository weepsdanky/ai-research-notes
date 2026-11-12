"""Validate formal Base code generations and prepare pinned isolated judge jobs.

No VM or model is started here. Reuses public problems and execution helpers,
without relabeling the older Qwen3.8 predictions/tokenizer/quality identities.
"""
import json
from pathlib import Path
from audit_qwen35_quality import audit
from qwen35_quality_data import load, digest, encoded_hash


def validated_jobs(predictions, generation_audit):
    metadata = load(str(predictions) + '.meta.json')
    identity = metadata['identity']
    binding = identity.get('binding', {})
    if binding.get('long_step') != 20 or binding.get('total_training_tokens') != 3932160:
        raise ValueError('Requires the independent formal800+20 Base generation identity')
    suite = identity.get('job')
    if suite not in ['code', 'multipl']:
        raise ValueError('Select one complete formal code suite')
    if metadata['status'] != 'complete' or not metadata['full_dataset_generated']:
        raise ValueError('Partial or capacity-failed generation is ineligible')
    # Rebuild source/IDs/raw decoding/stops and formal weight eligibility. The
    # saved audit is also required, rather than accepting a metadata assertion.
    rebuilt = audit(predictions)
    audit_sha = digest(generation_audit)
    saved = load(generation_audit)
    if Path(saved['predictions']).resolve() != Path(predictions).resolve():
        raise ValueError('Generation audit points to another output')
    for key, value in rebuilt.items():
        if key != 'predictions' and saved.get(key) != value:
            raise ValueError('Generation audit changed: ' + key)
    if rebuilt['status'] != 'passed_source_output_audit' or not rebuilt['full_dataset_generated']:
        raise ValueError('Complete independently audited generation required')
    protocol_path = 'profiles/qwen35_code_judging_protocol.json'
    protocol = load(protocol_path)
    sources = {**protocol['common_sources'], **protocol['suites'][suite]['source_hashes']}
    if any(digest(name) != sha for name, sha in sources.items()):
        raise ValueError('Predeclared public problem/test/runner source drift')
    if suite == 'code':
        from eval_code_tasks import datasets
        problems = {(name, row['task_id']): row for name, rows in datasets() for row in rows}
        expected = 542
    else:
        from eval_multipl_tasks import datasets
        problems = {(name, row['name']): row for name, rows in datasets().items() for row in rows}
        expected = 1277
    if len(problems) != expected or rebuilt['items'] != expected:
        raise ValueError('All public tasks are required before judging')
    if protocol['suites'][suite]['target_items'] != expected:
        raise ValueError('Do not shrink the declared judging target')
    jobs = {}
    prefix = 'evalplus-' if suite == 'code' else 'multipl-'
    with Path(predictions).open() as stream:
        for line in stream:
            row = json.loads(line)
            if not row['task'].startswith(prefix):
                raise ValueError('Mixed generation suite')
            key = row['task'][len(prefix):], row['gold']
            if key not in problems or key in jobs:
                raise ValueError('Missing, unknown or duplicate public problem')
            problem = problems[key]
            if row['correct'] is not None or row['prompt'] != problem['prompt']:
                raise ValueError('Requires unscored original public code prompts')
            jobs[key] = dict(suite=suite, dataset=key[0], name=key[1],
                             problem=problem, completion=row['prediction'],
                             solution=problem['prompt'] + row['prediction'])
    if jobs.keys() != problems.keys():
        raise ValueError('Incomplete source problem coverage')
    if (digest(predictions) != rebuilt['predictions_sha256'] or
            digest(str(predictions)+'.meta.json') != rebuilt['meta_sha256'] or
            digest(generation_audit) != audit_sha):
        raise ValueError('Generation/audit changed while preparing jobs')
    result = [jobs[key] for key in sorted(jobs)]
    provenance = dict(generation_identity=identity, predictions=str(predictions),
                      predictions_sha256=digest(predictions), generation_audit=str(generation_audit),
                      generation_audit_sha256=audit_sha,
                      adapter_sha256=digest(__file__), target_items=expected,
                      judging_protocol_sha256=digest(protocol_path),
                      public_source_hashes={name: digest(name) for name in
                          (['eval_code_tasks.py', 'eval_sources.lock.json'] if suite == 'code'
                           else ['eval_multipl_tasks.py', 'multipl_sources.lock.json'])},
                      problem_content_sha256=encoded_hash([(name, key, problems[name,key]) for name,key in sorted(problems)]),
                      judging_executed=False,
                      limitations='Uses pinned public judges; private Qwen harness unknown. MultiPL-E raw scaffolds include three invalid fixtures: raw pass counts are not an aggregate model accuracy. Separate declared repairs remain separate.')
    return provenance, result
