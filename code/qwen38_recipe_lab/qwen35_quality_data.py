"""Public task preparation and formal-weight metadata gates; no Torch import."""
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from qwen35_formal_contract import contract


PROTOCOL = 'profiles/qwen35_quality_protocol.json'


def runtime_versions():
    return dict(python=sys.version, platform=platform.platform(),
                packages={name: importlib.metadata.version(name) for name in ['torch', 'tokenizers', 'pyarrow']})


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def encoded_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def selection(protocol, arm, name):
    if arm not in protocol['arms']:
        raise ValueError('Undeclared arm')
    jobs = [job for job in protocol['jobs'] if job['name'] == name]
    if len(jobs) != 1:
        raise ValueError('Select one complete declared job')
    return jobs[0]


def training_binding(training_profile, audit_path, checkpoint, arm):
    """Metadata gate only; the worker must also check the loaded header."""
    protocol = load(PROTOCOL)
    profile = load(training_profile)
    contract(profile, 'formal_long')
    audit = load(audit_path)
    assert profile['arm'] == arm and profile['steps'] == 20 and profile['seq'] == 32768
    assert profile['formal_budget_sha256'] == protocol['formal_budget_sha256']
    assert Path(checkpoint).resolve() == (Path(profile['out']) / 'last.pt').resolve()
    assert audit['status'] == 'passed_checkpoint_checks' and audit['phase_complete']
    assert audit['step'] == audit['target_steps'] == 20 and audit['total_tokens'] == 3932160
    assert audit['parent_model_optimizer_rng_bitwise_preserved']
    assert audit['auditor_sha256'] == digest('audit_qwen35_formal_long.py')
    assert audit['source_hashes'] == profile['source_hashes']
    assert audit['profile_sha256'] == digest(training_profile)
    assert audit['checkpoint_sha256'] == digest(checkpoint)
    assert audit['parent_checkpoint_sha256'] == profile['parent_checkpoint_sha256'] == digest(profile['parent_checkpoint'])
    assert audit['initial_checkpoint_sha256'] == digest(Path(profile['out']) / 'initial.pt')
    spec = protocol['tokenizer']
    assert digest(spec['file']) == spec['sha256']
    return dict(training_profile=training_profile, training_profile_sha256=digest(training_profile),
                training_audit=audit_path, training_audit_sha256=digest(audit_path),
                checkpoint=checkpoint, checkpoint_sha256=audit['checkpoint_sha256'],
                arm=arm, long_step=20, total_training_tokens=3932160)


def check_header(saved, profile, binding):
    assert saved['profile'] == profile and saved['profile_sha256'] == binding['training_profile_sha256']
    assert saved['source_hashes'] == profile['source_hashes'] and saved['config'] == profile['config']
    assert saved['arm'] == binding['arm'] and saved['phase'] == '32k_continuation'
    assert saved['step'] == 20 and saved['phase_tokens'] == 655360 and saved['tokens'] == 3932160
    assert saved['optimizer_steps_offset'] == 800 and saved['parent_training_tokens'] == 3276800
    assert saved['parent_checkpoint_sha256'] == profile['parent_checkpoint_sha256']


def adapter_for(job):
    module = {'core': 'eval_tasks', 'extra': 'eval_extra_tasks', 'code': 'eval_code_tasks'}.get(job['suite'], 'eval_' + job['suite'] + '_tasks')
    adapter = importlib.import_module(module)
    names = set(job['tasks'])
    tasks = [task for task in adapter.load_tasks() if task.name in names]
    assert len(tasks) == len(names) and {task.name for task in tasks} == names
    assert sum(len(task.rows) for task in tasks) == job['target_items']
    if hasattr(adapter, 'validate_selection'):
        adapter.validate_selection(tasks)
    return adapter, tasks


def group_key(task, index):
    groups = getattr(task, 'prefix_groups', None)
    return task.name if groups is None else task.name + '/' + str(groups[index])


def prepare_jobs(tasks, tokenizer, job):
    pool = {}
    rows = [(task, index, prompt, gold,
             [pool.setdefault(token, token) for token in tokenizer.encode(prompt, add_special_tokens=False).ids])
            for task in tasks for index, (prompt, gold) in enumerate(task.rows)]
    rows.sort(key=lambda row: (len(row[4]), row[0].name, row[1]))
    assert len(rows) == job['target_items']
    if any(not ids or len(ids) + job['tokens'] > job['context'] for _, _, _, _, ids in rows):
        raise ValueError('Declared context is insufficient; retain every source prompt and declare a new execution profile before generation')
    content = hashlib.sha256()
    groups = {}
    for task, index, prompt, gold, ids in rows:
        content.update(json.dumps((task.name, index, task.shots, task.cot, task.kind, prompt, gold, ids), ensure_ascii=False).encode())
        groups.setdefault(group_key(task, index), []).append(ids)
    prefixes = {}
    for name, inputs in groups.items():
        length = min(map(len, inputs))
        first = inputs[0]
        for other in inputs[1:]:
            for position in range(length):
                if other[position] != first[position]:
                    length = position
                    break
        prefixes[name] = first[:length // job['prefill_chunk'] * job['prefill_chunk']] if job['prefix_cache'] else []
    return rows, content.hexdigest(), prefixes


def batches(rows, size):
    batch = []
    for row in rows:
        if batch and (len(batch) == size or len(batch[0][4]) != len(row[4])):
            yield batch
            batch = []
        batch.append(row)
    if batch:
        yield batch


def decoded(adapter, tokenizer, ids, stops):
    raw = tokenizer.decode(ids, skip_special_tokens=False)
    prediction = adapter.completion(tokenizer.decode(ids, skip_special_tokens=True), stops) if hasattr(adapter, 'completion') else raw
    return raw, prediction


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)
