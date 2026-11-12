"""Independently reconstruct a full formal-control task's saved public results."""
import argparse
import json
from pathlib import Path
from tokenizers import Tokenizer
from qwen35_quality_data import PROTOCOL, load, digest, selection, adapter_for, training_binding, encoded_hash, runtime_versions, write_json


def audit(predictions, allow_partial=False):
    meta = load(str(predictions) + '.meta.json')
    identity = meta['identity']
    protocol = load(PROTOCOL)
    assert identity['protocol_sha256'] == digest(PROTOCOL)
    from evaluate_qwen35_quality import runtime_sources
    assert identity['runtime_sources'] == runtime_sources()
    binding = identity['binding']
    assert training_binding(binding['training_profile'], binding['training_audit'], binding['checkpoint'], binding['arm']) == binding
    assert identity['generation'] == protocol['generation']
    assert identity['runtime_versions'] == runtime_versions()
    assert identity['vocab_size'] == load(binding['training_profile'])['config']['vocab_size']
    assert meta['status'] in ['complete', 'time_budget', 'memory_pressure', 'interrupted_signal']
    job = selection(protocol, identity['binding']['arm'], identity['job'])
    assert identity['job_spec'] == job
    adapter, tasks = adapter_for(job)
    table = {task.name: task for task in tasks}
    spec = protocol['tokenizer'];assert digest(spec['file']) == spec['sha256'] == identity['tokenizer_sha256']
    tokenizer = Tokenizer.from_file(spec['file'])
    expected = []
    for task in tasks:
        for index, (prompt, gold) in enumerate(task.rows):
            ids = tokenizer.encode(prompt, add_special_tokens=False).ids
            expected.append((len(ids), task.name, index, prompt, gold, ids))
    expected.sort(key=lambda row: row[:3])
    counts = {task.name: dict(n=0, scored=0, correct=0, unknown_tokenizer_items=0) for task in tasks}
    seen = set();items = 0;examples = [];content = __import__('hashlib').sha256()
    prefixes = {}
    for _, name, index, prompt, gold, ids in expected:
        task = table[name]
        content.update(json.dumps((name, index, task.shots, task.cot, task.kind, prompt, gold, ids), ensure_ascii=False).encode())
        groups = getattr(task, 'prefix_groups', None)
        key = name if groups is None else name + '/' + str(groups[index])
        if key not in prefixes:prefixes[key] = list(ids)
        else:
            common = prefixes[key];length = min(len(common), len(ids))
            for position in range(length):
                if common[position] != ids[position]:length = position;break
            prefixes[key] = common[:length]
    assert content.hexdigest() == identity['content_sha256']
    prefixes = {key: value[:len(value)//job['prefill_chunk']*job['prefill_chunk']] if job['prefix_cache'] else [] for key, value in prefixes.items()}
    assert encoded_hash(prefixes) == identity['prefix_plan_sha256']
    with Path(predictions).open() as stream:
        for line in stream:
            row = json.loads(line)
            assert items < len(expected)
            _, name, index, prompt, gold, ids = expected[items];task = table[name]
            assert (row['task'], row['index']) == (name, index) and (name, index) not in seen
            assert (row['prompt'], row['gold'], row['input_ids'], row['answer_kind'], row['shots'], row['cot']) == (prompt, gold, ids, task.kind, task.shots, task.cot)
            assert row['prompt_sha256'] == __import__('hashlib').sha256(prompt.encode()).hexdigest()
            output = row['output_ids']
            assert output and len(output) <= job['tokens'] and all(type(token) is int and 0 <= token < identity['vocab_size'] for token in output)
            assert len(ids) + job['tokens'] <= job['context']
            raw = tokenizer.decode(output, skip_special_tokens=False)
            stops = adapter.stopping(name) if hasattr(adapter, 'stopping') else []
            prediction = adapter.completion(tokenizer.decode(output, skip_special_tokens=True), stops) if hasattr(adapter, 'completion') else raw
            unknown = [token for token in output if tokenizer.id_to_token(token) is None]
            assert row['raw_prediction'] == raw and row['prediction'] == prediction and row['unused_token_ids'] == unknown
            assert row['correct'] == adapter.correct(prediction, gold, task.kind)
            assert row['stop'] in ['eos', 'stop_string', 'max_new_tokens']
            if row['stop'] == 'eos':assert output[-1] == protocol['eos'] and protocol['eos'] not in output[:-1]
            elif row['stop'] == 'max_new_tokens':assert len(output) == job['tokens'] and protocol['eos'] not in output
            else:
                assert stops and protocol['eos'] not in output
                tail = tokenizer.decode(output[-64:], skip_special_tokens=False)
                assert any(stop in tail for stop in stops)
            if stops:
                # Reproduce the generation callback, including its 64-ID tail;
                # a stop reason at the end cannot excuse earlier overgeneration.
                assert not any(any(stop in tokenizer.decode(output[max(0, end-64):end], skip_special_tokens=False) for stop in stops) for end in range(1, len(output)))
                ended = any(stop in tokenizer.decode(output[-64:], skip_special_tokens=False) for stop in stops)
                assert row['stop'] == 'eos' or ended == (row['stop'] == 'stop_string')
            count = counts[name];count['n'] += 1;count['unknown_tokenizer_items'] += bool(unknown)
            if row['correct'] is not None:count['scored'] += 1;count['correct'] += int(row['correct'])
            seen.add((name, index));items += 1
            if len(examples) < 3:examples.append({k: row[k] for k in ['task', 'index', 'raw_prediction', 'gold', 'correct']})
    assert items == meta['completed_items'] and items <= job['target_items']
    assert counts == {name: {key: meta['tasks'][name][key] for key in count} for name, count in counts.items()}
    complete = items == job['target_items']
    assert complete == meta['full_dataset_generated']
    assert (meta['status'] == 'complete') == complete
    if not allow_partial:assert complete
    total = sum(count['scored'] for count in counts.values())
    return dict(status='passed_source_output_audit',predictions=str(predictions),predictions_sha256=digest(predictions),meta_sha256=digest(str(predictions)+'.meta.json'),items=items,target=job['target_items'],full_dataset_generated=complete,coverage='full' if complete else 'partial',correct=sum(count['correct'] for count in counts.values()),micro_accuracy=sum(count['correct'] for count in counts.values())/total if total else None,tasks=counts,examples=examples,identity=identity,auditor_sha256=digest(__file__),scope='Public source prompts/gold/IDs/order/raw decoding/stops/scores; code remains unscored until isolated published judging. Padded model IDs are retained and reported, not silently masked.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('predictions');parser.add_argument('--out',required=True);parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
    assert not Path(args.out).exists()
    write_json(args.out,audit(args.predictions,args.allow_partial))
