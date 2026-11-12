"""Prepare full locked public prompts for Mac MTP acceptance measurements."""
import argparse
import hashlib
import json
from pathlib import Path

from qwen35_quality_data import adapter_for, digest, load, write_json

PROFILE = Path('profiles/mtp_acceptance_mac_public.json')


def public_items(profile, job, tokenizer):
    spec = profile['jobs'][job]
    source = load(profile['benchmark_protocol'])
    protocol, = [entry for entry in source['jobs'] if entry['name']==spec['benchmark_job']]
    _, tasks = adapter_for(protocol)
    rows = []
    identity = hashlib.sha256()
    for task in tasks:
        if spec.get('task') and task.name!=spec['task']:
            continue
        for index, (prompt, gold) in enumerate(task.rows):
            ids = tokenizer.encode(prompt, add_special_tokens=False).ids
            if not ids or len(ids)+profile['new_tokens']>profile['context_limit']:
                raise ValueError(f'Public prompt exceeds context capacity: {task.name}/{index}')
            identity.update(json.dumps((task.name,index,task.shots,task.cot,prompt,gold,ids),
                                       ensure_ascii=False).encode())
            rows.append(dict(group=job,task=task.name,source_id=index,input_ids=ids))
    assert len(rows)==spec['items'] and len({(row['task'],row['source_id']) for row in rows})==len(rows)
    return rows,identity.hexdigest()


def profile_and_tokenizer(path=PROFILE):
    from tokenizers import Tokenizer

    profile = load(path)
    assert profile['tokenizer_sha256']==digest(profile['tokenizer'])
    assert all(digest(path)==sha for path,sha in profile['source_hashes'].items())
    return profile,Tokenizer.from_file(profile['tokenizer'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-out', required=True, type=Path)
    args = parser.parse_args()
    assert args.check_out.parent.resolve()==Path('runs').resolve() and not args.check_out.exists()
    profile,tokenizer = profile_and_tokenizer()
    result = {}
    for job in profile['jobs']:
        rows,sha = public_items(profile,job,tokenizer)
        sizes = [len(row['input_ids']) for row in rows]
        result[job] = dict(items=len(rows),min_prompt_tokens=min(sizes),max_prompt_tokens=max(sizes),
                           source_tasks=sorted({row['task'] for row in rows}),content_sha256=sha)
    write_json(args.check_out,dict(status='full_public_prompts_verified',profile_sha256=digest(PROFILE),
                                   preparer_sha256=digest(__file__),jobs=result))


if __name__=='__main__':
    main()
