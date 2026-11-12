"""Audit full extra-task input capacity and exact-length batching before generation."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from tokenizers import Tokenizer
from eval_extra_tasks import load_tasks
from evaluate_suite import now


def main():
    path = Path('.cache/references/tokenizer.json')
    tokenizer = Tokenizer.from_file(str(path))
    families = {}
    for family, prefix, expected in [('redux', 'redux2-ok-', 5330),
                                     ('include', 'include-', 22639),
                                     ('supergpqa', 'supergpqa', 26529)]:
        subjects = {}; lengths = defaultdict(list)
        for task in load_tasks():
            if not task.name.startswith(prefix):continue
            counts = Counter(len(tokenizer.encode(p, add_special_tokens=False).ids) for p, _ in task.rows)
            subjects[task.name] = dict(items=sum(counts.values()), min=min(counts), max=max(counts))
            for size, count in counts.items():lengths[size].extend([task.name]*count)
        groups = [sorted(names)[i:i+8] for names in lengths.values() for i in range(0,len(names),8)]
        count = sum(s['items'] for s in subjects.values())
        if count != expected:raise ValueError(f'{family} public coverage drift')
        maximum = max(lengths);context = 1 << (maximum+64-1).bit_length()
        families[family] = dict(items=count, min=min(lengths), max=maximum, context=context,
            subjects=subjects, batch_groups=len(groups), average_batch_items=count/len(groups),
            homogeneous_task_groups=sum(len(set(group))==1 for group in groups))
        print(family, count, maximum, context, flush=True)
    result = dict(at=now(), tokenizer_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), families=families)
    Path('runs/remaining-extra-context-audit.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':main()
