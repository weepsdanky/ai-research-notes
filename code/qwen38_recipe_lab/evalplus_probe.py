"""Explicit official-judge integration controls, not model benchmark scores."""
import json
from pathlib import Path
from eval_code_tasks import datasets
from judge_evalplus import execute
from sandbox import Sandbox


def main():
    sandbox = Sandbox('evalplus',large=True);results = {}
    for dataset, problems in datasets():
        problem = problems[0];entry = problem['entry_point']
        controls = [
            ('canonical', problem['prompt']+problem['canonical_solution'], True),
            ('wrong', f'def {entry}(*args):\n    return None\n', False),
            ('exit_zero', 'import os\nos._exit(0)\n', False),
            ('system_exit', 'raise SystemExit(0)\n', False),
            ('syntax_error', 'this is not valid Python !', False),
            ('loop', f'def {entry}(*args):\n    while True: pass\n', False),
        ]
        samples = [dict(task_id=problem['task_id'], solution=code) for _, code, _ in controls]
        judged, execution = execute(sandbox, dataset, [problem], samples, sanitize=False)
        for (name, _, expected), result in zip(controls, judged['results'], strict=True):
            assert result['base']['passed'] is expected and result['plus']['passed'] is expected, (dataset, name, result)
            results[dataset+'-'+name] = result
            print(dataset, name, result['base']['status'], result['plus']['status'], flush=True)
        assert execution['cleanup_ok']
        results[dataset+'-execution'] = dict(directory=execution['job_directory'], image_id=execution['image_id'],
                                             seconds=execution['seconds'], cleanup_ok=execution['cleanup_ok'])
    # Also exercise the actual public sanitizer path, not just direct judging.
    dataset, problems = next(iter(datasets()));p = problems[0]
    judged, execution = execute(sandbox, dataset, [p], [dict(task_id=p['task_id'], solution=p['prompt']+p['canonical_solution'])])
    assert judged['results'][0]['base']['passed'] and judged['results'][0]['plus']['passed']
    results['sanitizer-canonical'] = judged['results'][0]
    Path('runs/evalplus-probe.json').write_text(json.dumps(dict(passed=True, controls=results,
        scope='Official judge integration controls only; no trained-model benchmark score'), indent=2)+'\n')


if __name__ == '__main__':
    main()
