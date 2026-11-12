"""Executed only inside the offline task container, never on the host.

Calls the pinned official judge, oracles, data conversion and timeouts. The
host injects a small task payload and package source hashes into this script.
"""
import base64
import gzip
import hashlib
import json
import os
import io
import builtins
import resource
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch


class UnusedOracleCache(io.RawIOBase):
    """Discard an optional disk cache, not the computed oracle values."""
    def writable(self):return True
    def write(self, data):return len(data)


def main(payload):
    dataset = payload['dataset']
    if dataset not in ('humaneval', 'mbpp'):
        raise ValueError('Unsupported official dataset')
    os.environ['XDG_CACHE_HOME'] = '/tmp/cache'
    file = Path('/tmp/problems.jsonl')
    file.write_text('\n'.join(json.dumps(row) for row in payload['problems'])+'\n')
    os.environ['HUMANEVAL_OVERRIDE_PATH' if dataset == 'humaneval' else 'MBPP_OVERRIDE_PATH'] = str(file)
    import evalplus
    import evalplus.evaluate as evaluator
    from evalplus.sanitize import sanitize
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    from evalplus.eval import PASS
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    package = Path(evalplus.__file__).parent
    if version('evalplus') != '0.3.1':
        raise ValueError('Unexpected official package version')
    for relative, expected in payload['package_sources'].items():
        if hashlib.sha256((package/relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Official evaluator source drift')
    problems = get_human_eval_plus() if dataset == 'humaneval' else get_mbpp_plus()
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    cache_file = str(Path(evaluator.CACHE_DIR)/(digest+'.pkl'))
    def cache_open(path, mode='r', *args, **kwargs):
        if str(path) == cache_file and mode == 'wb':return UnusedOracleCache()
        return builtins.open(path, mode, *args, **kwargs)
    # A fresh container calls this once; no later call can reuse its pickle.
    # Keep the official oracle computation intact, discard only cache writes.
    with patch.object(evaluator, 'open', cache_open, create=True):
        oracle = evaluator.get_groundtruth(problems, digest, [] if dataset == 'humaneval' else MBPP_OUTPUT_NOT_NONE_TASKS)
    results = []
    for sample in payload['samples']:
        task = sample['task_id']
        solution = sanitize(sample['solution'], entrypoint=problems[task]['entry_point']) if payload.get('sanitize', True) else sample['solution']
        result = evaluator.check_correctness(dataset, 0, problems[task], solution, oracle[task],
                                            base_only=False, fast_check=True, identifier=task)
        item = dict(task_id=task, raw_solution_sha256=hashlib.sha256(sample['solution'].encode()).hexdigest(),
                    solution_sha256=hashlib.sha256(solution.encode()).hexdigest(), sanitized_solution=solution)
        for mode in ('base', 'plus'):
            status, details = result[mode]
            item[mode] = dict(status=status, passed=status == PASS, total_inputs=len(problems[task][mode+'_input']), tested_inputs=len(details),
                              passed_inputs=sum(bool(v) for v in details), failed_indices=[i for i, v in enumerate(details) if not v])
        results.append(item)
    print('QWEN_EVALPLUS_RESULT '+json.dumps(dict(dataset=dataset, results=results,
        min_time_limit=evaluator.DEFAULT_MIN_TIME_LIMIT, gt_time_limit_factor=evaluator.DEFAULT_GT_TIME_LIMIT_FACTOR,
        fast_check=True,
        oracle_process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        oracle_cache='optional on-disk writes discarded; official values/computation unchanged',
        package_sources=payload['package_sources']), separators=(',', ':')), flush=True)


if __name__ == '__main__':
    main(json.loads(gzip.decompress(base64.b64decode(PAYLOAD_BASE64))))
