"""Publisher MultiPL-E eval_script, plus a test-completion sentinel safeguard."""
import base64
import gzip
import hashlib
import json
import secrets
from pathlib import Path

MODULES = dict(py='eval_python',cpp='eval_cpp',java='eval_java',php='eval_php',ts='eval_ts',cs='eval_cs',sh='eval_sh',js='eval_javascript')


def assemble(language, prompt, completion, tests, sentinel):
    lines = dict(py=f'print("{sentinel}")', cpp=f'std::cout << "{sentinel}\\n";',
        java=f'System.out.println("{sentinel}");', cs=f'System.Console.WriteLine("{sentinel}");',
        php=f'echo "{sentinel}\\n";',ts=f'console.log("{sentinel}");',js=f'console.log("{sentinel}");',sh=f'printf "%s\\n" "{sentinel}"')
    if language in ('java','cs'):
        before, middle, after = tests.rsplit('}',2);tests = before+'\n'+lines[language]+'\n}'+middle+'}'+after
    elif language == 'cpp':
        before, after = tests.rsplit('}',1);tests = before+'\n'+lines[language]+'\n}'+after
    else:tests += '\n'+lines[language]+'\n'
    return prompt+completion+'\n'+tests


def execute(sandbox, language, row, completion):
    hashes = json.loads(Path('multipl_judge_sources.lock.json').read_text())
    root = next(Path('.cache/multipl-source').glob('MultiPL-E-*'))/'evaluation/src'
    sources = {name:(root/name).read_text() for name in hashes}
    if any(hashlib.sha256(sources[n].encode()).hexdigest()!=h for n,h in hashes.items()):raise ValueError('Publisher evaluator drift')
    sentinel = 'QWEN_TESTS_COMPLETED_'+secrets.token_hex(16)
    program = assemble(language,row['prompt'],completion,row['tests'],sentinel)
    payload = dict(language=language,module=MODULES[language],program=program,sentinel=sentinel,sources=sources,source_hashes=hashes)
    packed = base64.b64encode(gzip.compress(json.dumps(payload).encode(),mtime=0)).decode()
    code = 'PAYLOAD_BASE64 = '+repr(packed)+'\n'+Path('multipl_driver.py').read_text()
    execution = sandbox.run(code,seconds=90,memory_mb=1024,cpu_seconds=60,output_bytes=1_048_576,tmp_exec=True)
    if execution['status'] != 'exited_zero':raise RuntimeError('MultiPL-E infrastructure failure: '+execution['job_directory']+' '+execution['status'])
    markers = [line.removeprefix('QWEN_MULTIPL_RESULT ') for line in execution['output'].splitlines() if line.startswith('QWEN_MULTIPL_RESULT ')]
    if len(markers)!=1:raise ValueError('Missing/ambiguous isolated judge result: '+execution['job_directory'])
    result = json.loads(markers[0])
    if result['source_hashes']!=hashes or result['program_sha256']!=hashlib.sha256(program.encode()).hexdigest():raise ValueError('Judge provenance drift')
    result.update(execution_directory=execution['job_directory'],image_id=sandbox.image)
    return result
