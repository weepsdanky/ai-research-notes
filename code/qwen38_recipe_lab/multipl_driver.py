"""Runs only inside the offline VM container, never on the host."""
import base64
import gzip
import hashlib
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path


payload = json.loads(gzip.decompress(base64.b64decode(PAYLOAD_BASE64)))
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    for name, source in payload['sources'].items():
        if hashlib.sha256(source.encode()).hexdigest() != payload['source_hashes'][name]:raise ValueError('Publisher source drift')
        path = root/name;path.parent.mkdir(parents=True, exist_ok=True);path.write_text(source)
    sys.path.insert(0, str(root));os.chdir(root)
    module = importlib.import_module(payload['module'])
    program = root/('candidate.'+payload['language']);program.write_text(payload['program'])
    result = module.eval_script(program)
    # C# can return a bytes error stream on compile failures.
    result = {k:v.decode(errors='replace') if isinstance(v,bytes) else v for k,v in result.items()}
    result['tests_completed'] = payload['sentinel'] in result['stdout'].splitlines()
    result['passed'] = result['status'] == 'OK' and result['exit_code'] == 0 and result['tests_completed']
    result['program_sha256'] = hashlib.sha256(payload['program'].encode()).hexdigest()
    result['source_hashes'] = payload['source_hashes']
    print('QWEN_MULTIPL_RESULT '+json.dumps(result),flush=True)
