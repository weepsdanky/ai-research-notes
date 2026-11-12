"""Explicit local integration probe; never part of lightweight unit discovery."""
import json,tempfile
from pathlib import Path
from sandbox import Sandbox,ROOT


def main():
    sandbox=Sandbox();results={}
    def run(name,code,**budgets):
        result=sandbox.run(code,**budgets)
        results[name]={k:v for k,v in result.items() if k!='output'}
        results[name]['output_prefix']=result['output'][:120]
        print(name,result['status'],result['exit_code'],flush=True)
        return result
    r=run('uid_and_temp','import os\nassert os.getuid()==65534\nopen("/tmp/probe","w").write("ok")\nprint("ok")')
    assert r['status']=='exited_zero' and r['output'].strip()=='ok'
    r=run('readonly_task','try:\n open("/work/task.py","w").write("bad")\nexcept OSError:\n print("blocked")\nelse:\n raise RuntimeError("writable task mount")')
    assert r['status']=='exited_zero' and r['output'].strip()=='blocked'
    directory=Path(tempfile.mkdtemp(prefix='sandbox-boundary-',dir=ROOT/'runs'))
    fixture=directory/'fixture.txt';fixture.write_text('controlled boundary fixture, not user data')
    r=run('host_boundary',f'try:\n open({str(fixture)!r}).read()\nexcept OSError:\n print("blocked")\nelse:\n raise RuntimeError("host file visible")')
    assert r['status']=='exited_zero' and r['output'].strip()=='blocked'
    r=run('network','import socket\ns=socket.socket();s.settimeout(.5)\ntry:\n s.connect(("1.1.1.1",443))\nexcept OSError:\n print("blocked")\nelse:\n raise RuntimeError("network reachable")')
    assert r['status']=='exited_zero' and r['output'].strip()=='blocked'
    r=run('pids','import subprocess\ntry:\n children=[subprocess.Popen(["sleep","10"]) for _ in range(50)]\nexcept OSError:\n print("blocked")\nelse:\n raise RuntimeError("pid cap missing")')
    assert r['status']=='exited_zero' and r['output'].strip()=='blocked'
    r=run('wall_timeout','import time;time.sleep(30)',seconds=1)
    assert r['status']=='wall_timeout'
    r=run('cpu_limit','while True: pass',seconds=8,cpu_seconds=1)
    assert r['exit_code']!=0 and not r['oom_killed'] and r['seconds']<8
    r=run('memory_limit','x=bytearray(768*1024*1024)',memory_mb=128)
    assert r['status']=='memory_limit' and r['oom_killed']
    r=run('output_limit','print("a"*(2*1024*1024))',output_bytes=32768)
    assert r['status']=='output_limit' and len(r['output'])<=32768
    assert all(r['cleanup_ok'] for r in results.values())
    Path('runs/sandbox-probe.json').write_text(json.dumps(dict(passed=True,cases=results,
        scope='local infrastructure isolation/resource probe; not EvalPlus or MultiPL-E scoring'),indent=2)+'\n')


if __name__=='__main__':main()
