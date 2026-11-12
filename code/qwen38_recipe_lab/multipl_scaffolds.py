"""Infrastructure controls: every public test scaffold must invoke its stub.

These are deliberately failing functions, never model benchmark scores.
"""
import argparse
import fcntl
import json
import time
from pathlib import Path
from eval_multipl_tasks import datasets
from evaluate import digest
from evaluate_suite import now,save_meta
from multipl_judge import execute
from sandbox import Sandbox

MARK='QWEN_SCAFFOLD_REACHED'
STUBS=dict(py=f'    print("{MARK}",flush=True)\n    raise RuntimeError("control")\n',
    cpp=f'std::cerr << "{MARK}\\n"; throw std::runtime_error("control");',
    java=f'System.err.println("{MARK}"); throw new RuntimeException("control");',
    cs=f'System.Console.Error.WriteLine("{MARK}"); throw new System.Exception("control");',
    php=f'fwrite(STDERR,"{MARK}\\n"); throw new Exception("control");\n}}',
    ts=f'console.error("{MARK}"); throw new Error("control");\n}}',
    js=f'console.error("{MARK}"); throw new Error("control");\n}}',
    sh=f'printf "%s\\n" "{MARK}" >&2; return 1')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    p.add_argument('--seconds',type=int,default=1200);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if not 1<=a.seconds<=1200:p.error('seconds must be 1..1200')
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);meta=Path(a.out+'.meta.json')
    with Path(a.out+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=sorted((lang,r['name'],r) for lang,rows in datasets().items() for r in rows)
        sandbox=Sandbox('multipl');identity=dict(image_id=sandbox.image,stubs=STUBS,source_hashes={f:digest(f) for f in ['multipl_scaffolds.py','multipl_judge.py','multipl_driver.py','multipl_judge_sources.lock.json','eval_multipl_tasks.py','multipl_sources.lock.json','sandbox.py','vm.sh']})
        done=0
        if a.resume:
            report=json.loads(meta.read_text());assert report['identity']==identity
            for line in out.read_text().splitlines():
                row=json.loads(line);lang,name,_=jobs[done];assert (row['language'],row['name'])==(lang,name);done+=1
        else:
            if out.exists() or meta.exists():raise FileExistsError('New path or explicit resume required')
            out.touch(exist_ok=False);report=dict(identity=identity,started=now(),events=[],not_model_scores=True)
        report['events'].append(dict(start=now(),starting_items=done,budget_seconds=a.seconds))
        report.update(status='running',completed_items=done,target_items=len(jobs));save_meta(meta,report);tick=time.monotonic()
        with out.open('a') as stream:
            for lang,name,problem in jobs[done:]:
                try:result=execute(sandbox,lang,problem,STUBS[lang])
                except Exception as error:
                    report.update(status='infrastructure_failure',failed_task=[lang,name],error=str(error));save_meta(meta,report);raise
                invoked=MARK in (result['stdout']+'\n'+result['stderr']).splitlines()
                valid=invoked and not result['passed']
                result.update(language=lang,name=name,stub_invoked=invoked,scaffold_valid=valid,not_model_score=True)
                stream.write(json.dumps(result)+'\n');stream.flush();done+=1
                report.update(completed_items=done);save_meta(meta,report);print(now(),done,len(jobs),lang,name,flush=True)
                if time.monotonic()-tick>=a.seconds:break
        failures=[]
        for line in out.read_text().splitlines():
            row=json.loads(line)
            if not row['scaffold_valid']:failures.append([row['language'],row['name']])
        report.update(status=('complete' if not failures else 'scaffold_issues') if done==len(jobs) else 'time_budget',completed_items=done,
                      all_scaffolds_invoked_stub=done==len(jobs) and not failures,failed_scaffolds=failures)
        report['events'][-1].update(end=now(),ending_items=done);save_meta(meta,report)


if __name__=='__main__':main()
