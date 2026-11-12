"""Resume the fixed full HF target with one model, native pressure and wall guards."""
import argparse,fcntl,hashlib,json,os,re,shlex,signal,subprocess,sys,time
from datetime import datetime,timedelta
from pathlib import Path

DRIVER_SHA='b249444166f870f6496115d76b145674da29e4a1edec4485c002cf28de4b2102'


def native(argv):return subprocess.run(argv,capture_output=True,text=True,check=True).stdout


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline',help='Aware deadline, at most2hours away; default now+2hours')
    parser.add_argument('--prefix',help='New evidence prefix inside runs/; default unique timestamp')
    args=parser.parse_args();start=datetime.now().astimezone()
    deadline=datetime.fromisoformat(args.deadline) if args.deadline else start+timedelta(hours=2)
    if deadline.tzinfo is None or deadline>start+timedelta(hours=2,seconds=1):parser.error('Aware deadline at most2hours away required')
    prefix=Path(args.prefix or 'runs/hf-guard-'+start.strftime('%Y%m%dT%H%M%S%f'))
    if prefix.resolve().parent!=Path('runs').resolve() or not re.fullmatch('[A-Za-z0-9_-]+',prefix.name):parser.error('Prefix must be a simple name inside runs/')
    paths={key:Path(str(prefix)+suffix) for key,suffix in [('state','-state.json'),('resources','-resources.jsonl'),('log','-driver.log'),('controller','-controller.json')]}
    if any(path.exists() for path in paths.values()):raise FileExistsError('Preserve existing evidence; choose a new prefix')
    state=dict(started=start.isoformat(),deadline=deadline.isoformat(),wrapper_pid=os.getpid(),status='preflight',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),wall_sigint_seconds=1180,guard_pressure=2,signals=[])
    def save():paths['state'].write_text(json.dumps(state,indent=2)+'\n')
    Path('runs').mkdir(exist_ok=True)
    if start>=deadline-timedelta(seconds=200):state['status']='refused_deadline';save();return
    lock=Path('runs/hf-mmmlu-global.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    watched={'run_hf_mmmlu.py','run_paper_text.py','evaluate_batched.py','evaluate_mrcr.py','prepare_mrcr.py','train.py','sample.py'}
    busy=[]
    for line in native(['ps','-axo','pid=,command=']).splitlines():
        pid,command=line.strip().split(None,1)
        if int(pid)==os.getpid():continue
        try:tokens=shlex.split(command)
        except ValueError:continue
        matches=watched&{Path(token).name for token in tokens}
        if matches:busy.append(dict(pid=int(pid),scripts=sorted(matches)))
    if busy:state.update(status='refused_busy',busy=busy);save();return
    level=int(native(['sysctl','-n','kern.memorystatus_vm_pressure_level']))
    free=int(re.search(r'free percentage:\s*(\d+)%',native(['memory_pressure','-Q'])).group(1))
    state.update(initial_pressure=level,initial_free_percent=free)
    if level!=1 or free<55:state['status']='refused_resources';save();return
    assert hashlib.sha256(Path('run_hf_mmmlu.py').read_bytes()).hexdigest()==DRIVER_SHA,'Review changed driver before resume'
    child=None;stopped=False;seen=set()
    def stop(signum,frame):
        nonlocal stopped
        if child is not None and child.poll() is None and not stopped:child.send_signal(signal.SIGINT)
        stopped=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    try:
        with paths['log'].open('x') as log,paths['resources'].open('x') as samples:
            child=subprocess.Popen([sys.executable,'run_hf_mmmlu.py','--deadline',deadline.isoformat(),'--controller-state',str(paths['controller'])],stdout=log,stderr=subprocess.STDOUT)
            state.update(status='running',driver_pid=child.pid);save();print(json.dumps(state),flush=True)
            while True:
                now=datetime.now().astimezone();level=int(native(['sysctl','-n','kern.memorystatus_vm_pressure_level']))
                samples.write(json.dumps(dict(time=now.isoformat(),pressure=level,vm_stat=native(['vm_stat']),swap=native(['sysctl','vm.swapusage'])))+'\n');samples.flush()
                if child.poll() is not None:break
                if not stopped and (level>=2 or now>=deadline-timedelta(seconds=150)):
                    state['driver_stop']=dict(at=now.isoformat(),pressure=level,reason='pressure' if level>=2 else 'deadline');save();stop(signal.SIGINT,None)
                if not stopped:
                    try:job=json.loads(Path('runs/hf-mmmlu-runner-state.json').read_text())
                    except (FileNotFoundError,json.JSONDecodeError):job=None
                    if job and job['status']=='child_running' and job['driver_pid']==child.pid and job['pid'] not in seen and (now-datetime.fromisoformat(job['started'])).total_seconds()>=1180:
                        command=subprocess.run(['ps','-p',str(job['pid']),'-o','ppid=,command='],capture_output=True,text=True).stdout.strip()
                        if command and command.split(None,1)[0]==str(child.pid) and 'evaluate_batched.py runs/qwen-sparse/last.pt' in command and '--out runs/eval-sparse-mmmlu-'+job['language']+'-full.jsonl' in command:
                            try:os.kill(job['pid'],signal.SIGINT)
                            except ProcessLookupError:pass
                            else:seen.add(job['pid']);state['signals'].append(dict(at=now.isoformat(),pid=job['pid'],language=job['language'],elapsed_seconds=(now-datetime.fromisoformat(job['started'])).total_seconds(),reason='1180s model child wall budget'));save()
                time.sleep(5)
            state.update(status='terminal',returncode=child.returncode)
    except BaseException as error:
        state.update(status='error',error_type=type(error).__name__);stop(signal.SIGINT,None)
        if child is not None:state['returncode']=child.wait()
        raise
    finally:state['ended']=datetime.now().astimezone().isoformat();save()
    if state['returncode']:raise SystemExit(state['returncode'])


if __name__=='__main__':main()
