"""One local long-context model with native pressure, wall and deadline protection."""
import argparse,fcntl,hashlib,json,os,re,signal,subprocess,sys,time
from datetime import datetime,timedelta
from pathlib import Path


def native(command):return subprocess.check_output(command,text=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--suite',choices=['mrcr','ruler','ruler500'],required=True);parser.add_argument('--stage',choices=['sparse','dense'],required=True);parser.add_argument('--deadline',required=True);parser.add_argument('--prefix',required=True);parser.add_argument('--seconds',type=int,default=1180);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    started=datetime.now().astimezone();deadline=datetime.fromisoformat(args.deadline);assert deadline.tzinfo is not None and 1<=args.seconds<=1200
    prefix=Path(args.prefix);assert prefix.parent.resolve()==Path('runs').resolve()
    state_path=Path(str(prefix)+'-state.json');assert not any(Path(str(prefix)+tail).exists() for tail in ('-state.json','.log','-resources.jsonl')),'Preserve existing evidence; choose new prefix'
    state=dict(started=started.isoformat(),deadline=deadline.isoformat(),wrapper_pid=os.getpid(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),suite=args.suite,stage=args.stage,status='preflight',signals=[])
    def save():
        temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(state_path)
    save()
    if not started+timedelta(seconds=240)<deadline<=started+timedelta(hours=2):state['status']='refused_deadline';save();return
    lock=Path('runs/hf-mmmlu-global.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:state['status']='refused_busy_global_lock';save();return
    level=int(native(['sysctl','-n','kern.memorystatus_vm_pressure_level']));free=int(re.search(r'free percentage:\s*(\d+)%',native(['memory_pressure','-Q'])).group(1));state.update(initial_pressure=level,initial_free_percent=free)
    if level!=1 or free<55:state['status']='refused_resources';save();return
    # Also reject manually launched lab compute that does not own the shared lock.
    busy=[]
    for line in native(['ps','-axo','pid=,command=']).splitlines():
        match=re.match(r'\s*(\d+)\s+(\S+)\s+(.*)',line)
        if not match or int(match[1])==os.getpid() or not Path(match[2]).name.startswith('python'):continue
        if re.search(r'(?:^|\s)(?:\S*/)?(?:evaluate\w*|train|prepare_mrcr|prepare_ruler|ruler_public500|validate_devices|audit_mrcr_results|audit_ruler_results|audit_ruler_public500)\.py(?:\s|$)',match[3]):busy.append(int(match[1]))
    if busy:state.update(status='refused_other_compute',busy_pids=busy);save();return
    script='evaluate_ruler_public500.py' if args.suite=='ruler500' else 'evaluate_mrcr_dense.py' if args.suite=='mrcr' and args.stage=='dense' else f'evaluate_{args.suite}.py'
    command=[sys.executable,script,'--stage',args.stage,'--out',f'runs/{args.suite}-paper-{args.stage}.jsonl','--deadline',deadline.isoformat(),'--seconds',str(min(args.seconds,1180))]+(['--resume'] if args.resume else [])
    requested=[]
    def request(signum,frame):requested.append(signal.Signals(signum).name)
    for signum in (signal.SIGINT,signal.SIGTERM):signal.signal(signum,request)
    with Path(str(prefix)+'.log').open('x') as log,Path(str(prefix)+'-resources.jsonl').open('x') as samples:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);tick=time.monotonic();stop_at=None;escalated=False;state.update(status='running',child_pid=child.pid,command=command);save()
        while child.poll() is None:
            now=datetime.now().astimezone();level=int(native(['sysctl','-n','kern.memorystatus_vm_pressure_level']))
            probe=subprocess.run(['ps','-p',str(child.pid),'-o','pid=,ppid=,rss=,command='],capture_output=True,text=True).stdout
            samples.write(json.dumps(dict(at=now.isoformat(),pressure=level,child=probe,vm_stat=native(['vm_stat']),swap=native(['sysctl','vm.swapusage'])))+'\n');samples.flush()
            reason='parent_signal' if requested else 'memory_pressure' if level>=2 else 'wall_budget' if time.monotonic()-tick>=1180 else 'deadline' if now>=deadline-timedelta(seconds=180) else None
            if stop_at is None and reason:
                child.send_signal(signal.SIGINT);stop_at=time.monotonic();state['signals'].append(dict(at=now.isoformat(),signal='SIGINT',reason=reason,pressure=level));save()
            elif stop_at is not None and not escalated and (time.monotonic()-stop_at>=60 or time.monotonic()-tick>=1198):
                child.kill();escalated=True;state['signals'].append(dict(at=now.isoformat(),signal='SIGKILL',reason='Own child exceeded checkpoint grace or1198-second hard wall margin'));save()
            time.sleep(2)
        state.update(status='terminal',returncode=child.returncode,ended=datetime.now().astimezone().isoformat());save()
    print(json.dumps(state),flush=True)


if __name__=='__main__':main()
