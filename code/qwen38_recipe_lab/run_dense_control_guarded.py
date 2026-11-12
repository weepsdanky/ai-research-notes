"""Run the declared matched Dense training control under the local resource guard."""
import argparse,fcntl,hashlib,json,os,re,signal,subprocess,sys,time
from datetime import datetime,timedelta
from pathlib import Path


def native(command):return subprocess.check_output(command,text=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--deadline',required=True);parser.add_argument('--prefix',required=True);parser.add_argument('--seconds',type=int,default=1140);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    wall_started=time.monotonic();started=datetime.now().astimezone();deadline=datetime.fromisoformat(args.deadline);assert deadline.tzinfo is not None and 30<=args.seconds<=1140
    prefix=Path(args.prefix);assert prefix.parent.resolve()==Path('runs').resolve()
    state_path=Path(str(prefix)+'-state.json');assert not any(Path(str(prefix)+tail).exists() for tail in ('-state.json','.log','-resources.jsonl')),'Preserve existing evidence; choose new prefix'
    state=dict(started=started.isoformat(),deadline=deadline.isoformat(),wrapper_pid=os.getpid(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),suite='matched_dense_training',stage='pretrain',status='preflight',signals=[])
    def save():
        temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(state_path)
    save()
    if not started+timedelta(seconds=660)<deadline<=started+timedelta(hours=2):state['status']='refused_deadline';save();return
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
    profile_path=Path('profiles/paper_dense_control.json');profile=json.loads(profile_path.read_text())
    def digest(path):
        with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    readiness_path=Path('runs/paper-dense-control-readiness-round25.json');readiness=json.loads(readiness_path.read_text())
    assert readiness['status']=='passed' and readiness['all8000_training_data_draws_replayed'] and not readiness['training_started']
    assert readiness['profile_sha256']==digest(profile_path) and readiness['runner_sha256']==digest('audit_dense_control.py')
    assert all(digest(path)==sha for path,sha in {**profile['source_hashes'],**readiness['sources']}.items())
    assert all(digest(path+'/meta.json')==sha for path,sha in profile['data_meta_sha256'].items())
    assert profile['steps']==8000 and profile['phase_tokens']==1024000 and profile['device']=='mps'
    assert os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO') is None and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') is None
    for name in ('init','sparse_init','comparison'):assert digest(profile[name])==profile[name+'_sha256']
    command=[sys.executable,*profile['command'][3:]];assert command[1]=='train.py'
    checkpoint=Path(profile['out'])/'last.pt'
    if args.resume:
        assert checkpoint.exists(),'No original control checkpoint to resume'
        index=command.index('--init');del command[index:index+2];command+=['--resume',str(checkpoint)]
    elif Path(profile['out']).exists():
        state.update(status='refused_existing_run',checkpoint=str(checkpoint));save();return
    seconds=min(args.seconds,1140-int(time.monotonic()-wall_started),int((deadline-datetime.now().astimezone()).total_seconds())-600)
    if seconds<30:state['status']='refused_deadline';save();return
    command[command.index('--seconds')+1]=str(seconds)
    state.update(profile_sha256=digest(profile_path),readiness_sha256=digest(readiness_path),checkpoint=str(checkpoint),child_seconds=seconds,cleanup_reserve_seconds=600,wall_budget_includes_preflight=True,mps_high_watermark_ratio='default',mps_fallback='default');save()
    requested=[]
    def request(signum,frame):requested.append(signal.Signals(signum).name)
    for signum in (signal.SIGINT,signal.SIGTERM):signal.signal(signum,request)
    with Path(str(prefix)+'.log').open('x') as log,Path(str(prefix)+'-resources.jsonl').open('x') as samples:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);tick=time.monotonic();stop_at=None;escalated=False;state.update(status='running',child_pid=child.pid,command=command);save()
        while child.poll() is None:
            now=datetime.now().astimezone();level=int(native(['sysctl','-n','kern.memorystatus_vm_pressure_level']))
            probe=subprocess.run(['ps','-p',str(child.pid),'-o','pid=,ppid=,rss=,command='],capture_output=True,text=True).stdout
            samples.write(json.dumps(dict(at=now.isoformat(),pressure=level,child=probe,vm_stat=native(['vm_stat']),swap=native(['sysctl','vm.swapusage'])))+'\n');samples.flush()
            reason='parent_signal' if requested else 'memory_pressure' if level>=2 else 'wall_budget' if time.monotonic()-wall_started>=1180 else 'deadline' if now>=deadline-timedelta(seconds=600) else None
            if stop_at is None and reason:
                child.send_signal(signal.SIGINT);stop_at=time.monotonic();state['signals'].append(dict(at=now.isoformat(),signal='SIGINT',reason=reason,pressure=level));save()
            elif stop_at is not None and not escalated and (time.monotonic()-stop_at>=60 or time.monotonic()-wall_started>=1198):
                child.kill();escalated=True;state['signals'].append(dict(at=now.isoformat(),signal='SIGKILL',reason='Own child exceeded checkpoint grace or1198-second hard wall margin'));save()
            time.sleep(2)
        state.update(status='terminal',returncode=child.returncode,ended=datetime.now().astimezone().isoformat());save()
    print(json.dumps(state),flush=True)


if __name__=='__main__':main()
