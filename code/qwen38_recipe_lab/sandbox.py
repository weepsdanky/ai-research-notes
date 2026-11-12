"""Bounded local-VM execution. Exit zero is NOT a benchmark correctness score.

Only task files are mounted, read-only, into an unprivileged offline container.
An official benchmark harness must establish correctness separately; candidate
stdout or an early sys.exit(0) must never be treated as a passing test suite.
"""
import hashlib,json,os,selectors,subprocess,tempfile,time,uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def docker(*args,timeout=15):
    return subprocess.run(['zsh',str(ROOT/'vm.sh'),'docker',*args],capture_output=True,text=True,timeout=timeout)


class Sandbox:
    def __init__(self,image='python',large=False):
        info=docker('info','--format','{{json .}}')
        if info.returncode:raise RuntimeError('Start the task VM with zsh vm.sh start')
        info=json.loads(info.stdout)
        maximum=(3.2 if large else 2.2)*1024**3
        if info['MemTotal']>maximum or large and info['MemTotal']<2.8*1024**3 or not all(info[k] for k in ('MemoryLimit','SwapLimit','PidsLimit')):
            raise RuntimeError('Unexpected VM size or missing cgroup limits')
        self.memory_limit_mb=2048 if large else 1024
        endpoint=docker('context','inspect','colima-qwen38','--format','{{.Endpoints.docker.Host}}').stdout.strip()
        expected=Path(os.environ.get('QWEN_VM_ROOT',str(Path.home()/'.local/share/q38eval')))/'c/qwen38/docker.sock'
        if endpoint!='unix://'+str(expected):raise RuntimeError('Refuse an unexpected or remote Docker endpoint')
        self.image=json.loads((ROOT/'code_images.lock.json').read_text())[image]['image_id']
        inspect=docker('image','inspect',self.image)
        if inspect.returncode:raise RuntimeError('Pinned local image is unavailable')
        details=json.loads(inspect.stdout)[0]
        if details['Id']!=self.image or details['Architecture']!='arm64':raise RuntimeError('Image identity mismatch')

    def run(self,code,*,seconds=10,memory_mb=256,cpu_seconds=5,output_bytes=65536,file_bytes=1_048_576,tmp_exec=False):
        if not (0<seconds<=120 and 32<=memory_mb<=self.memory_limit_mb and 0<cpu_seconds<=120 and 0<output_bytes<=1_048_576 and 1_048_576<=file_bytes<=16_777_216):
            raise ValueError('Execution budgets out of range')
        base=ROOT/'runs/code-jobs';base.mkdir(parents=True,exist_ok=True)
        directory=Path(tempfile.mkdtemp(prefix='job-',dir=base));directory.chmod(0o755)
        script=directory/'task.py';script.write_text(code);script.chmod(0o644)
        name='qwen38-code-'+uuid.uuid4().hex
        flags=['run','--name',name,'--pull=never','--network=none','--read-only','--user=65534:65534',
            '--cap-drop=ALL','--security-opt=no-new-privileges','--pids-limit=32',
            f'--memory={memory_mb}m',f'--memory-swap={memory_mb}m','--cpus=1',
            f'--ulimit=cpu={cpu_seconds}:{cpu_seconds}',f'--ulimit=fsize={file_bytes}:{file_bytes}',
            '--ulimit=nofile=64:64','--ulimit=core=0:0','--log-driver=none',
            '--tmpfs=/tmp:rw,nosuid,'+('exec' if tmp_exec else 'noexec')+',size=67108864',
            '--mount',f'type=bind,src={directory},dst=/work,readonly','--workdir=/work',
            self.image,'python','-I','/work/task.py']
        started=time.monotonic();buffer=bytearray();reason=None;state={};cleanup_ok=False
        process=subprocess.Popen(['zsh',str(ROOT/'vm.sh'),'docker',*flags],stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        selector=selectors.DefaultSelector();os.set_blocking(process.stdout.fileno(),False)
        selector.register(process.stdout,selectors.EVENT_READ);eof=False
        try:
            while True:
                for key,_ in selector.select(.05):
                    chunk=os.read(key.fd,65536)
                    if not chunk:eof=True;break
                    room=output_bytes-len(buffer);buffer.extend(chunk[:room])
                    if len(chunk)>room:reason='output_limit';break
                if reason:break
                if process.poll() is not None and eof:break
                if time.monotonic()-started>=seconds:reason='wall_timeout';break
            if reason:docker('kill',name,timeout=5)
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait();reason=reason or 'client_timeout'
            observed=docker('inspect',name,'--format','{{json .State}}',timeout=5)
            if observed.returncode==0:state=json.loads(observed.stdout)
        finally:
            selector.close();process.stdout.close()
            try:cleanup_ok=docker('rm','-f',name,timeout=5).returncode==0
            except subprocess.TimeoutExpired:cleanup_ok=False
        status=reason or ('memory_limit' if state.get('OOMKilled') else ('exited_zero' if process.returncode==0 else 'nonzero_exit'))
        result=dict(status=status,exit_code=state.get('ExitCode',process.returncode),oom_killed=state.get('OOMKilled'),
            output=buffer.decode('utf-8',errors='replace'),seconds=time.monotonic()-started,cleanup_ok=cleanup_ok,
            image_id=self.image,code_sha256=hashlib.sha256(code.encode()).hexdigest(),
            limits=dict(wall_seconds=seconds,memory_mb=memory_mb,cpu_seconds=cpu_seconds,pids=32,cpus=1,output_bytes=output_bytes,file_bytes=file_bytes),
            job_directory=str(directory),benchmark_correctness='not inferred from process status')
        if tmp_exec:result['limits'].update(tmpfs_exec=True,tmpfs_bytes=67108864)
        (directory/'execution.json').write_text(json.dumps(result,indent=2)+'\n')
        if not cleanup_ok:raise RuntimeError(f'Container cleanup failed; stop this task VM before more jobs: {name}')
        return result
