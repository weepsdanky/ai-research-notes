"""Build only the bounded native arm64 eight-language runtime in task VM."""
import argparse
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path


BASE = 'sha256:65bd3d9f1d7fa268dd1909973ba7eb4bdf41f6dcf2cfbe76ba9528b170f3fd79'
PYTHON_DEPS = 'sha256:813de4a18b376e4d4bda2727878f178737d64aaf967e7cf1fbfd200d60f3ec72'


def docker(*args, **kwargs):
    return subprocess.run(['zsh','vm.sh','docker',*args],check=True,**kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--seconds',type=int,default=1200)
    args = parser.parse_args()
    if not 1<=args.seconds<=1200:parser.error('seconds must be 1..1200')
    context = Path('.cache/multipl-build');context.mkdir(parents=True, exist_ok=True)
    cache_policy=Path('apt-keep-cache.conf');(context/cache_policy.name).write_bytes(cache_policy.read_bytes())
    jar = context/'javatuples-1.2.jar'
    lock_path = Path('multipl_runtime_sources.lock.json')
    url = 'https://repo.maven.apache.org/maven2/org/javatuples/javatuples/1.2/javatuples-1.2.jar'
    if not jar.exists():
        with urllib.request.urlopen(url, timeout=60) as response:data = response.read(1_000_001)
        if len(data)>1_000_000:raise ValueError('Jar cap')
        jar.write_bytes(data)
    jar_spec = dict(url=url, bytes=jar.stat().st_size, sha256=hashlib.sha256(jar.read_bytes()).hexdigest())
    if lock_path.exists() and json.loads(lock_path.read_text()) != jar_spec:raise ValueError('Runtime source drift')
    lock_path.write_text(json.dumps(jar_spec, indent=2)+'\n')
    recipe = Path('Dockerfile.multipl').read_text()
    source_line = 'FROM '+PYTHON_DEPS+' AS python_deps'
    if recipe.splitlines()[0]!=source_line or json.loads(Path('code_images.lock.json').read_text())['evalplus']['image_id']!=PYTHON_DEPS:
        raise ValueError('Pinned Python package layer drift')
    command = next(line[4:] for line in recipe.splitlines() if line.startswith('RUN '))
    state_path = Path('runs/multipl-build-state.json')
    if args.resume:
        state = json.loads(state_path.read_text())
        if state['recipe_sha256']!=hashlib.sha256(recipe.encode()).hexdigest():raise ValueError('Build resume recipe drift')
    else:
        if state_path.exists():raise FileExistsError('Use --resume for the saved dependency build')
        created = docker('create','--network=bridge','--memory=1024m','--memory-swap=1024m','--cpus=1','--pids-limit=128',BASE,'/bin/sh','-c',command,capture_output=True,text=True)
        state = dict(container=created.stdout.strip(),recipe_sha256=hashlib.sha256(recipe.encode()).hexdigest())
    inspected = json.loads(docker('inspect',state['container'],capture_output=True,text=True).stdout)[0]
    if inspected['Image']!=BASE or inspected['Config']['Cmd']!=['/bin/sh','-c',command] or inspected['Mounts']:raise ValueError('Unexpected dependency-build container')
    if inspected['State']['Running']:raise ValueError('Another build is already running')
    if state.get('status')=='complete':
        locked=json.loads(Path('code_images.lock.json').read_text())['multipl']['image_id']
        if state['image_id']!=locked:raise ValueError('Completed image identity drift')
        docker('image','inspect',locked,capture_output=True,text=True)
        print('Already complete; preserving image identity '+locked);return
    # Match the Dockerfile's pre-RUN COPY even when continuing a legacy container.
    docker('cp',str(cache_policy),state['container']+':/etc/apt/apt.conf.d/docker-clean')
    ready = inspected['State']['Status']=='exited' and inspected['State']['ExitCode']==0
    if not ready:
        state['status'] = 'running';state_path.write_text(json.dumps(state,indent=2)+'\n')
        process = subprocess.Popen(['zsh','vm.sh','docker','start','-a',state['container']])
        try:process.wait(timeout=args.seconds)
        except subprocess.TimeoutExpired:
            docker('kill',state['container']);process.wait(timeout=15)
            state['status']='time_budget';state_path.write_text(json.dumps(state,indent=2)+'\n')
            print('Dependency downloads preserved. Stop VM; later start it and rerun with --resume.');return
        inspected = json.loads(docker('inspect',state['container'],capture_output=True,text=True).stdout)[0]
        if inspected['State']['ExitCode']!=0:
            state.update(status='build_failure',exit_code=inspected['State']['ExitCode'],oom_killed=inspected['State']['OOMKilled'])
            state_path.write_text(json.dumps(state,indent=2)+'\n');raise RuntimeError('Dependency build failed; retained container and downloads')
    committed = docker('commit',state['container'],capture_output=True,text=True).stdout.strip()
    # The dependency layer is exactly the locked RUN above; no host files/mounts.
    final_recipe = source_line+'\nFROM '+committed+'\n'+'\n'.join(line for line in recipe.splitlines() if line.startswith(('COPY ','ENV ')))+'\n'
    (context/'Dockerfile').write_text(final_recipe)
    docker('build','--platform=linux/arm64','--network=none','-t','qwen38-multipl:locked',str(context),timeout=120)
    r = subprocess.run(['zsh','vm.sh','docker','image','inspect','qwen38-multipl:locked'],check=True,capture_output=True,text=True)
    image = json.loads(r.stdout)[0]
    if image['Architecture'] != 'arm64' or image['Size'] > 2_000_000_000:raise ValueError('Runtime image budget/architecture')
    path = Path('code_images.lock.json');images = json.loads(path.read_text())
    images['multipl'] = dict(image_id=image['Id'],architecture='arm64',os='linux',size_bytes=image['Size'],
        source_tag='local qwen38-multipl:locked',dependency_image_id=committed,recipe_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['Dockerfile.multipl','apt-keep-cache.conf','multipl_runtime_sources.lock.json']})
    path.write_text(json.dumps(images,indent=2)+'\n')
    state.update(status='complete',image_id=image['Id'],exit_code=0,oom_killed=False);state_path.write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(images['multipl'],indent=2))


if __name__ == '__main__':main()
