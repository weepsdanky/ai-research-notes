"""Fetch only the two byte-locked 8-needle shards inside the declared Mac cap."""
import argparse
from datetime import datetime,timedelta
import fcntl,hashlib,json,signal,subprocess,time
from pathlib import Path
from urllib.request import Request,urlopen


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline',required=True)
    parser.add_argument('--out',default='runs/mrcr-source-fetch.json',help='New metadata path; existing evidence is preserved')
    args=parser.parse_args();record_path=Path(args.out)
    if record_path.exists():raise FileExistsError('Choose a new metadata path to preserve fetch evidence')
    deadline=datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:parser.error('Aware deadline required')
    spec=json.loads(Path('profiles/mrcr_mac.json').read_text())
    assert sum(row['bytes'] for row in spec['shards'])<=spec['raw_download_cap_bytes']==500000000
    state=dict(started=datetime.now().astimezone().isoformat(),source_sha256=digest(__file__),profile_sha256=digest('profiles/mrcr_mac.json'),status='running',files=[])
    output=Path('data/mrcr-source');output.mkdir(parents=True,exist_ok=True)
    lock=(output/'.fetch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stopping=False;tick=time.monotonic()
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    def guard():
        level=int(subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True))
        if stopping or level>=2 or time.monotonic()-tick>=1200 or datetime.now().astimezone()>=deadline-timedelta(seconds=150):
            raise InterruptedError('Pressure/signal/deadline budget: preserve partial download')
    try:
        for row in spec['shards']:
            guard();path=output/Path(row['file']).name
            assert len(row['sha256'])==64
            if path.exists():
                assert path.stat().st_size==row['bytes'] and digest(path)==row['sha256'],'Existing source differs; preserve it'
                state['files'].append(dict(file=str(path),bytes=row['bytes'],sha256=row['sha256']));continue
            part=Path(str(path)+'.part');offset=part.stat().st_size if part.exists() else 0
            assert offset<=row['bytes'];headers={'User-Agent':'qwen38-local-research'}
            if offset==row['bytes']:
                assert digest(part)==row['sha256'],'Complete partial file differs; preserve it'
                part.rename(path);state['files'].append(dict(file=str(path),bytes=offset,sha256=row['sha256']));continue
            if offset:headers['Range']=f'bytes={offset}-'
            url=f"https://huggingface.co/datasets/{spec['dataset']}/resolve/{spec['revision']}/{row['file']}"
            with urlopen(Request(url,headers=headers),timeout=60) as response,part.open('ab') as stream:
                if offset:
                    assert response.status==206 and response.headers['Content-Range']==f"bytes {offset}-{row['bytes']-1}/{row['bytes']}"
                else:assert response.status==200
                total=offset;last_check=0
                while chunk:=response.read(1<<20):
                    if time.monotonic()-last_check>=1:guard();last_check=time.monotonic()
                    total+=len(chunk);assert total<=row['bytes'],'Source exceeds locked size'
                    stream.write(chunk)
            assert total==row['bytes'] and digest(part)==row['sha256'],'Downloaded source hash/size mismatch'
            part.rename(path);state['files'].append(dict(file=str(path),bytes=total,sha256=row['sha256']))
            print('verified',path,total,flush=True)
        state['status']='complete_byte_verified_sources_only'
    except BaseException as error:
        state.update(status='stopped_or_error',error_type=type(error).__name__)
        raise
    finally:
        state['ended']=datetime.now().astimezone().isoformat()
        record_path.write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':main()
