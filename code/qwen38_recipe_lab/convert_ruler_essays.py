"""Run the locked RULER author's essay conversion on all218 verified local inputs."""
import argparse,hashlib,io,json,os,resource,runpy,subprocess,tempfile
from datetime import datetime,timedelta
from pathlib import Path
from unittest.mock import patch


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--deadline',required=True);args=parser.parse_args()
    deadline=datetime.fromisoformat(args.deadline);assert deadline.tzinfo is not None
    assert datetime.now().astimezone()<deadline-timedelta(seconds=180)
    assert int(subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True))==1
    root=Path.cwd();target=root/'data/ruler-source/PaulGrahamEssays.json';assert not target.exists(),'Preserve existing conversion'
    plan=json.loads(Path('runs/ruler-essay-intake-plan.json').read_text());state=json.loads(Path('runs/ruler-essay-intake.json').read_text())
    assert state['status']=='all_raw_inputs_fetched' and len(state['files'])==len(plan['inputs'])==218 and not state['errors']
    locks=json.loads(Path('ruler_sources.lock.json').read_text())
    for item in locks:assert digest(item['file'])==item['sha256']
    from bs4 import BeautifulSoup
    import importlib.metadata
    blobs={};inputs=[]
    for item in plan['inputs']:
        stored=state['files'][item['file']];data=Path(item['file']).read_bytes();assert digest(item['file'])==stored['sha256'] and len(data)==stored['bytes']
        if item['kind']=='html':assert BeautifulSoup(data.decode('unicode_escape','utf-8'),'html.parser').find('font') is not None,item['file']
        else:data.decode('utf-8')
        blobs[item['original_url']]=data;inputs.append(dict(file=item['file'],sha256=stored['sha256'],bytes=len(data)))
    source=root/'.cache/long-context-sources/ruler-scripts-data-synthetic-json-download_paulgraham_essay.py'
    with tempfile.TemporaryDirectory(dir=root/'.cache',prefix='ruler-conversion-') as temp:
        os.chdir(temp)
        try:
            Path('PaulGrahamEssays_URLs.txt').write_bytes((root/'.cache/ruler-sources/PaulGrahamEssays_URLs.txt').read_bytes())
            with patch('urllib.request.urlopen',side_effect=lambda url:io.BytesIO(blobs[url])):runpy.run_path(str(source),run_name='__main__')
            derived=Path('PaulGrahamEssays.json').read_bytes();assert json.loads(derived)['text']
            core=json.loads((root/'ruler_data.lock.json').read_text());profile=json.loads((root/'profiles/ruler_mac.json').read_text())
            assert sum(x['bytes'] for x in core['files'])+sum(x['bytes'] for x in inputs)+len(derived)<=profile['raw_data_cap_bytes']
            assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<=profile['preparation_peak_rss_cap_bytes']
            target.write_bytes(derived)
        finally:os.chdir(root)
    record=dict(at=datetime.now().astimezone().isoformat(),status='all218_author_conversion_complete',author_source_sha256=digest(source),wrapper_sha256=digest(__file__),inputs=inputs,file=str(target.relative_to(root)),bytes=len(derived),sha256=digest(target),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,packages={name:importlib.metadata.version(name) for name in ('html2text','beautifulsoup4')},scope='Exact pinned author script; urlopen serves verified local bytes; all218 inputs and first font present; original sorted concatenation retained; no RULER model score')
    Path('runs/ruler-essay-conversion.json').write_text(json.dumps(record,indent=2)+'\n');print(record['status'],len(derived),flush=True)


if __name__=='__main__':main()
