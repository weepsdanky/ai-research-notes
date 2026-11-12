"""Verify byte-preserved imported generations and their distinct source versions."""
import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from import_retention_prefix import digest
from evaluate_suite import now


LEGACY_ARCHIVE='a076eaab1e9db14eb48e750ea287a91fcfb00d53'


def archived_source(name,sha,commit):
    if Path(name).exists() and digest(name)==sha:return
    raw=subprocess.check_output(['git','show',commit+':code/qwen38_recipe_lab/'+name])
    assert hashlib.sha256(raw).hexdigest()==sha


def verify(predictions):
    path=Path(predictions);meta=json.loads(Path(predictions+'.meta.json').read_text());imp=meta['prefix_import']
    if meta['status'] not in ('complete','time_budget','interrupted_memory_pressure'):raise ValueError('Terminal output required')
    oldpath=Path(imp['source']);oldmeta_path=Path(str(oldpath)+'.meta.json');oldmeta=json.loads(oldmeta_path.read_text())
    assert digest(oldpath)==imp['source_sha256'] and digest(oldmeta_path)==imp['source_meta_sha256']
    assert digest(imp['source_audit'])==imp['source_audit_sha256']
    archived_source(imp.get('helper_path','import_retention_prefix.py'),imp['helper_sha256'],imp.get('helper_archive_commit',LEGACY_ARCHIVE))
    assert digest(imp.get('proof_path','runs/retention-trained-proof.json'))==imp['proof_sha256']
    upstream=verify(str(oldpath)) if 'prefix_import' in oldmeta else None
    expected=copy.deepcopy(oldmeta['identity'])
    key=next(k for k in expected['source_hashes'] if Path(k).name=='evaluate_batched.py')
    assert expected['source_hashes'][key]==imp['old_engine_sha256']
    expected['source_hashes'][key]=imp['new_engine_sha256']
    assert meta['identity']==expected
    archived_source('evaluate_batched.py',imp['new_engine_sha256'],imp.get('new_engine_archive_commit',LEGACY_ARCHIVE))
    n=imp['items'];segments=meta['generation_source_segments']
    expected_segments=copy.deepcopy(oldmeta.get('generation_source_segments',
        [dict(start=0,end_exclusive=n,engine_sha256=imp['old_engine_sha256'],source=imp['source'])]))
    for segment in expected_segments:
        if segment.pop('end_exclusive_ref',None):segment['end_exclusive']=n
    assert segments==expected_segments+[dict(start=n,end_exclusive_ref='completed_items',engine_sha256=imp['new_engine_sha256'],source=predictions)]
    prefix=hashlib.sha256();whole=hashlib.sha256();count=0
    with path.open('rb') as stream:
        for line in stream:
            json.loads(line);whole.update(line)
            if count<n:prefix.update(line)
            count+=1
    assert count==meta['completed_items']>=n and prefix.hexdigest()==imp['source_sha256']
    result=dict(at=now(),status=meta['status'],items=count,imported_items=n,new_items=count-n,
        target_items=sum(t['target'] for t in meta['tasks'].values()),file_sha256=whole.hexdigest(),
        original_file_and_metadata_unchanged=True,imported_prefix_byte_identical=True,
        only_engine_retention_source_changed=True,source_segments=segments,
        upstream_import_verified=upstream is not None,
        note='Provenance audit; source prompt/IDs/decoding/scores require audit_core_results.py separately. Do not sum old and imported files.')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('predictions');p.add_argument('--out',required=True);a=p.parse_args()
    result=verify(a.predictions)
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__=='__main__':main()
