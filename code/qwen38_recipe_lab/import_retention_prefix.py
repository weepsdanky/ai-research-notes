"""Import an audited prefix into a new file after the exact retention-only fix.

Original generations and metadata remain untouched. Mixed source provenance is
explicit; every other inference, data and checkpoint identity must be unchanged.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from evaluate_suite import now

BASELINE='e0b850ee9f81f8e50c2ac692e05f1be4182b4394'


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source');p.add_argument('target');p.add_argument('--audit',required=True)
    p.add_argument('--baseline',default=BASELINE)
    p.add_argument('--proof',default='runs/retention-trained-proof.json')
    a=p.parse_args();source=Path(a.source);target=Path(a.target)
    oldmeta=Path(a.source+'.meta.json');newmeta=Path(a.target+'.meta.json')
    if target.exists() or newmeta.exists() or source.resolve()==target.resolve():raise FileExistsError('New target required')
    with Path(a.source+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        meta=json.loads(oldmeta.read_text());identity=meta['identity']
        if meta['status'] not in ('time_budget','interrupted_memory_pressure') or meta['full_dataset_generated']:
            raise ValueError('Terminal incomplete source required')
        audit=json.loads(Path(a.audit).read_text())['files'][a.source]
        source_sha=digest(source)
        if not audit['all_source_prompts_gold_input_ids_raw_decoding_and_scores_verified'] or audit['file_sha256']!=source_sha or audit['items']!=meta['completed_items']:
            raise ValueError('Current audited prefix required')
        old=subprocess.check_output(['git','show',a.baseline+':code/qwen38_recipe_lab/evaluate_batched.py'])
        needle=b'existing.append(r);count(r);completed += 1'
        replacement=b"existing.append(dict(output_ids=r['output_ids'], prediction=r['prediction']));count(r);completed += 1"
        current=Path('evaluate_batched.py').read_bytes()
        expected=old.replace(needle,replacement)
        encode_old=b'        jobs = [(t, i, prompt, gold, tokenizer.encode(prompt, add_special_tokens=False).ids)\n'
        encode_new=(b'        token_pool = {}\n        def encode(prompt):\n'
            b'            return [token_pool.setdefault(token, token) for token in tokenizer.encode(prompt, add_special_tokens=False).ids]\n'
            b'        jobs = [(t, i, prompt, gold, encode(prompt))\n')
        if current!=expected:expected=expected.replace(encode_old,encode_new)
        if current==old or current!=expected:raise ValueError('Only the exact retention/token-interning changes are allowed')
        old_sha=hashlib.sha256(old).hexdigest();new_sha=hashlib.sha256(current).hexdigest()
        engine_keys=[name for name in identity['source_hashes'] if Path(name).name=='evaluate_batched.py']
        if len(engine_keys)!=1:raise ValueError('One original evaluator required')
        engine_key=engine_keys[0]
        if identity['source_hashes'][engine_key]!=old_sha:raise ValueError('Unexpected original evaluator')
        for section in ('source_hashes','runtime_model_sources','chunk_sources'):
            for name,sha in identity[section].items():
                if name!=engine_key and digest(name)!=sha:raise ValueError('Other source changed: '+name)
        if digest('prefix_inference.py')!=identity['prefix_cache']['source_sha256'] or digest('.cache/references/tokenizer.json')!=identity['tokenizer_sha256']:
            raise ValueError('Prefix/tokenizer drift')
        if digest('runs/qwen-sparse/last.pt')!=identity['checkpoint_sha256']:raise ValueError('Checkpoint drift')
        proof=json.loads(Path(a.proof).read_text())
        if proof['engine_sha256']!=new_sha or proof['source_file_sha256']!=source_sha or not (proof['all_original_ids_exact'] and proof['partial_resume_exact']):
            raise ValueError('Real checkpoint and partial-batch proof required')
        archive=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        for name in ('import_retention_prefix.py','evaluate_batched.py'):
            if subprocess.check_output(['git','show',archive+':code/qwen38_recipe_lab/'+name])!=Path(name).read_bytes():
                raise ValueError('Commit the helper and evaluator before importing')
        n=meta['completed_items']
        meta['prefix_import']=dict(at=now(),source=a.source,source_sha256=source_sha,source_meta_sha256=digest(oldmeta),
            source_audit=a.audit,source_audit_sha256=digest(a.audit),baseline_commit=a.baseline,
            helper_sha256=digest(__file__),helper_path='import_retention_prefix.py',helper_archive_commit=archive,
            proof_path=a.proof,proof_sha256=digest(a.proof),new_engine_archive_commit=archive,
            old_engine_sha256=old_sha,new_engine_sha256=new_sha,items=n,
            prior_resources=meta.pop('resources',None),
            change='Only retain required replay fields and/or intern equal integer token IDs; values and all inference settings unchanged',
            source_interruption=meta.pop('interruption',None))
        segments=copy.deepcopy(meta.get('generation_source_segments',[dict(start=0,end_exclusive=n,engine_sha256=old_sha,source=a.source)]))
        for segment in segments:
            if segment.pop('end_exclusive_ref',None):segment['end_exclusive']=n
        meta['generation_source_segments']=segments+[dict(start=n,end_exclusive_ref='completed_items',engine_sha256=new_sha,source=a.target)]
        identity['source_hashes'][engine_key]=new_sha
        meta.update(status='imported_prefix',full_dataset_generated=False,full_dataset_attempted=False,resources={})
        with target.open('xb') as out,source.open('rb') as inp:shutil.copyfileobj(inp,out)
        if digest(target)!=source_sha:raise ValueError('Imported prefix differs')
        with newmeta.open('x') as out:json.dump(meta,out,ensure_ascii=False,indent=2);out.write('\n')
        print(json.dumps(meta['prefix_import'],indent=2))


if __name__=='__main__':main()
