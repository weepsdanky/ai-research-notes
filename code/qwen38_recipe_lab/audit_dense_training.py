"""Audit a real matched Dense checkpoint; passing does not imply benchmark quality."""
import argparse,fcntl,hashlib,json
from datetime import datetime
from pathlib import Path
import torch
from audit_stage import finite_tree


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


@torch.no_grad()
def audit_checkpoint(path='runs/qwen-dense-paper-control/last.pt'):
    torch.set_num_threads(1)
    profile_path=Path('profiles/paper_dense_control.json');profile=json.loads(profile_path.read_text());out=Path(profile['out']);metadata=json.loads((out/'config.json').read_text())
    assert Path(path).resolve()==(out/'last.pt').resolve()
    actual={name:digest(name) for name in profile['source_hashes']};assert actual==profile['source_hashes']
    assert digest(profile['init'])==profile['init_sha256'] and metadata['device']=='mps'
    saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True);initial=torch.load(profile['init'],map_location='cpu',weights_only=True,mmap=True)
    assert saved['source_hashes']==metadata['source_hashes']==actual and saved['config']==initial['config']
    keys=('stage','score_mode','steps','seq','batch','accum','lr','schedule','optimizer','ns_dtype','weight_decay','seed','clip','eval_every','eval_batches','device')
    assert all(saved['args'][key]==profile[key] for key in keys)
    assert saved['args']['data']==profile['data'][0] and saved['args']['mix_data']==profile['data'][1]
    step=saved['step'];assert 0<=step<=profile['steps']==8000 and profile['seq']==128 and profile['batch']==profile['accum']==1
    assert finite_tree(saved['model']) and finite_tree(saved['optimizers'])
    assert saved['model'].keys()==initial['model'].keys();changed=[];frozen=0
    for name,value in saved['model'].items():
        assert value.shape==initial['model'][name].shape and value.dtype==initial['model'][name].dtype,name
        equal=torch.equal(value,initial['model'][name])
        if '.indexer.' in name:assert equal,name;frozen+=value.numel()
        elif not equal:changed.append(name)
    assert step==0 or changed,'No backbone update in a nonzero-step checkpoint'
    corpora=[];sizes=[];data_hashes={}
    for name in profile['data']:
        assert digest(name+'/meta.json')==profile['data_meta_sha256'][name]
        meta=json.loads(Path(name+'/meta.json').read_text());corpora.append(meta);sizes.append(meta['files']['train']['tokens'])
        for split in ('train','val'):
            filename=name+'/'+split+'.bin';data_hashes[filename]=digest(filename);assert data_hashes[filename]==meta['files'][split]['sha256']
    assert saved['corpora']==metadata['corpora']==corpora
    rng=torch.Generator().manual_seed(profile['seed']);draws=[0,0]
    for _ in range(step):
        index=int(torch.rand((),generator=rng).item()<profile['mix_probability']);draws[index]+=1
        torch.randint(sizes[index]-profile['seq'],(profile['batch'],),generator=rng)
    assert torch.equal(saved['rng'],rng.get_state()),'Training data RNG differs from the declared ordered draws'
    records=[json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
    training=[row for row in records if 'step_seconds' in row];assert not step or training
    assert all(row.get('mps_driver_bytes',0)>0 for row in training),'Missing actual MPS allocation evidence'
    validation=[row for row in records if row.get('step')==step and 'val0/lm' in row]
    assert validation,'Saved step lacks its fixed validation record'
    return dict(at=datetime.now().astimezone().isoformat(),status='passed_checkpoint_checks',scope='Checkpoint identity/finite state, actual MPS records, frozen unused indexer and data-RNG continuity; not a downstream-quality acceptance',auditor_sha256=digest(__file__),profile_sha256=digest(profile_path),checkpoint=str(path),checkpoint_sha256=digest(path),source_hashes=actual,finite_checker_sha256=digest('audit_stage.py'),step=step,target_steps=profile['steps'],phase_complete=step==profile['steps'],tokens_saved=step*profile['seq'],all_model_and_optimizer_tensors_finite=True,changed_backbone_tensors=changed,unchanged_unused_indexer_elements=frozen,data_rng_replayed_to_saved_step=True,training_corpus_draws=draws,data_hashes=data_hashes,actual_training_device_from_config=metadata['device'],mps_driver_peak_bytes=max((row['mps_driver_bytes'] for row in training),default=0),process_peak_rss_bytes=max((row['process_peak_rss_bytes'] for row in training),default=0),maximum_logged_step=max((row['step'] for row in training),default=0),fixed_validation=validation[-1])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--deadline',required=True);p.add_argument('--out',required=True);a=p.parse_args();deadline=datetime.fromisoformat(a.deadline);target=Path(a.out);assert not target.exists() and deadline.tzinfo is not None and datetime.now().astimezone()<deadline
    with Path('runs/hf-mmmlu-global.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);report=audit_checkpoint();assert datetime.now().astimezone()<deadline
    target.write_text(json.dumps(report,indent=2)+'\n');print(report['step'],report['target_steps'],report['phase_complete'])


if __name__=='__main__':main()
