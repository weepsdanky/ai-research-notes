"""Independently check a control checkpoint and replay every accepted data draw."""
import argparse,json,hashlib
from pathlib import Path
import torch
from qwen35_controls import Qwen35Control,ControlConfig
from optim import ownership,view


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def finite(value):
    if isinstance(value,torch.Tensor):assert torch.isfinite(value).all().item()
    elif isinstance(value,dict):
        for v in value.values():finite(v)
    elif isinstance(value,(list,tuple)):
        for v in value:finite(v)


def audit(profile_path,checkpoint):
    profile=json.loads(Path(profile_path).read_text());saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert saved['profile']==profile and saved['profile_sha256']==digest(profile_path)
    assert saved['config']==profile['config'] and saved['arm']==profile['arm']
    assert all(digest(path)==sha for path,sha in saved['source_hashes'].items())
    assert all(saved['source_hashes'][path]==sha for path,sha in profile['source_hashes'].items())
    assert 0<=saved['step']<=profile['steps'] and saved['tokens']==saved['step']*profile['seq']*profile['batch']
    finite(saved['model']);finite(saved['optimizer'])
    model=Qwen35Control(ControlConfig(**saved['config']),saved['arm'],seed=profile['seed'])
    model.load_state_dict(saved['model'],strict=True)
    plans=ownership(model);groups=saved['optimizer']['param_groups'];assert len(plans)==len(groups)
    parameter_ids=[group['params'][0] for group in groups]
    assert len(parameter_ids)==len(set(parameter_ids))
    assert set(saved['optimizer']['state'])<=set(parameter_ids)
    for (name,p,specs),group in zip(plans,groups,strict=True):
        assert name==group['name'] and specs==group['pieces'] and len(group['params'])==1
        assert group['lr']==profile['lr'] and group['weight_decay']==profile['weight_decay']
        assert tuple(group['adam_betas'])==(.9,.95) and group['adam_eps']==1e-8
        assert group['ns_dtype']==profile['ns_dtype']
        buffers=saved['optimizer']['state'].get(group['params'][0],{})
        if not buffers:continue  # Unselected experts may legitimately have no state.
        assert len(buffers['pieces'])==len(specs)
        for spec,state in zip(specs,buffers['pieces'],strict=True):
            tensor=view(p,spec)
            if spec['kind']=='muon':assert state['momentum'].shape==tensor.shape and state['momentum'].dtype==tensor.dtype
            else:
                assert state['m'].shape==state['v'].shape==tensor.shape
                assert state['m'].dtype==state['v'].dtype==tensor.dtype
                assert 0<state['step']<=saved['step']
    rng=torch.Generator().manual_seed(profile['seed']);corpora=[]
    for row in profile['data']:
        directory=Path(row['directory']);assert digest(directory/'meta.json')==row['meta_sha256']
        meta=json.loads((directory/'meta.json').read_text());corpora.append(meta)
        for split in ['train','val']:assert digest(directory/f'{split}.bin')==meta['files'][split]['sha256']
    assert corpora==saved['corpora']
    draws=[0,0]
    for _ in range(saved['step']):
        index=int(torch.rand((),generator=rng).item()<profile['mix_probability']);draws[index]+=1
        torch.randint(corpora[index]['files']['train']['tokens']-profile['seq'],(profile['batch'],),generator=rng)
    assert torch.equal(saved['rng'],rng.get_state())
    tensor_bytes=0
    def count(v):
        nonlocal tensor_bytes
        if isinstance(v,torch.Tensor):tensor_bytes+=v.numel()*v.element_size()
        elif isinstance(v,dict):
            for x in v.values():count(x)
        elif isinstance(v,(list,tuple)):
            for x in v:count(x)
    count(saved['optimizer']['state'])
    return dict(status='passed_checkpoint_checks',scope='Identity, finite model/optimizer, strict architecture/state shapes and accepted private data-RNG replay; not training/quality completion',checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),profile_sha256=digest(profile_path),auditor_sha256=digest(__file__),step=saved['step'],target_steps=profile['steps'],tokens=saved['tokens'],phase_complete=saved['step']==profile['steps'],accepted_training_draws=draws,optimizer_tensor_bytes=tensor_bytes,source_hashes=saved['source_hashes'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    report=audit(a.profile,a.checkpoint)
    with Path(a.out).open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report))
