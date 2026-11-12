"""Verify exact phase boundary, finite state and private RNG across both contexts."""
import argparse,json
from pathlib import Path
import torch
from qwen35_long_phase import load_parent,exact
from qwen35_controls import ControlConfig,Qwen35Control
from optim import ownership,view
from audit_qwen35_training import finite,digest


def replay(profile,steps,rng,corpora):
    for _ in range(steps):
        index=int(torch.rand((),generator=rng).item()<profile['mix_probability'])
        torch.randint(corpora[index]['files']['train']['tokens']-profile['seq'],(profile['batch'],),generator=rng)


def audit(profile_path,checkpoint):
    torch.set_num_threads(1);profile=json.loads(Path(profile_path).read_text());parent=load_parent(profile)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert saved['profile']==profile and saved['profile_sha256']==digest(profile_path)
    assert saved['source_hashes']==profile['source_hashes'] and all(digest(p)==sha for p,sha in saved['source_hashes'].items())
    assert saved['config']==profile['config'] and saved['arm']==profile['arm'] and saved['phase']=='32k_continuation'
    assert saved['parent_checkpoint_sha256']==profile['parent_checkpoint_sha256']
    step=saved['step'];assert 0<=step<=2 and saved['phase_tokens']==step*32768*profile['batch']
    assert saved['tokens']==parent['tokens']+saved['phase_tokens'] and saved['optimizer_steps_offset']==80 and saved['parent_training_tokens']==parent['tokens']
    initial_path=Path(profile['out'])/'initial.pt';initial=torch.load(initial_path,map_location='cpu',weights_only=True)
    assert initial['step']==0 and initial['phase_tokens']==0 and initial['tokens']==parent['tokens']
    assert initial['profile_sha256']==digest(profile_path) and initial['parent_checkpoint_sha256']==profile['parent_checkpoint_sha256']
    assert initial['profile']==profile and initial['source_hashes']==profile['source_hashes'] and initial['phase']=='32k_continuation'
    for key in ['model','optimizer','rng','corpora']:exact(initial[key],parent[key])
    finite(saved['model']);finite(saved['optimizer'])
    model=Qwen35Control(ControlConfig(**saved['config']),saved['arm'],seed=profile['seed']);model.load_state_dict(saved['model'],strict=True)
    plans=ownership(model);groups=saved['optimizer']['param_groups'];assert len(plans)==len(groups)
    ids=[g['params'][0] for g in groups];assert len(ids)==len(set(ids)) and set(saved['optimizer']['state'])<=set(ids)
    optimizer_bytes=0
    for (name,p,specs),group in zip(plans,groups,strict=True):
        assert group['name']==name and group['pieces']==specs and len(group['params'])==1
        assert group['lr']==profile['lr'] and group['weight_decay']==profile['weight_decay'] and group['ns_dtype']==profile['ns_dtype']
        assert tuple(group['adam_betas'])==(.9,.95) and group['adam_eps']==1e-8
        state=saved['optimizer']['state'].get(group['params'][0],{})
        if not state:continue
        assert len(state['pieces'])==len(specs)
        for spec,buffers in zip(specs,state['pieces'],strict=True):
            tensor=view(p,spec)
            names=['momentum'] if spec['kind']=='muon' else ['m','v']
            for key in names:
                assert buffers[key].shape==tensor.shape and buffers[key].dtype==tensor.dtype
                optimizer_bytes+=buffers[key].numel()*buffers[key].element_size()
            if spec['kind']!='muon':assert 0<buffers['step']<=80+step
    corpora=[]
    for row in profile['data']:
        path=Path(row['directory']);assert digest(path/'meta.json')==row['meta_sha256'];meta=json.loads((path/'meta.json').read_text());corpora.append(meta)
        for split in ['train','val']:assert digest(path/f'{split}.bin')==meta['files'][split]['sha256']
    assert corpora==parent['corpora']==saved['corpora']
    rng=torch.Generator().manual_seed(profile['seed']);replay(parent['profile'],80,rng,corpora);assert torch.equal(rng.get_state(),parent['rng'])
    replay(profile,step,rng,corpora);assert torch.equal(rng.get_state(),saved['rng'])
    return dict(status='passed_checkpoint_checks',scope='Exact model/optimizer/private RNG boundary plus finite long-phase state, strict shapes and both-context data replay; not quality/Table1 completion',profile_sha256=digest(profile_path),checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),auditor_sha256=digest(__file__),step=step,target_steps=2,phase_complete=step==2,phase_tokens=saved['phase_tokens'],total_tokens=saved['tokens'],parent_checkpoint_sha256=profile['parent_checkpoint_sha256'],initial_checkpoint=str(initial_path),initial_checkpoint_sha256=digest(initial_path),parent_model_optimizer_rng_bitwise_preserved=True,optimizer_tensor_bytes=optimizer_bytes,source_hashes=saved['source_hashes'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);a=p.parse_args();result=audit(a.profile,a.checkpoint)
    with Path(a.out).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'}))
