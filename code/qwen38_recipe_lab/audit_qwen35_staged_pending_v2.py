"""Independent finite/shape/progress and private batch replay checks for pending work."""
import argparse,math
from pathlib import Path
import torch
from qwen35_quality_data import load,digest,write_json
from qwen35_formal_contract import contract
from train_qwen35_controls import Corpus


def finite(value,shape):
    assert value.shape==shape and value.dtype==torch.float32 and value.device.type=='cpu'
    for part in value.reshape(-1).split(1048576):assert torch.isfinite(part).all().item()


def audit(profile_path):
    torch.set_num_threads(1);profile=load(profile_path);contract(profile,'formal_long');out=Path(profile['out']);cp=out/'last.pt';path=out/'pending.pt';sha=digest(cp)
    base=torch.load(cp,map_location='cpu',weights_only=True,mmap=True);p=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    assert p['checkpoint_sha256']==sha and p['profile_sha256']==base['profile_sha256']==digest(profile_path) and p['target_step']==base['step']+1<=20
    assert math.isfinite(p['active_seconds']) and p['active_seconds']>=0 and p['peak_rss_bytes']>0
    c=profile['config'];batch=profile['batch'];seq=32768;layers=c['layers'];chunk=8;count=(batch*seq+chunk-1)//chunk
    rng=torch.Generator();rng.set_state(base['rng']);i=int(torch.rand((),generator=rng).item()<profile['mix_probability']);data=profile['data'][i]
    x,y=Corpus(data['directory'],data['meta_sha256'],seq).batch('train',batch,seq,rng,'cpu')
    s=p['update'];assert s['spec']==dict(version=1,binding=sha,config=dict(c,query_chunk=32),arm=profile['arm'],chunk=chunk,balance_weight=profile['balance_weight'])
    assert torch.equal(x,s['x']) and torch.equal(y,s['targets']) and torch.equal(rng.get_state(),p['rng_after_draw'])
    phase=s['phase'];f=len(s['routers']);ce=s['ce_done'];width=(batch,seq,c['width']);route=(batch*seq,c['experts'])
    assert 0<=ce<=count and len(s['losses'])==ce and 0<=f<=layers
    assert len(s['inputs'])==(0 if phase=='embedding_forward' else f+1)
    for value in s['inputs']:finite(value,width)
    for value in s['routers']:finite(value,route)
    for value in s['losses']:finite(value,())
    if phase in ['embedding_forward','layer_forward','norm_forward']:assert ce==0
    elif phase=='head':assert ce<count and f==layers
    else:assert ce==count and f==layers
    if phase=='layer_forward':assert f<layers
    if phase=='norm_forward':assert f==layers
    back=layers-1-s.get('backward_index',layers-1)
    units={'embedding_forward':0,'layer_forward':1+f,'norm_forward':1+layers,'head':2+layers+ce,
           'aux':2+layers+count,'norm_backward':3+layers+count,'layer_backward':4+layers+count+back,
           'embedding_backward':4+2*layers+count,'done':5+2*layers+count}[phase]
    assert s['units']==units
    if phase=='layer_backward':assert 0<=back<layers
    if ce:finite(s['hidden_adjoint'],width)
    if phase in ['head','aux','norm_backward','layer_backward','embedding_backward','done']:finite(s['hidden'],width)
    if phase in ['norm_backward','layer_backward','embedding_backward','done']:
        assert len(s['router_adjoints'])==layers
        for value in s['router_adjoints']:finite(value,route)
        finite(s['balance'],())
    if ce==count:finite(s['lm'],())
    if phase in ['layer_backward','embedding_backward','done']:finite(s['adjoint'],width)
    allowed={'head.weight'} if ce else set()
    if phase in ['layer_backward','embedding_backward','done']:allowed.add('norm.weight')
    if phase in ['layer_backward','embedding_backward','done']:
        finished=back if phase=='layer_backward' else layers
        allowed.update(name for name in base['model'] if any(name.startswith(f'layers.{j}.') for j in range(layers-finished,layers)))
    if phase=='done':allowed.add('embedding.weight')
    assert s['grads'].keys()<=allowed
    assert all(name in s['grads'] for name in allowed if not name.startswith('layers.'))
    for name,value in s['grads'].items():finite(value,base['model'][name].shape)
    return dict(status='passed_pending_state_checks',pending=str(path),pending_sha256=digest(path),checkpoint_sha256=sha,profile_sha256=digest(profile_path),target_step=p['target_step'],phase=phase,units=units,ce_chunks=ce,actual_context=seq,router_shape=list(route),accepted_new_formal_updates=0,scope='Finite tensor/shape/progress and private batch replay; gradients are not a completed optimizer update')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--profile',required=True);parser.add_argument('--out',required=True);a=parser.parse_args();assert not Path(a.out).exists();write_json(a.out,audit(a.profile))
