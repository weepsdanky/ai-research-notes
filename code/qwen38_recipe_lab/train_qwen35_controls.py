"""Independent Mac Qwen3.5 control training; public-family, not private Table1.

Use only through run_qwen35_control_guarded.py. Explicit JSON profiles bind all
Mac budgets and data. Optimizer, schedule and initialization differences must be
reported. Every completed update is atomically checkpointed; interruption during
an update preserves the previous accepted checkpoint without serializing a
partially updated optimizer/model. No padding labels or vocabulary truncation.
"""
import argparse,hashlib,json,os,platform,resource,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from chunked_lm_loss import chunked_cross_entropy
from qwen35_controls import ControlConfig,Qwen35Control
from optim import RecipeOptimizer,ownership


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


class Corpus:
    def __init__(self,path,expected_meta_sha,seq):
        self.path=Path(path);assert digest(self.path/'meta.json')==expected_meta_sha
        self.meta=json.loads((self.path/'meta.json').read_text())
        assert self.meta['tokenizer']=='Qwen/Qwen3.5-35B-A3B-Base'
        assert self.meta['tokenizer_revision']=='0f0813072d2358973511097385626f21fcb6d422'
        assert self.meta['tokenizer_sha256']=='fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927'
        self.arrays={}
        for split in ['train','val']:
            path=self.path/f'{split}.bin';assert digest(path)==self.meta['files'][split]['sha256']
            self.arrays[split]=np.memmap(path,dtype='<u4',mode='r')
            assert len(self.arrays[split])==self.meta['files'][split]['tokens']>seq
    def batch(self,split,batch,seq,rng,device):
        data=self.arrays[split];starts=torch.randint(len(data)-seq,(batch,),generator=rng).tolist()
        ids=torch.from_numpy(np.stack([data[i:i+seq+1] for i in starts]).astype(np.int64)).to(device)
        return ids[:,:-1],ids[:,1:]


def balance_loss(routers,experts,top_k):
    probs=torch.cat(routers).softmax(-1)
    selected=probs.topk(top_k,dim=-1).indices
    frequency=F.one_hot(selected,experts).float().mean((0,1))
    return experts*(frequency*probs.mean(0)).sum()


@torch.no_grad()
def evaluate(model,corpora,profile,device):
    model.eval();rows={}
    for index,corpus in enumerate(corpora):
        rng=torch.Generator().manual_seed(1234);losses=[]
        for _ in range(profile['eval_batches']):
            x,y=corpus.batch('val',profile['batch'],profile['seq'],rng,device)
            hidden,_=model.hidden(x)
            losses.append(chunked_cross_entropy(hidden,model.head.weight,y,profile['head_chunk']).item())
        rows[f'val{index}/lm']=sum(losses)/len(losses)
    model.train();return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',type=Path,required=True);p.add_argument('--seconds',type=int,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    # The parent guard owns the global lock and native pressure/deadline monitoring.
    guard_pid=int(os.environ['QWEN35_CONTROL_GUARD_PID']);assert os.getppid()==guard_pid
    assert os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO') is None and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') is None
    profile=json.loads(a.profile.read_text());assert 1<=a.seconds<=1140
    assert profile['status']=='resource_probe_candidate' and profile['scope']=='Mac public-family control, not private Table1'
    assert profile['schedule']=='constant' and profile['precision']=='float32'
    assert min(profile[k] for k in ['seq','steps','batch','eval_batches','eval_every','head_chunk'])>0
    assert profile['seq']>128,'SWA128 must remove real history'
    assert 0<=profile['mix_probability']<=1 and profile['lr']>0
    required_sources={'train_qwen35_controls.py','qwen35_controls.py','chunked_lm_loss.py','model.py','optim.py','polar_coefficients.json','audit_qwen35_training.py','run_qwen35_control_guarded.py','run_dense_text_guarded.py'}
    assert required_sources<=profile['source_hashes'].keys()
    sources={path:digest(path) for path in profile['source_hashes']}
    assert all(digest(path)==sha for path,sha in profile['source_hashes'].items())
    config=ControlConfig(**profile['config']);device=profile['device'];assert device in ['cpu','mps']
    if device=='mps':assert torch.backends.mps.is_available()
    torch.manual_seed(profile['seed']);torch.set_num_threads(1)
    corpora=[Corpus(row['directory'],row['meta_sha256'],profile['seq']) for row in profile['data']]
    assert len(corpora)==2
    out=Path(profile['out']);checkpoint=out/'last.pt';saved=None
    if a.resume:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert saved['profile_sha256']==digest(a.profile) and saved['source_hashes']==sources
        assert saved['corpora']==[c.meta for c in corpora]
        assert saved['step']<profile['steps'],'Already complete'
    else:out.mkdir(parents=True,exist_ok=False)
    model=Qwen35Control(config,profile['arm'],seed=profile['seed']).to(device)
    optimizer=RecipeOptimizer(ownership(model),profile['lr'],weight_decay=profile['weight_decay'],ns_dtype=profile['ns_dtype'])
    rng=torch.Generator().manual_seed(profile['seed']);step=0
    if saved:
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);rng.set_state(saved['rng']);step=saved['step']
    del saved
    def log(row):
        with (out/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    def save():
        state=dict(profile=profile,profile_sha256=digest(a.profile),source_hashes=sources,config=asdict(config),arm=profile['arm'],model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng.get_state(),corpora=[c.meta for c in corpora],step=step,tokens=step*profile['seq']*profile['batch'])
        torch.save(state,out/'last.tmp');(out/'last.tmp').replace(checkpoint)
    if not a.resume:
        save();log(dict(step=0,**evaluate(model,corpora,profile,device)))
    began=time.monotonic();model.train()
    for step_next in range(step+1,profile['steps']+1):
        tick=time.monotonic();optimizer.zero_grad(set_to_none=True)
        index=int(torch.rand((),generator=rng).item()<profile['mix_probability'])
        x,y=corpora[index].batch('train',profile['batch'],profile['seq'],rng,device)
        hidden,routers=model.hidden(x);lm=chunked_cross_entropy(hidden,model.head.weight,y,profile['head_chunk'])
        balance=balance_loss(routers,config.experts,config.top_k);loss=lm+profile['balance_weight']*balance
        assert torch.isfinite(loss).item()
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),profile['clip'],error_if_nonfinite=True,foreach=False)
        optimizer.step()
        if device=='mps':torch.mps.synchronize()
        step=step_next;save()  # Only serialize fully accepted updates, before evaluation.
        seconds=time.monotonic()-tick;stop=step==profile['steps'] or time.monotonic()-began>=a.seconds
        row=dict(step=step,tokens=step*profile['seq']*profile['batch'],lm=lm.item(),balance=balance.item(),grad_norm=norm.item(),step_seconds=seconds,tokens_per_second=profile['seq']*profile['batch']/seconds,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024))
        if device=='mps':row['mps_driver_bytes']=torch.mps.driver_allocated_memory()
        if step%profile['eval_every']==0 or stop:row.update(evaluate(model,corpora,profile,device))
        log(row)
        if stop:log(dict(step=step,status='complete' if step==profile['steps'] else 'time_budget',checkpoint=str(checkpoint)));break


if __name__=='__main__':main()
