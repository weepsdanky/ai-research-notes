"""Generate a bounded real CPU sample from an independently audited control.

Recompute the full prefix each token: this is a qualitative sample, not a cache,
long-context benchmark, official chat template or sparse-kernel speed claim.
"""
import argparse,fcntl,json,os,re,subprocess,time
from pathlib import Path
from datetime import datetime,timedelta
import torch
from tokenizers import Tokenizer
from qwen35_controls import Qwen35Control,ControlConfig
from train_qwen35_controls import digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--audit',required=True);p.add_argument('--out',required=True);p.add_argument('--prompt',required=True);p.add_argument('--tokens',type=int,default=32);p.add_argument('--temperature',type=float,default=.8);p.add_argument('--seed',type=int,default=42);p.add_argument('--deadline',required=True);a=p.parse_args()
    deadline=datetime.fromisoformat(a.deadline);assert deadline.tzinfo and 1<=a.tokens<=64 and a.temperature>=0
    assert not Path(a.out).exists() and datetime.now().astimezone()+timedelta(seconds=660)<deadline
    with Path('runs/hf-mmmlu-global.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert int(subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True))==1
        assert int(re.search(r'free percentage:\s*(\d+)%',subprocess.check_output(['memory_pressure','-Q'],text=True))[1])>=55
        assert os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO') is None and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') is None
        audit=json.loads(Path(a.audit).read_text());assert audit['status']=='passed_checkpoint_checks' and audit['checkpoint_sha256']==digest(a.checkpoint)
        saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
        assert all(digest(path)==sha for path,sha in saved['source_hashes'].items())
        spec=json.loads(Path('qwen35_tokenizer_sources.lock.json').read_text())[0];assert digest(spec['file'])==spec['sha256']
        tokenizer=Tokenizer.from_file(spec['file']);prompt=tokenizer.encode(a.prompt,add_special_tokens=False).ids
        assert prompt and len(prompt)<=256 and len(prompt)+a.tokens<=saved['profile']['seq']
        torch.set_num_threads(1);model=Qwen35Control(ControlConfig(**saved['config']),saved['arm'],seed=saved['profile']['seed'])
        model.load_state_dict(saved['model'],strict=True);arm=saved['arm'];step=saved['step'];phase=saved.get('phase','4k_short');training_tokens=saved['tokens'];del saved;model.eval()
        ids=list(prompt);generated=[];rng=torch.Generator().manual_seed(a.seed);started=time.monotonic();status='complete'
        with torch.no_grad():
            for _ in range(a.tokens):
                level=int(subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True))
                if level>=2 or datetime.now().astimezone()>=deadline-timedelta(seconds=600) or time.monotonic()-started>=60:
                    status='resource_or_time_stop';break
                hidden,_=model.hidden(torch.tensor([ids],dtype=torch.long));logits=model.head(hidden[:,-1])[0]
                token=int(logits.argmax()) if a.temperature==0 else int(torch.multinomial((logits/a.temperature).softmax(-1),1,generator=rng))
                ids.append(token);generated.append(token)
                if token==248044:status='eos';break
        result=dict(checkpoint=a.checkpoint,checkpoint_sha256=digest(a.checkpoint),audit_sha256=digest(a.audit),source_sha256=digest(__file__),arm=arm,phase=phase,phase_step=step,training_tokens=training_tokens,prompt=a.prompt,prompt_ids=prompt,generated_ids=generated,text=tokenizer.decode(generated,skip_special_tokens=False),unknown_to_tokenizer_ids=[i for i in generated if i>=tokenizer.get_vocab_size()],temperature=a.temperature,seed=a.seed,requested_tokens=a.tokens,actual_tokens=len(generated),status=status,seconds=time.monotonic()-started,execution='CPU full-prefix recomputation, full vocabulary, no chat template')
        with Path(a.out).open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
        print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
