"""Actual CPU/MPS initial equivalence and full data-RNG replay; no training."""
import argparse,fcntl,hashlib,json,os,resource,subprocess,sys,time
from datetime import datetime
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import torch
from model import Config,Model
from train import Corpus

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def gate(deadline):
 assert datetime.now().astimezone()<deadline
 assert int(subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True))==1

def main():
 p=argparse.ArgumentParser();p.add_argument('--deadline',required=True);p.add_argument('--out',required=True);a=p.parse_args();deadline=datetime.fromisoformat(a.deadline);out=Path(a.out);assert not out.exists();profile=json.loads(Path('profiles/paper_dense_control.json').read_text());gate(deadline)
 assert os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO') is None and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') is None
 torch.set_num_threads(1);tick=time.monotonic();record=dict(started=datetime.now().astimezone().isoformat(),status='running',profile_sha256=sha('profiles/paper_dense_control.json'),runner_sha256=sha(__file__),training_started=False,tests_rerun=False,devices={},sources={f:sha(f) for f in ('model.py','optim.py','polar_coefficients.json','train.py','audit_stage.py','uv.lock','pyproject.toml')},mps_high_watermark_ratio='default',mps_fallback='default')
 with Path('runs/hf-mmmlu-global.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   checkpoints={}
   for key in ('init','sparse_init','comparison'):
    assert sha(profile[key])==profile[key+'_sha256'];checkpoints[key]=torch.load(profile[key],map_location='cpu',weights_only=True,mmap=True)
   before=checkpoints['init'];after=checkpoints['sparse_init'];trained=checkpoints['comparison'];assert before['config']==after['config']==trained['config'];assert before['model'].keys()==after['model'].keys()
   changed=[];active=0
   for name,value in before['model'].items():
    if '.indexer.' in name:
     if not torch.equal(value,after['model'][name]):changed.append(name)
    else:assert torch.equal(value,after['model'][name]),name;active+=value.numel()
   assert changed;record.update(all_non_indexer_tensors_bit_exact=True,non_indexer_elements=active,changed_unused_indexer_tensors=changed)
   assert trained['step']==profile['steps']==8000 and trained['args']['stage']=='sparse'
   for key in ('seq','batch','accum','lr','seed','clip','optimizer','ns_dtype','weight_decay'):assert trained['args'][key]==profile[key]
   assert trained['args']['schedule'] in ('auto','constant')
   assert all(sha(name)==wanted for name,wanted in profile['source_hashes'].items())
   corpora=[Corpus(name,profile['seq']) for name in profile['data']];assert [c.meta for c in corpora]==trained['corpora'];rng=torch.Generator().manual_seed(profile['seed']);h=hashlib.sha256();counts=[0,0]
   for _ in range(profile['steps']):
    index=int(torch.rand((),generator=rng).item()<profile['mix_probability']);counts[index]+=1
    x,y=corpora[index].batch('train',profile['batch'],profile['seq'],rng,'cpu');h.update(bytes([index]));h.update(x.numpy().astype('<u4').tobytes());h.update(y[:,-1].numpy().astype('<u4').tobytes())
   assert torch.equal(rng.get_state(),trained['rng']);record.update(all8000_training_data_draws_replayed=True,training_input_sequence_sha256=h.hexdigest(),training_corpus_draws=counts)
   ids=np.fromfile(profile['data'][0]+'/val.bin',dtype='<u4',count=64).astype(np.int64);record['prompt_ids']=ids.tolist()
   for device in ('cpu','mps'):
    gate(deadline);logits=[]
    for saved in (before,after):
     model=Model(Config(**saved['config'])).eval();model.load_state_dict(saved['model'],assign=True);model=model.to(device);model.stage('pretrain')
     prompt=torch.from_numpy(ids)[None].to(device);actual_device=str(prompt.device);assert prompt.device.type==device
     with torch.no_grad():output=model(prompt,'pretrain')[0]
     assert torch.isfinite(output).all();logits.append(output.cpu());del output,model,prompt
    assert torch.equal(*logits),'Dense initialization logits differ';record['devices'][device]=dict(actual_device=actual_device,dense_logits_bit_exact=True,shape=list(logits[0].shape),max_absolute_difference=0.0);del logits
    if device=='mps':torch.mps.empty_cache()
   gate(deadline);record['status']='passed'
  except BaseException as error:record.update(status='failed_or_interrupted',error_type=type(error).__name__);raise
  finally:
   record.update(ended=datetime.now().astimezone().isoformat(),seconds=time.monotonic()-tick,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ending_pressure=int(subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True)));out.write_text(json.dumps(record,indent=2)+'\n')
 print(record['status'])


if __name__=='__main__':main()
