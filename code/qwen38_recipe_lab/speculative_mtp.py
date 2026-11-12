"""B1 greedy MTP reference, aligned to the pinned public vLLM proposer.

Verification is batched. Rejection restores the target cache and replays its
accepted prefix because a GDN recurrent state cannot be truncated like KV.
This eager reference is not a CUDA speedup or a Table4 reproduction.
"""
import torch
from inference import Inference
from mtp_inference import IncrementalMTP


def copy_cache(value):
    if isinstance(value,torch.Tensor):return value.clone()
    if isinstance(value,dict):return {key:copy_cache(item) for key,item in value.items()}
    return value


def verify_greedy(proposals,target_tokens,eos=()):
    """Target has K+1 predictions; return accepted drafts and recovery/bonus."""
    if len(target_tokens)!=len(proposals)+1:raise ValueError('Verification alignment')
    accepted=[]
    for i,token in enumerate(proposals):
        if token!=target_tokens[i]:return i,accepted+[target_tokens[i]],'correction'
        accepted.append(token)
        if token in eos:return i+1,accepted,'accepted_eos'
    return len(proposals),accepted+[target_tokens[-1]],'bonus'


class GreedyMTP:
    def __init__(self,target,draft,stage='sparse',context_limit=8192,*,prefill_last_only=False):
        if target.config!=draft.c:raise ValueError('Target/draft configuration mismatch')
        if target.training or draft.training:raise ValueError('Evaluation mode required')
        self.target,self.draft,self.stage,self.context_limit=target,draft,stage,context_limit
        self.teacher=Inference(target,stage,grouped_gqa=True)
        self.proposer=IncrementalMTP(draft,stage)
        self.prefill_last_only=prefill_last_only

    def _target(self,ids,cache):
        hidden=self.teacher.hidden(ids,cache)
        logits=self.target.head(self.target.readout(hidden))
        if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite target logits')
        return logits,hidden

    def _sample_draft(self,single,prefix,proposals):
        logits=self.target.head(single[:,-1:])[0,0]
        if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite draft logits')
        return int(logits.argmax())

    def _propose(self,prefix,ids,hidden,cache,count):
        # At absolute P, embed token P+1 with target hidden P to predict P+2.
        single,multi,_=self.proposer(self.target.embedding(ids),hidden,cache)
        proposals=[]
        for i in range(count):
            token=self._sample_draft(single,prefix,proposals)
            proposals.append(token)
            if token in self.eos:break
            if i+1<count:
                single,multi,_=self.proposer(self.target.embedding(ids.new_tensor([[token]])),multi[:,-1:],cache,reuse=self.stage=='sparse')
        return proposals

    @torch.no_grad()
    def generate(self,prompt,max_new_tokens=64,draft_tokens=4,eos_token_ids=()):
        if prompt.ndim!=2 or prompt.shape[0]!=1 or prompt.shape[1]<1:raise ValueError('One nonempty request required')
        if max_new_tokens<0 or draft_tokens<1:raise ValueError('Generation budget')
        if prompt.shape[1]+max_new_tokens>self.context_limit:raise ValueError('Context budget')
        self.eos=set(eos_token_ids);output=[];trace=[];target_calls=0;draft_calls=0
        if max_new_tokens:
            cache={}
            if self.prefill_last_only:
                hidden=self.teacher.hidden(prompt,cache)
                logits=self.target.head(self.target.readout(hidden[:,-1:]))
                if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite target seed logits')
            else:
                logits,hidden=self._target(prompt,cache)
            target_calls+=1
            seed=int(logits[0,-1].argmax());output.append(seed)
            ready_ids=torch.cat((prompt[:,1:],prompt.new_tensor([[seed]])),1)
            ready_hidden=hidden;draft_cache={};base=prompt.shape[1]
            while len(output)<max_new_tokens and output[-1] not in self.eos:
                count=min(draft_tokens,max_new_tokens-len(output)-1)
                if count==0:
                    logits,_=self._target(prompt.new_tensor([[output[-1]]]),cache);target_calls+=1
                    output.append(int(logits[0,-1].argmax()))
                    trace.append(dict(kind='target_only_boundary',emitted=1));break
                prefix=torch.cat((prompt,prompt.new_tensor([output])),1)
                proposals=self._propose(prefix,ready_ids,ready_hidden,draft_cache,count)
                if not proposals or len(proposals)>count:raise ValueError('Proposal count')
                draft_calls+=len(proposals)
                step0_length=base  # MTP positions trail input token positions by one.
                prior=copy_cache(cache)
                query=prompt.new_tensor([[output[-1]]+proposals])
                logits,verified_hidden=self._target(query,cache);target_calls+=1
                target_tokens=logits[0].argmax(-1).tolist()
                accepted,emitted,kind=verify_greedy(proposals,target_tokens,self.eos)
                consumed=accepted+1
                replayed=consumed<query.shape[1]
                if replayed:
                    cache=prior
                    _,verified_hidden=self._target(query[:,:consumed],cache);target_calls+=1
                output.extend(emitted)
                trace.append(dict(kind=kind,drafted=len(proposals),accepted=accepted,emitted=len(emitted),
                                  proposals=proposals,target_tokens=target_tokens,seed_position=base,
                                  target_consumed=consumed,target_replayed=replayed,
                                  mtp_selected_count=int(draft_cache['selection'][0,-1]) if self.stage=='sparse' else base,
                                  mtp_cache_before_rollback=draft_cache['k'].shape[1],mtp_keep=step0_length))
                base+=consumed
                if output[-1] in self.eos or len(output)==max_new_tokens:break
                # Discard recursive draft hidden states even for accepted drafts.
                # Refill them from the target verification hiddens next round.
                self.proposer.rollback(draft_cache,step0_length)
                ready_ids=prompt.new_tensor([proposals[:accepted]+[emitted[-1]]])
                ready_hidden=verified_hidden
        rounds=[row for row in trace if 'drafted' in row]
        accepted=sum(row['accepted'] for row in rounds)
        return dict(output_ids=output,stop='eos' if output and output[-1] in self.eos else 'max_new_tokens',
                    rounds=trace,target_calls=target_calls,draft_calls=draft_calls,
                    num_drafts=len(rounds),num_draft_tokens=sum(row['drafted'] for row in rounds),
                    num_accepted_tokens=accepted,
                    mean_acceptance_length_vllm=1+accepted/len(rounds) if rounds else None,
                    actual_emitted_tokens_in_draft_rounds=sum(row['emitted'] for row in rounds),
                    scope='B1 greedy eager reference; conventional metric includes bonus; EOS/budget may truncate actual emission')
