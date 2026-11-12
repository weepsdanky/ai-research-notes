"""Real model/full-prefix oracles and forced decoder paths; no trained MTP score."""
import ast,hashlib,json,os,unittest
from dataclasses import dataclass,field
from pathlib import Path
import torch
from model import Model
from mtp import MultiTokenPredictor
from speculative_mtp import GreedyMTP,verify_greedy
from test_recipe import tiny


def public_stats(rows,steps):
    """Execute only the locked upstream stats class and logging expression."""
    item=next(x for x in json.loads(Path('mtp_proposer_sources.lock.json').read_text()) if x['upstream_path']=='vllm/v1/spec_decode/metrics.py')
    raw=Path(item['file']).read_bytes();assert hashlib.sha256(raw).hexdigest()==item['sha256']
    tree=ast.parse(raw);node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SpecDecodingStats')
    namespace=dict(dataclass=dataclass,field=field,__name__=__name__)
    exec(compile(ast.Module(body=[node],type_ignores=[]),item['file'],'exec'),namespace)
    stats=namespace['SpecDecodingStats'].new(steps)
    for row in rows:stats.observe_draft(row['drafted'],row['accepted'])
    expression=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='mean_acceptance_length' for t in n.targets))
    mean=eval(compile(ast.Expression(expression),item['file'],'eval'),{},dict(num_accepted_tokens=stats.num_accepted_tokens,num_drafts=stats.num_drafts)) if stats.num_drafts else None
    return stats,mean


@torch.no_grad()
def full_prefix_greedy(model,prompt,count,stage,eos=()):
    ids=prompt.clone();result=[]
    for _ in range(count):
        token=int(model(ids,stage)[0][0,-1].argmax());result.append(token)
        if token in eos:break
        ids=torch.cat((ids,ids.new_tensor([[token]])),1)
    return result


class OracleDraft(GreedyMTP):
    """Force proposal paths while still executing the real MTP cache/fusion."""
    rejection_positions=()
    def _propose(self,prefix,ids,hidden,cache,count):
        self.round_index=getattr(self,'round_index',-1)+1
        # Independent full Model.forward exposes its final residual to readout.
        captured=[];hook=self.target.readout.register_forward_pre_hook(lambda m,args:captured.append(args[0]))
        try:self.target(prefix[:,:-1],self.stage)
        finally:hook.remove()
        torch.testing.assert_close(hidden,captured[0][:,-ids.shape[1]:],atol=4e-5,rtol=4e-4)
        torch.testing.assert_close(ids,prefix[:,-ids.shape[1]:],atol=0,rtol=0)
        return super()._propose(prefix,ids,hidden,cache,count)

    def _sample_draft(self,single,prefix,proposals):
        context=torch.cat((prefix,prefix.new_tensor([proposals])),1) if proposals else prefix
        token=int(self.target(context,self.stage)[0][0,-1].argmax())
        if self.rejection_positions and len(proposals)==self.rejection_positions[self.round_index%len(self.rejection_positions)]:
            token=(token+1)%self.target.config.vocab_size
        return token


class SpeculativeMTPTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(713);self.device=os.environ.get('LAB_TEST_DEVICE','cpu')
        self.model=Model(tiny()).to(self.device).eval();self.mtp=MultiTokenPredictor(self.model.config).to(self.device).eval()
        self.prompt=torch.tensor([[1,4,2,9,3,6,5,7,8,2,1,4,6,5,9,8,1]],device=self.device)

    def test_verifier_rejections_bonus_and_eos(self):
        draft=[2,3,4,5]
        for position in range(4):
            target=draft+[6];target[position]=9
            accepted,out,kind=verify_greedy(draft,target)
            self.assertEqual((accepted,out,kind),(position,draft[:position]+[9],'correction'))
        self.assertEqual(verify_greedy(draft,draft+[6]),(4,[2,3,4,5,6],'bonus'))
        self.assertEqual(verify_greedy(draft,draft+[6],{4}),(3,[2,3,4],'accepted_eos'))
        self.assertEqual(verify_greedy(draft,[2,9,4,5,6],{9}),(1,[2,9],'correction'))
        self.assertEqual(verify_greedy(draft,draft+[6],{6}),(4,[2,3,4,5,6],'bonus'))
        with self.assertRaises(ValueError):verify_greedy([1],[1])

    def test_real_untrained_drafter_matches_independent_teacher(self):
        for stage in ('pretrain','sparse'):
            for length in (1,7,17):
                with self.subTest(stage=stage,length=length):
                    prompt=self.prompt[:,:length];decoder=GreedyMTP(self.model,self.mtp,stage)
                    result=decoder.generate(prompt,23)
                    self.assertEqual(result['output_ids'],full_prefix_greedy(self.model,prompt,23,stage))
                    self.assertTrue(any(row.get('target_replayed') for row in result['rounds']))
                    self.assertEqual(len(result['output_ids']),23)
                    for row in result['rounds']:
                        if 'drafted' in row:self.assertEqual(row['mtp_cache_before_rollback'],row['seed_position']+row['drafted']-1)

    def test_last_only_prefill_preserves_tokens_and_verification_trace(self):
        for stage in ('pretrain', 'sparse'):
            old = GreedyMTP(self.model, self.mtp, stage).generate(self.prompt, 23)
            new = GreedyMTP(self.model, self.mtp, stage, prefill_last_only=True).generate(self.prompt, 23)
            self.assertEqual(new, old)

    def test_all_accept_and_bonus_are_five_not_four(self):
        for stage in ('pretrain','sparse'):
            decoder=OracleDraft(self.model,self.mtp,stage);result=decoder.generate(self.prompt,16)
            self.assertEqual(result['output_ids'],full_prefix_greedy(self.model,self.prompt,16,stage))
            self.assertEqual(result['num_drafts'],3);self.assertEqual(result['num_accepted_tokens'],12)
            self.assertEqual(result['mean_acceptance_length_vllm'],5)
            self.assertTrue(all(row['kind']=='bonus' and row['emitted']==5 and not row['target_replayed'] for row in result['rounds']))
            stats,mean=public_stats(result['rounds'],4)
            self.assertEqual(mean,result['mean_acceptance_length_vllm']);self.assertEqual(stats.num_accepted_tokens_per_pos,[3,3,3,3])

    def test_every_rejection_position_recovers_gdn_ngram_and_mtp(self):
        for stage in ('pretrain','sparse'):
            decoder=OracleDraft(self.model,self.mtp,stage);decoder.rejection_positions=(0,1,2,3)
            result=decoder.generate(self.prompt,41)
            self.assertEqual(result['output_ids'],full_prefix_greedy(self.model,self.prompt,41,stage))
            corrections=[row for row in result['rounds'] if row['kind']=='correction']
            self.assertEqual({row['accepted'] for row in corrections},{0,1,2,3})
            self.assertTrue(all(row['target_replayed'] for row in corrections))
            stats,mean=public_stats([row for row in result['rounds'] if 'drafted' in row],4)
            self.assertEqual(stats.num_accepted_tokens,result['num_accepted_tokens']);self.assertEqual(mean,result['mean_acceptance_length_vllm'])

    def test_eos_and_output_boundaries(self):
        first=full_prefix_greedy(self.model,self.prompt,1,'sparse')[0]
        decoder=GreedyMTP(self.model,self.mtp)
        result=decoder.generate(self.prompt,20,eos_token_ids={first})
        self.assertEqual(result['output_ids'],[first]);self.assertEqual(result['stop'],'eos');self.assertEqual(result['num_drafts'],0)
        sequence=full_prefix_greedy(self.model,self.prompt,16,'sparse')
        stop=next(token for token in sequence[1:] if token!=first)
        for reject in (False,True):
            controlled=OracleDraft(self.model,self.mtp)
            if reject:controlled.rejection_positions=(0,)
            result=controlled.generate(self.prompt,20,eos_token_ids={stop})
            self.assertEqual(result['output_ids'],full_prefix_greedy(self.model,self.prompt,20,'sparse',{stop}))
            self.assertEqual(result['stop'],'eos')
            if not reject:self.assertEqual(result['rounds'][-1]['kind'],'accepted_eos')
        for count in (0,1,2,5,6,7):
            result=OracleDraft(self.model,self.mtp).generate(self.prompt,count)
            self.assertEqual(result['output_ids'],full_prefix_greedy(self.model,self.prompt,count,'sparse'))
            self.assertEqual(len(result['output_ids']),count)
        with self.assertRaises(ValueError):decoder.generate(self.prompt,-1)
        with self.assertRaises(ValueError):decoder.generate(self.prompt,3,draft_tokens=0)
        with self.assertRaises(ValueError):decoder.generate(self.prompt.expand(2,-1),3)
        with self.assertRaises(ValueError):GreedyMTP(self.model,self.mtp,context_limit=18).generate(self.prompt,2)
        with torch.no_grad():self.mtp.fc_embedding.weight.fill_(float('nan'))
        with self.assertRaises(FloatingPointError):decoder.generate(self.prompt,3)


if __name__=='__main__':unittest.main()
