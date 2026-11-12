"""Source/output integrity and resume-boundary tests, without model allocation."""
import copy
import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
import qwen35_quality_data as data
import audit_qwen35_quality as auditor


class QualityDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tokenizer = Tokenizer(models.WordLevel({'[UNK]':0,'same':1,'prompt':2,'other':3,'A':4,'STOP':5,'<eos>':6},unk_token='[UNK]'))
        self.tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
        self.tokenizer.decoder=decoders.WordPiece(prefix='##')
        self.tokenizer.save(str(self.root/'tokenizer.json'))
        self.task=SimpleNamespace(name='fixture',shots=0,cot=False,kind='choice',rows=[('same prompt','A'),('same other','A'),('same prompt other','A')])
        self.job=dict(name='fixture',tasks=['fixture'],target_items=3,tokens=2,context=8,batch_size=2,prefill_chunk=1,prefix_cache=True)
        self.adapter=SimpleNamespace(correct=lambda prediction,gold,kind:prediction==gold)

    def fixture(self, count=3):
        profile=self.root/'training.json';data.write_json(profile,{'config':{'vocab_size':8}})
        binding=dict(training_profile=str(profile),training_audit='fixture-only',checkpoint='fixture-only',arm='full')
        protocol=dict(arms=['full'],jobs=[self.job],generation={'fixture_only':True},eos=6,tokenizer=dict(file=str(self.root/'tokenizer.json'),sha256=data.digest(self.root/'tokenizer.json')))
        protocol_path=self.root/'protocol.json';data.write_json(protocol_path,protocol)
        rows,content,prefixes=data.prepare_jobs([self.task],self.tokenizer,self.job)
        identity=dict(protocol_sha256=data.digest(protocol_path),runtime_sources={'fixture_only':'not model evidence'},runtime_versions=data.runtime_versions(),binding=binding,generation=protocol['generation'],vocab_size=8,job='fixture',job_spec=self.job,tokenizer_sha256=protocol['tokenizer']['sha256'],content_sha256=content,prefix_plan_sha256=data.encoded_hash(prefixes))
        records=[]
        for task,index,prompt,gold,ids in rows[:count]:
            records.append(dict(task=task.name,index=index,prompt=prompt,gold=gold,input_ids=ids,answer_kind=task.kind,shots=task.shots,cot=task.cot,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),output_ids=[4,6],raw_prediction='A <eos>',prediction='A <eos>',unused_token_ids=[],correct=False,stop='eos'))
        path=self.root/'predictions.jsonl';path.write_text(''.join(json.dumps(row)+'\n' for row in records))
        counts=dict(fixture=dict(n=count,scored=count,correct=0,unknown_tokenizer_items=0))
        data.write_json(str(path)+'.meta.json',dict(identity=identity,status='complete' if count==3 else 'time_budget',completed_items=count,full_dataset_generated=count==3,tasks=counts))
        return path,binding,protocol_path,records

    def run_audit(self,path,binding,protocol_path,partial=False):
        with ExitStack() as stack:
            stack.enter_context(patch.object(auditor,'PROTOCOL',str(protocol_path)))
            stack.enter_context(patch.object(auditor,'training_binding',return_value=binding))
            stack.enter_context(patch.object(auditor,'adapter_for',return_value=(self.adapter,[self.task])))
            stack.enter_context(patch('evaluate_qwen35_quality.runtime_sources',return_value={'fixture_only':'not model evidence'}))
            return auditor.audit(path,partial)

    def test_stable_order_prefix_batches_and_capacity(self):
        rows,_,prefixes=data.prepare_jobs([self.task],self.tokenizer,self.job)
        self.assertEqual([row[1] for row in rows],[0,1,2])
        self.assertEqual(prefixes,{'fixture':[1]})
        self.assertEqual([len(batch) for batch in data.batches(rows,2)],[2,1])
        with self.assertRaises(ValueError):data.prepare_jobs([self.task],self.tokenizer,{**self.job,'context':4})
        with self.assertRaises(ValueError):data.selection({'arms':['full'],'jobs':[self.job]},'full','missing')

    def test_full_and_partial_audit(self):
        path,binding,protocol,records=self.fixture()
        self.assertEqual(self.run_audit(path,binding,protocol)['items'],3)
        path,binding,protocol,records=self.fixture(1)
        self.assertEqual(self.run_audit(path,binding,protocol,True)['coverage'],'partial')
        with self.assertRaises(AssertionError):self.run_audit(path,binding,protocol)

    def test_corruptions_and_duplicates_rejected(self):
        path,binding,protocol,records=self.fixture()
        mutations=[lambda r:r[1].update(index=r[0]['index']),lambda r:r[0].update(gold='B'),lambda r:r[0].update(input_ids=[1]),lambda r:r[0].update(raw_prediction='fabricated'),lambda r:r[0].update(correct=True),lambda r:r[0].update(output_ids=[8]),lambda r:r[0].update(stop='max_new_tokens')]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                bad=copy.deepcopy(records);mutation(bad);path.write_text(''.join(json.dumps(row)+'\n' for row in bad))
                with self.assertRaises(AssertionError):self.run_audit(path,binding,protocol)

    def test_padded_model_ids_retained_and_counted(self):
        path,binding,protocol,records=self.fixture()
        for row in records:
            row.update(output_ids=[7,6],unused_token_ids=[7],raw_prediction='<eos>',prediction='<eos>')
        path.write_text(''.join(json.dumps(row)+'\n' for row in records))
        meta=data.load(str(path)+'.meta.json');meta['tasks']['fixture']['unknown_tokenizer_items']=3;data.write_json(str(path)+'.meta.json',meta)
        self.assertEqual(self.run_audit(path,binding,protocol)['tasks']['fixture']['unknown_tokenizer_items'],3)

    def test_metadata_identity_prefix_and_counts_rejected(self):
        path,binding,protocol,records=self.fixture()
        original=data.load(str(path)+'.meta.json')
        mutations=[lambda m:m['identity'].update(prefix_plan_sha256='wrong'),
                   lambda m:m['identity'].update(generation={}),
                   lambda m:m['identity'].update(runtime_versions={}),
                   lambda m:m['tasks']['fixture'].update(correct=1),
                   lambda m:m.update(completed_items=2),
                   lambda m:m.update(full_dataset_generated=False),
                   lambda m:m.update(status='time_budget')]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                bad=copy.deepcopy(original);mutation(bad);data.write_json(str(path)+'.meta.json',bad)
                with self.assertRaises(AssertionError):self.run_audit(path,binding,protocol)

    def test_early_stop_overgeneration_rejected(self):
        path,binding,protocol,records=self.fixture()
        self.adapter.stopping=lambda name:['STOP']
        records[0].update(output_ids=[5,4],raw_prediction='STOP A',prediction='STOP A',stop='stop_string')
        path.write_text(''.join(json.dumps(row)+'\n' for row in records))
        with self.assertRaises(AssertionError):self.run_audit(path,binding,protocol)

    def test_incomplete_training_refused_before_checkpoint_io(self):
        profile=dict(arm='full',steps=20,seq=32768,formal_budget_sha256='fixture',out=str(self.root))
        protocol=dict(formal_budget_sha256='fixture')
        for steps, complete in [(2,True),(3,False),(19,False)]:
            receipt=dict(status='passed_checkpoint_checks',phase_complete=complete,step=steps,target_steps=20,total_tokens=3932160)
            def read(path):return protocol if path==data.PROTOCOL else profile if path=='profile' else receipt
            with self.subTest(steps=steps), patch.object(data,'load',side_effect=read), patch.object(data,'contract'), patch.object(data,'digest') as hashing:
                with self.assertRaises(AssertionError):data.training_binding('profile','audit',self.root/'last.pt','full')
                hashing.assert_not_called()


if __name__=='__main__':unittest.main()
