"""Public job-planning fixtures only: no actual formal model/VM judging."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from qwen35_quality_data import digest, write_json
import qwen35_code_jobs as module


class CodeJobsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def fixture(self,suite):
        if suite=='code':
            from eval_code_tasks import datasets
            data={(name,row['task_id']):row for name,rows in datasets() for row in rows}
        else:
            from eval_multipl_tasks import datasets
            data={(name,row['name']):row for name,rows in datasets().items() for row in rows}
        rows=[dict(task=('evalplus-' if suite=='code' else 'multipl-')+name,
                   gold=key,prompt=problem['prompt'],correct=None,
                   prediction='\n# fixture-only, not a model output') for (name,key),problem in data.items()]
        path=self.root/(suite+'.jsonl');path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        identity=dict(job=suite,binding=dict(long_step=20,total_training_tokens=3932160),fixture_only=True)
        write_json(str(path)+'.meta.json',dict(identity=identity,status='complete',full_dataset_generated=True))
        receipt=dict(status='passed_source_output_audit',predictions=str(path),predictions_sha256=digest(path),meta_sha256=digest(str(path)+'.meta.json'),items=len(rows),full_dataset_generated=True,identity=identity)
        audit=self.root/(suite+'-audit.json');write_json(audit,receipt)
        return path,audit,receipt,rows

    def test_all_public_job_coverage_and_literal_solution(self):
        for suite,target in [('code',542),('multipl',1277)]:
            with self.subTest(suite=suite):
                path,audit,receipt,_=self.fixture(suite)
                with patch.object(module,'audit',return_value=receipt) as rebuilt:
                    provenance,jobs=module.validated_jobs(path,audit)
                rebuilt.assert_called_once_with(path)
                self.assertEqual(len(jobs),target)
                self.assertEqual([(r['dataset'],r['name']) for r in jobs],sorted((r['dataset'],r['name']) for r in jobs))
                self.assertTrue(all(r['solution']==r['problem']['prompt']+r['completion'] for r in jobs))
                self.assertFalse(provenance['judging_executed'])
                self.assertEqual(provenance['generation_identity']['fixture_only'],True)

    def test_missing_duplicate_prompt_or_claimed_score_refused(self):
        path,audit,receipt,rows=self.fixture('code')
        mutations=[lambda r:r.pop(),lambda r:r.append(copy.deepcopy(r[0])),
                   lambda r:r[0].update(prompt='changed'),lambda r:r[0].update(correct=False),
                   lambda r:r[0].update(prediction='changed after audit')]
        for mutation in mutations:
            bad=copy.deepcopy(rows);mutation(bad);path.write_text(''.join(json.dumps(row)+'\n' for row in bad))
            with self.subTest(mutation=mutation),patch.object(module,'audit',return_value=receipt):
                with self.assertRaises(ValueError):module.validated_jobs(path,audit)

    def test_legacy_and_partial_rejected_before_source_or_vm(self):
        path,audit,receipt,_=self.fixture('code');meta=Path(str(path)+'.meta.json')
        for identity,status,complete in [({'suite':'code'},'complete',True),
                ({'job':'code','binding':{'long_step':7}},'complete',True),
                (receipt['identity'],'time_budget',False)]:
            write_json(meta,dict(identity=identity,status=status,full_dataset_generated=complete))
            with patch.object(module,'audit') as rebuilt:
                with self.assertRaises(ValueError):module.validated_jobs(path,audit)
                rebuilt.assert_not_called()

    def test_saved_audit_mismatch_refused(self):
        path,audit,receipt,_=self.fixture('code')
        for key,value in [('items',541),('predictions','another-file'),('full_dataset_generated',False)]:
            bad=dict(receipt);bad[key]=value;write_json(audit,bad)
            with self.subTest(key=key),patch.object(module,'audit',return_value=receipt):
                with self.assertRaises(ValueError):module.validated_jobs(path,audit)

    def test_public_tests_are_part_of_plan_identity(self):
        path,audit,receipt,_=self.fixture('code')
        from eval_code_tasks import datasets
        original=list(datasets());changed=copy.deepcopy(original)
        changed[0][1][0]['fixture_only_test_change']='must change judge identity'
        with patch.object(module,'audit',return_value=receipt):
            first,_=module.validated_jobs(path,audit)
            with patch('eval_code_tasks.datasets',return_value=changed):
                second,_=module.validated_jobs(path,audit)
        self.assertNotEqual(first['problem_content_sha256'],second['problem_content_sha256'])
        self.assertEqual(first['predictions_sha256'],second['predictions_sha256'])

    def test_source_and_target_protocol_drift_rejected(self):
        path,audit,receipt,_=self.fixture('code')
        real_digest=module.digest;real_load=module.load
        with patch.object(module,'audit',return_value=receipt):
            for source in ['eval_sources.lock.json','code_images.lock.json']:
                with self.subTest(source=source),patch.object(module,'digest',side_effect=lambda name:'changed' if name==source else real_digest(name)):
                    with self.assertRaisesRegex(ValueError,'source drift'):module.validated_jobs(path,audit)
            protocol_path='profiles/qwen35_code_judging_protocol.json'
            changed=real_load(protocol_path);changed['suites']['code']['target_items']=541
            with patch.object(module,'load',side_effect=lambda name:changed if name==protocol_path else real_load(name)):
                with self.assertRaisesRegex(ValueError,'shrink'):module.validated_jobs(path,audit)


if __name__=='__main__':unittest.main()
