"""Audit completed EvalPlus identities, isolated executions and aggregate scores."""
import argparse
import hashlib
import json
import subprocess
from collections import Counter,defaultdict
from datetime import datetime
from pathlib import Path
from tokenizers import Tokenizer
from eval_code_tasks import datasets,stopping,completion
from audit_generation_sources import verify_generation_sources


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',default='runs/eval-sparse-code-full.jsonl')
    parser.add_argument('--judged',default='runs/evalplus-score-full-v4.jsonl')
    parser.add_argument('--out',default='runs/evalplus-completed-audit.json')
    parser.add_argument('--source-archive',default='9de3240')
    parser.add_argument('--checkpoint',default='runs/qwen-sparse/last.pt')
    args=parser.parse_args()
    predictions=Path(args.predictions);judged=Path(args.judged)
    archive=subprocess.run(['git','rev-parse',args.source_archive],capture_output=True,text=True,check=True).stdout.strip()
    generation=json.loads(Path(str(predictions)+'.meta.json').read_text());meta=json.loads(Path(str(judged)+'.meta.json').read_text())
    assert generation['identity']['checkpoint_sha256']==digest(args.checkpoint), 'Checkpoint identity mismatch'
    assert generation['status']==meta['status']=='complete' and generation['full_dataset_generated']
    assert meta['identity']['predictions_sha256']==digest(predictions) and meta['identity']['generation_identity']==generation['identity']
    generation_sources=verify_generation_sources(generation['identity'],archive)
    sources={};prefix=subprocess.run(['git','rev-parse','--show-prefix'],capture_output=True,text=True,check=True).stdout.strip()
    for file,sha in meta['identity']['source_hashes'].items():
        data=subprocess.run(['git','show',archive+':'+prefix+file],capture_output=True,check=True).stdout
        assert hashlib.sha256(data).hexdigest()==sha;sources[file]=sha
    data={(dataset,r['task_id']):r for dataset,rows in datasets() for r in rows}
    tokenizer=Tokenizer.from_file('.cache/references/tokenizer.json');inputs={}
    for line in predictions.open():
        row=json.loads(line);key=row['task'].removeprefix('evalplus-'),row['gold'];problem=data[key]
        assert key not in inputs and row['prompt']==problem['prompt'] and row['correct'] is None
        assert row['input_ids']==tokenizer.encode(problem['prompt'],add_special_tokens=False).ids
        assert row['output_ids'] and row['raw_prediction']==tokenizer.decode(row['output_ids'],skip_special_tokens=False)
        assert row['prediction']==completion(tokenizer.decode(row['output_ids'],skip_special_tokens=True),stopping(row['task']))
        inputs[key]=problem['prompt']+row['prediction']
    assert inputs.keys()==data.keys()
    package=json.loads(Path('evalplus_source.lock.json').read_text())['package_sources']
    scores=defaultdict(lambda:dict(n=0,base_correct=0,plus_correct=0));seen=set();peak=0;zero_tests=Counter()
    root=Path('runs/code-jobs').resolve()
    for line in judged.open():
        row=json.loads(line);key=row['dataset'],row['task_id'];assert key not in seen;seen.add(key)
        assert row['raw_solution_sha256']==hashlib.sha256(inputs[key].encode()).hexdigest()
        assert row['solution_sha256']==hashlib.sha256(row['sanitized_solution'].encode()).hexdigest()
        directory=Path(row['execution_directory']).resolve();assert directory.parent==root
        execution=json.loads((directory/'execution.json').read_text())
        assert execution['status']=='exited_zero' and execution['exit_code']==0 and execution['cleanup_ok'] and not execution['oom_killed']
        assert execution['limits']==dict(wall_seconds=120,memory_mb=2048,cpu_seconds=90,pids=32,cpus=1,output_bytes=1048576,file_bytes=1048576)
        assert execution['image_id']==row['image_id']==meta['identity']['image_id']
        assert digest(directory/'task.py')==execution['code_sha256']==row['driver_sha256']
        markers=[s.removeprefix('QWEN_EVALPLUS_RESULT ') for s in execution['output'].splitlines() if s.startswith('QWEN_EVALPLUS_RESULT ')]
        assert len(markers)==1;official=json.loads(markers[0]);assert official['package_sources']==package and official['fast_check'] is True
        assert official['dataset']==key[0] and len(official['results'])==1
        for field,value in official['results'][0].items():assert row[field]==value
        assert row['base_pass']==row['base']['passed'] and row['plus_pass']==(row['base']['passed'] and row['plus']['passed'])
        assert all(row[suite]['total_inputs']==len(data[key][suite+'_input']) for suite in ('base','plus'))
        peak=max(peak,official['oracle_process_peak_rss_bytes'])
        for suite in ('base','plus'):zero_tests[suite]+=int(row[suite]['tested_inputs']==0)
        count=scores[key[0]];count['n']+=1;count['base_correct']+=row['base_pass'];count['plus_correct']+=row['plus_pass']
    assert seen==inputs.keys() and dict(scores)==meta['tasks']
    mean=sum(c[k]/c['n'] for c in scores.values() for k in ('base_correct','plus_correct'))/4
    assert mean==meta['four_suite_mean']
    report=dict(at=datetime.now().astimezone().isoformat(),judged_sha256=digest(judged),predictions_sha256=digest(predictions),
        judged_items=len(seen),scores=dict(scores),four_suite_mean=mean,judge_source_archive=archive,source_hashes=sources,
        predictions_path=str(predictions),judged_path=str(judged),verified_generation_sources=generation_sources,generation_checkpoint_sha256=generation['identity']['checkpoint_sha256'],
        all_prompts_ids_solutions_driver_bytes_official_results_images_and_cleanups_verified=True,
        oracle_max_process_peak_rss_bytes=peak,items_with_zero_assertion_inputs_executed=dict(zero_tests),
        note='All tasks officially judged; fast-check can fail invalid/missing-function solutions before assertions. This is not a claim that every assertion ran on each model answer. Full expanded input/reference sets were retained.')
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
