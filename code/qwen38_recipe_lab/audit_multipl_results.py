"""Audit isolated MultiPL-E executions; never execute candidate code on host."""
import argparse
import ast
import base64
import gzip
import hashlib
import json
import subprocess
from pathlib import Path
from eval_multipl_tasks import datasets
from judge_multipl import validated_jobs
from multipl_judge import assemble
from tokenizers import Tokenizer
from eval_multipl_repaired_tasks import datasets as repaired_datasets,completion,stopping
from audit_generation_sources import verify_generation_sources


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def verify_execution(row,problem,body,package,root,image):
    key=row['language'],row['name']
    directory=Path(row['execution_directory']).resolve();assert directory.parent==root
    execution=json.loads((directory/'execution.json').read_text())
    assert execution['status']=='exited_zero' and execution['exit_code']==0 and execution['cleanup_ok'] and not execution['oom_killed']
    assert execution['image_id']==row['image_id']==image
    assert execution['limits']==dict(wall_seconds=90,memory_mb=1024,cpu_seconds=60,pids=32,cpus=1,output_bytes=1048576,file_bytes=1048576,tmpfs_exec=True,tmpfs_bytes=67108864)
    assert digest(directory/'task.py')==execution['code_sha256']
    assignment=ast.parse((directory/'task.py').read_text()).body[0];encoded=ast.literal_eval(assignment.value)
    payload=json.loads(gzip.decompress(base64.b64decode(encoded)));assert payload['source_hashes']==package
    assert all(hashlib.sha256(payload['sources'][name].encode()).hexdigest()==sha for name,sha in package.items())
    expected=assemble(key[0],problem['prompt'],body,problem['tests'],payload['sentinel'])
    assert payload['program']==expected and row['program_sha256']==hashlib.sha256(expected.encode()).hexdigest()
    markers=[s.removeprefix('QWEN_MULTIPL_RESULT ') for s in execution['output'].splitlines() if s.startswith('QWEN_MULTIPL_RESULT ')]
    assert len(markers)==1;official=json.loads(markers[0])
    assert all(row[k]==value for k,value in official.items())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',default='runs/eval-sparse-multipl-full.jsonl')
    parser.add_argument('--checkpoint',default='runs/qwen-sparse/last.pt')
    parser.add_argument('--raw',default='runs/multipl-raw-full.jsonl')
    parser.add_argument('--scaffolds',default='runs/multipl-scaffolds-full-v2.jsonl')
    parser.add_argument('--summary',default='runs/multipl-final-audit.json')
    parser.add_argument('--repaired-predictions',default='runs/eval-sparse-multipl-repaired.jsonl')
    parser.add_argument('--repaired-scores',default='runs/multipl-repaired-score.json')
    parser.add_argument('--out',default='runs/multipl-executions-audit.json')
    parser.add_argument('--source-archive',default='c12faae')
    args=parser.parse_args()
    archive=subprocess.run(['git','rev-parse',args.source_archive],check=True,capture_output=True,text=True).stdout.strip()
    generation,data,jobs=validated_jobs(args.predictions)
    assert generation['identity']['checkpoint_sha256']==digest(args.checkpoint), 'Checkpoint identity mismatch'
    completions={(lang,name):body for lang,name,body in jobs}
    raw_meta=json.loads(Path(args.raw+'.meta.json').read_text())
    assert raw_meta['status']=='complete' and raw_meta['not_model_accuracy']
    assert digest(args.predictions)==raw_meta['identity']['predictions_sha256']
    prefix=subprocess.run(['git','rev-parse','--show-prefix'],check=True,capture_output=True,text=True).stdout.strip()
    package=json.loads(Path('multipl_judge_sources.lock.json').read_text());root=Path('runs/code-jobs').resolve()
    totals={};invalid=[];raw_rows={}
    assert raw_meta['identity']['generation_identity']==generation['identity']
    for path,scaffold in [(Path(args.raw),False),(Path(args.scaffolds),True)]:
        filename=path.stem
        meta=json.loads(Path(str(path)+'.meta.json').read_text());assert meta['completed_items']==meta['target_items']==1277
        for file,sha in meta['identity']['source_hashes'].items():
            archived=subprocess.run(['git','show',archive+':'+prefix+file],check=True,capture_output=True).stdout
            assert hashlib.sha256(archived).hexdigest()==sha
        seen=set()
        for line in path.open():
            row=json.loads(line);key=row['language'],row['name'];assert key not in seen;seen.add(key)
            body=meta['identity']['stubs'][key[0]] if scaffold else completions[key]
            verify_execution(row,data[key],body,package,root,meta['identity']['image_id'])
            if not scaffold:raw_rows[key]=row
            if scaffold and not row['scaffold_valid']:invalid.append(list(key))
        assert seen==data.keys();totals[filename]=len(seen)
    final=json.loads(Path(args.summary).read_text())
    assert sorted(invalid)==sorted(final['original_ungradable_scaffolds']) and final['original_full_model_accuracy'] is None
    for path in (args.raw,args.scaffolds,args.repaired_predictions):
        bound=[sha for file,sha in final['source_sha256'].items() if Path(file).resolve()==Path(path).resolve()]
        assert bound==[digest(path)], 'Summary bound to different prediction or execution files'
    repaired_meta=json.loads(Path(args.repaired_predictions+'.meta.json').read_text())
    assert repaired_meta['status']=='complete' and repaired_meta['full_dataset_generated']
    assert repaired_meta['identity']['checkpoint_sha256']==generation['identity']['checkpoint_sha256']
    assert repaired_meta['identity']['tokens']==generation['identity']['tokens']==1024
    verify_generation_sources(repaired_meta['identity'],archive)
    repair_data={(r['language'],r['name']):r for r in repaired_datasets()}
    assert set(repair_data)==set(map(tuple,invalid))
    tokenizer=Tokenizer.from_file('.cache/references/tokenizer.json');repair_inputs={}
    with Path(args.repaired_predictions).open() as stream:
        for line in stream:
            row=json.loads(line);key=row['task'].removeprefix('multipl-repaired-'),row['gold'];problem=repair_data[key]
            assert key not in repair_inputs and row['prompt']==problem['prompt'] and row['correct'] is None
            assert row['input_ids']==tokenizer.encode(problem['prompt'],add_special_tokens=False).ids
            assert row['output_ids'] and row['raw_prediction']==tokenizer.decode(row['output_ids'],skip_special_tokens=False)
            assert row['prediction']==completion(tokenizer.decode(row['output_ids'],skip_special_tokens=True),stopping(row['task']))
            repair_inputs[key]=row['prediction']
    repaired_rows={}
    for row in json.loads(Path(args.repaired_scores).read_text()):
        key=row['language'],row['name'];assert key not in repaired_rows
        verify_execution(row,repair_data[key],repair_inputs[key],package,root,raw_meta['identity']['image_id'])
        repaired_rows[key]=row
    assert repaired_rows.keys()==repair_inputs.keys()==repair_data.keys()
    counts={}
    for key,row in raw_rows.items():
        chosen=repaired_rows.get(key,row);count=counts.setdefault(key[0],dict(n=0,correct=0,repaired_fixtures=0))
        count['n']+=1;count['correct']+=int(chosen['passed']);count['repaired_fixtures']+=int(key in repaired_rows)
    assert counts==final['tasks'] and len(counts)==8
    macro=sum(c['correct']/c['n'] for c in counts.values())/8
    assert macro==final['mac_repaired_eight_language_macro']
    assert final['original_raw_tested']==final['repaired_composite_items']==len(raw_rows)==1277
    assert final['original_raw_passes']==sum(row['passed'] for row in raw_rows.values())
    assert final['original_valid_scaffolds']==1277-len(invalid)
    assert final['original_valid_correct']==sum(row['passed'] for key,row in raw_rows.items() if key not in repaired_rows)
    for file,sha in final['source_sha256'].items():
        relative=Path(file)
        if relative.is_absolute():relative=relative.relative_to(Path.cwd())
        if relative.parts[0]=='runs':assert digest(file)==sha
        else:
            archived=subprocess.run(['git','show',archive+':'+prefix+relative.as_posix()],check=True,capture_output=True).stdout
            assert hashlib.sha256(archived).hexdigest()==sha
    artifacts = dict(predictions=args.predictions, raw=args.raw, scaffolds=args.scaffolds,
                     summary=args.summary, repaired_predictions=args.repaired_predictions,
                     repaired_scores=args.repaired_scores)
    for name in ('predictions', 'raw', 'scaffolds', 'repaired_predictions'):
        artifacts[f'{name}_meta'] = artifacts[name] + '.meta.json'
    report=dict(auditor_sha256=digest(__file__),all_program_inputs_generated_ids_official_returns_images_limits_driver_bytes_and_cleanups_verified=True,
        execution_records=totals,invalid_original_fixtures=invalid,original_full_model_accuracy=None,
        mac_repaired_macro=macro,repaired_execution_records_verified=len(repaired_rows),all_eight_language_counts_recomputed=True,not_official_protocol=True,source_archive=archive,generation_checkpoint_sha256=generation['identity']['checkpoint_sha256'],
        predictions_path=args.predictions,raw_path=args.raw,scaffolds_path=args.scaffolds,summary_path=args.summary,
        artifacts={name: dict(path=path, sha256=digest(path)) for name, path in artifacts.items()})
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
