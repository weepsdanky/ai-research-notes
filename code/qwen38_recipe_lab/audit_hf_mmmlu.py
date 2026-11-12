"""Require the original fourteen-language HF target, preserving historical engines."""
import argparse
import json
from pathlib import Path
from audit_core_results import digest, now
from audit_generation_sources import verify_generation_sources

CHECKPOINT='runs/qwen-sparse/last.pt'
CHECKPOINT_SHA='996f1abf6540637b63da742077db4bd7d49f2f13bbe247f8aca2710c3a072cb8'
ARCHIVES=('e0c82b7','af011ff','15e8acc')


def matching_archive(identity):
    for revision in ARCHIVES:
        try:
            sources=verify_generation_sources(identity,revision)
            return revision,sources
        except (AssertionError,StopIteration):
            continue
    raise ValueError('No immutable archive matches recorded generation sources')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    plan=json.loads(Path('profiles/mmmlu_eval.json').read_text())
    languages=plan['languages_in_order']
    assert len(languages)==len(set(languages))==14
    assert plan['items_per_language']==14042 and plan['total_items']==196588
    audits={}
    for path in Path('runs').glob('mmmlu-*-completed-audit.json'):
        document=json.loads(path.read_text())
        for file,result in document['files'].items():
            if file in audits:
                assert audits[file][1]==result,'Ambiguous complete audit for '+file
            audits[file]=(path,result,document['scorer_source_sha256'])
    for lang in languages:
        file=f'runs/eval-sparse-mmmlu-{lang}-full.jsonl'
        if file not in audits:
            raise ValueError('Full original HF collection unavailable; missing complete audit: '+file)
    assert digest(CHECKPOINT)==CHECKPOINT_SHA
    entries=[]
    for lang in languages:
        name='mmmlu-'+lang;file=f'runs/eval-sparse-{name}-full.jsonl'
        path,result,scorer=audits[file]
        meta=json.loads(Path(file+'.meta.json').read_text());identity=meta['identity']
        assert identity['checkpoint_sha256']==CHECKPOINT_SHA and meta['status']=='complete'
        assert identity['scope']=='full selected datasets' and identity['suite']=='mmmlu'
        assert (identity['tokens'],identity['context'],identity['batch_size'],identity['prefill_chunk'])==(64,8192,8,128)
        assert identity['gdn_scan']=='chunk' and identity['grouped_gqa']
        assert identity['stop_strings']=={name:[]}
        assert meta['full_dataset_generated'] and result['full_dataset_generated']
        assert result['all_source_prompts_gold_input_ids_raw_decoding_and_scores_verified']
        assert set(meta['tasks'])==set(result['tasks'])=={name}
        task=meta['tasks'][name]
        assert task['n']==task['scored']==task['total_dataset']==result['items']==result['tasks'][name]['n']==14042
        assert task['capacity_failures']==0 and task['correct']==result['correct']==result['tasks'][name]['correct']
        assert digest(file)==result['file_sha256']
        assert scorer==next(value for key,value in identity['source_hashes'].items() if key.endswith('eval_mmmlu_tasks.py'))
        revision,sources=matching_archive(identity)
        entries.append(dict(language=lang,items=14042,correct=task['correct'],file=file,file_sha256=result['file_sha256'],
                            audit=str(path),audit_sha256=digest(path),meta_sha256=digest(file+'.meta.json'),
                            source_archive=revision,generation_sources=sources,scorer_source_sha256=scorer,
                            prefix_cache_groups=len(identity['prefix_cache']['plans'])))
    items=sum(row['items'] for row in entries);correct=sum(row['correct'] for row in entries)
    assert items==196588
    result=dict(at=now(),checkpoint_sha256=CHECKPOINT_SHA,auditor_sha256=digest(__file__),full_languages=14,
                full_dataset_generated=True,items=items,correct=correct,micro_accuracy=correct/items,
                language_macro_accuracy=sum(row['correct']/14042 for row in entries)/14,
                every_file_bound_to_its_complete_prompt_ids_raw_score_audit=True,languages=entries,
                protocol='Original HF score variant; Mac same-subject English dev5, greedy64/context8192. Historical cache source versions retained per language; no paper checkpoint predictions.',
                historical_equivalence_evidence=['runs/mmmlu-prefix-trained-proof.json','runs/mmmlu-rows-before-groups.json','runs/mmmlu-rows-after-groups.json'])
    Path(args.out).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='languages'},indent=2))


if __name__=='__main__':main()
