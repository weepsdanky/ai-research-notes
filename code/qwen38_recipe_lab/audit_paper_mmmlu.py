"""Bind all fourteen paper MMMLU audits before reporting a collection mean."""
import argparse
import json
from pathlib import Path
from audit_generation_sources import verify_generation_sources
from audit_core_results import digest, now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    profile = json.loads(Path('profiles/mmmlu_eval.json').read_text())
    paper = json.loads(Path('profiles/paper_evaluation.json').read_text())
    languages = profile['languages_in_order']
    assert len(languages) == len(set(languages)) == 14
    assert profile['items_per_language'] == 14042 and profile['total_items'] == 196588
    entries = []
    # Require every complete audit first. Never emit a subset mean or partial output.
    audits = {lang: Path(f'runs/paper-mmmlu-{lang}-completed-audit.json') for lang in languages}
    for path in audits.values():
        if not path.exists():
            raise ValueError('Full collection unavailable; missing audit: '+str(path))
    for lang in languages:
        name = 'mmmlu-'+lang
        output = f'runs/eval-paper-{name}-full.jsonl'
        audit = json.loads(audits[lang].read_text())
        result = audit['files'][output]
        meta = json.loads(Path(output+'.meta.json').read_text())
        identity = meta['identity']
        assert identity['checkpoint_sha256'] == paper['checkpoint_sha256']
        assert identity['tokens'] == 64 and identity['context'] == 8192 and identity['scope'] == 'full selected datasets'
        assert meta['status'] == result['terminal_status'] == 'complete'
        assert meta['full_dataset_generated'] and result['full_dataset_generated']
        assert result['all_source_prompts_gold_input_ids_raw_decoding_and_scores_verified']
        assert set(meta['tasks']) == set(result['tasks']) == {name}
        task = meta['tasks'][name]
        assert task['n'] == task['scored'] == task['total_dataset'] == result['items'] == result['target_items'] == 14042
        assert task['capacity_failures'] == 0
        assert task['correct'] == result['correct'] == result['tasks'][name]['correct']
        assert result['tasks'][name]['n'] == 14042
        assert digest(output) == result['file_sha256']
        assert digest('eval_mmmlu_tasks.py') == audit['scorer_source_sha256']
        sources = verify_generation_sources(identity, '15e8acc')
        assert paper['engine_sha256'] in sources.values()
        entries.append(dict(language=lang, items=task['n'], correct=task['correct'],
                            file=output, file_sha256=result['file_sha256'], audit=str(audits[lang]),
                            audit_sha256=digest(audits[lang]), meta_sha256=digest(output+'.meta.json')))
    total = sum(row['items'] for row in entries)
    correct = sum(row['correct'] for row in entries)
    assert total == profile['total_items']
    result = dict(at=now(), checkpoint_sha256=paper['checkpoint_sha256'], source_archive='15e8acc',
                  auditor_sha256=digest(__file__), full_languages=14, full_dataset_generated=True,
                  items=total, correct=correct, micro_accuracy=correct/total,
                  language_macro_accuracy=sum(row['correct']/row['items'] for row in entries)/14,
                  every_file_bound_to_its_complete_prompt_ids_raw_score_audit=True, languages=entries,
                  protocol='Mac 5-shot same-subject English dev, no CoT, greedy64/context8192; private Qwen harness unknown; HF checkpoint results excluded.')
    Path(args.out).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'languages'}, indent=2))


if __name__ == '__main__':
    main()
