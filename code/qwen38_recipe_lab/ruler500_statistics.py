"""Streaming statistics for an independently audited, stopped RULER500 snapshot."""
import argparse,hashlib,json,math,statistics
from datetime import datetime
from pathlib import Path


FIELDS=('source_id','task','kind','context','tokens','author_row','prompt','input_ids','output_ids','raw','response','score','stop','unused_token_ids')


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def question_digest(row):
    return hashlib.sha256(json.dumps({key:row[key] for key in FIELDS},sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def snapshot(filename,entry,old_audits):
    path=Path(filename);meta_path=Path(filename+'.meta.json');meta=json.loads(meta_path.read_text())
    assert meta['status']!='running','Model is still writing; wait for a stopped audited snapshot'
    assert entry['meta_sha256']==digest(meta_path),'Audit does not bind the current metadata'
    stage=meta['identity']['stage'];baseline={};baseline_path=Path(f'runs/ruler-paper-{stage}.jsonl')
    baseline_audit=Path('runs/ruler-paper-completed-audit.json');old16=json.loads(baseline_audit.read_text())['files'][str(baseline_path)]
    assert digest(baseline_path)==old16['sha256'] and digest(str(baseline_path)+'.meta.json')==old16['meta_sha256']
    baseline_meta=json.loads(Path(str(baseline_path)+'.meta.json').read_text())
    assert baseline_meta['identity']['checkpoint_sha256']==meta['identity']['checkpoint_sha256'] and baseline_meta['identity']['stage']==stage
    with baseline_path.open() as stream:
        for line in stream:
            row=json.loads(line);baseline[row['source_id']]=question_digest(row)
    prefixes={}
    for audit_path in old_audits:
        previous=json.loads(Path(audit_path).read_text())
        assert previous['auditor_sha256']==digest('audit_ruler_public500.py')
        assert previous['prepared_meta_sha256']==meta['identity']['prepared_meta_sha256']
        old=previous.get('files',{}).get(filename)
        if old:prefixes[str(audit_path)]=old
    checks={};h=hashlib.sha256();size=0;counts={};matched=[];seconds=[];prefill=[];tokens=0;examples=[]
    for audit_path,old in prefixes.items():
        if old['items']==0:
            assert old['sha256']==h.hexdigest()
            checks[audit_path]=dict(items=0,bytes=0,sha256=h.hexdigest(),audit_sha256=digest(audit_path))
    with path.open('rb') as stream:
        for number,line in enumerate(stream,1):
            h.update(line);size+=len(line);row=json.loads(line)
            for audit_path,old in prefixes.items():
                if number==old['items']:
                    assert h.hexdigest()==old['sha256'],'Previously audited prefix changed'
                    checks[audit_path]=dict(items=number,bytes=size,sha256=h.hexdigest(),audit_sha256=digest(audit_path))
            key=f"{row['context']}/{row['task']}";counts[key]=counts.get(key,0)+1
            if row['source_id'] in baseline:
                assert question_digest(row)==baseline[row['source_id']],'Original16-case question/output changed'
                matched.append(row['source_id'])
            assert math.isfinite(row['seconds']) and row['seconds']>0 and 0<=row['prefill_seconds']<=row['seconds']
            seconds.append(row['seconds']);prefill.append(row['prefill_seconds']);tokens+=len(row['output_ids'])
            if counts[key]==1 and len(examples)<3:examples.append(dict(source_id=row['source_id'],raw=row['raw'],output_ids=row['output_ids'],score=row['score']))
    assert set(checks)==set(prefixes) and len(seconds)==entry['items']==meta['completed_items']
    assert h.hexdigest()==entry['sha256']==meta['predictions_sha256'] and digest(meta_path)==entry['meta_sha256']
    assert entry['complete_mac_cohort']==(len(seconds)==13000) and entry['author_summary']==meta['author_summary']
    return dict(stage=stage,status=meta['status'],items=len(seconds),target=13000,bytes=size,sha256=h.hexdigest(),meta_sha256=entry['meta_sha256'],
                checkpoint_sha256=meta['identity']['checkpoint_sha256'],device=meta['identity']['device'],threads=meta['identity']['threads'],
                counts=counts,output_tokens=tokens,row_seconds_p50=statistics.median(seconds) if seconds else None,prefill_seconds_p50=statistics.median(prefill) if prefill else None,
                row_seconds_sum=sum(seconds),output_tokens_per_row_second=tokens/sum(seconds) if seconds else None,matched_old16_questions=matched,
                original16_fields_checked=FIELDS,original16_audit_sha256=digest(baseline_audit),preserved_audited_prefixes=checks,author_summary=entry['author_summary'],
                complete_mac_cohort=entry['complete_mac_cohort'],last_call_peak_rss_bytes=meta['process_peak_rss_bytes'],real_examples=examples)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',required=True);p.add_argument('--old-audit',action='append',default=[]);p.add_argument('--out',required=True);p.add_argument('--deadline',required=True);a=p.parse_args()
    deadline=datetime.fromisoformat(a.deadline);assert deadline.tzinfo is not None and datetime.now().astimezone()<deadline
    target=Path(a.out);assert not target.exists();audit=json.loads(Path(a.audit).read_text());assert audit['auditor_sha256']==digest('audit_ruler_public500.py')
    assert audit['all13000_source_token_budgets_and_first16_byte_prefixes_verified']
    assert audit['prepared_meta_sha256']==digest('data/ruler-public500-repair/meta.json')
    result=dict(at=datetime.now().astimezone().isoformat(),source_sha256=digest(__file__),audit_file=a.audit,audit_sha256=digest(a.audit),
                variant=audit['variant'],full_paper_protocol=audit['full_paper_protocol'],prepared_meta_sha256=audit['prepared_meta_sha256'],files={})
    for filename,entry in audit['files'].items():
        result['files'][filename]=snapshot(filename,entry,a.old_audit)
        assert datetime.now().astimezone()<deadline
    result['scope']='Per-stage frozen snapshot; row timing includes prefill/decode/scoring, excludes loading/resume checks; not a kernel speedup'
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(target)


if __name__=='__main__':main()
