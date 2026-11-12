"""Serial full public500 Mac modes; resume resource stops only after bound review evidence."""
import argparse,hashlib,json,os,subprocess,sys
from datetime import datetime,timedelta
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def verified_time_stop(meta,guard):
    if not guard or guard.get('status')!='terminal' or guard.get('returncode')!=0 or meta.get('status')!='interrupted_signal':return False
    signals=guard.get('signals',[])
    return bool(signals) and all(s.get('signal')=='SIGINT' and s.get('reason') in ('wall_budget','deadline') and s.get('pressure')==1 for s in signals)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--deadline',required=True);p.add_argument('--prefix',required=True);p.add_argument('--review-stop');p.add_argument('--time-stop-state');a=p.parse_args();deadline=datetime.fromisoformat(a.deadline);started=datetime.now().astimezone()
    assert deadline.tzinfo is not None and started+timedelta(seconds=390)<deadline<=started+timedelta(hours=2)
    prefix=Path(a.prefix);assert prefix.parent.resolve()==Path('runs').resolve();state_path=Path(str(prefix)+'-state.json');assert not state_path.exists(),'Preserve prior driver evidence; choose new prefix'
    state=dict(started=started.isoformat(),deadline=deadline.isoformat(),pid=os.getpid(),source_sha256=digest(__file__),status='preflight',calls=[])
    def save():
        temporary=state_path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(state_path)
    save();prepared=Path('data/ruler-public500-repair/meta.json');data=json.loads(prepared.read_text());assert data['status']=='complete_frozen_mac_cohort' and data['validated_rows']==13000
    audit=Path('runs/ruler500-repair-cohort-audit.json');source=json.loads(audit.read_text());assert source['auditor_sha256']==digest('audit_ruler_public500.py') and source['prepared_meta_sha256']==digest(prepared) and source['all13000_source_token_budgets_and_first16_byte_prefixes_verified']
    review=json.loads(Path(a.review_stop).read_text()) if a.review_stop else None;review_used=False;number=0;last_guard=json.loads(Path(a.time_stop_state).read_text()) if a.time_stop_state else None
    for stage in ('sparse','dense'):
        while True:
            meta_path=Path(f'runs/ruler500-paper-{stage}.jsonl.meta.json');meta=json.loads(meta_path.read_text()) if meta_path.exists() else None
            if meta and meta['status']=='complete':break
            if meta and meta['status']!='time_budget' and verified_time_stop(meta,last_guard):
                command=last_guard['command'];assert command[command.index('--out')+1]==str(meta_path).removesuffix('.meta.json')
                assert command[command.index('--stage')+1]==stage
                assert last_guard['source_sha256']==digest('run_long_context_guarded.py')
                assert datetime.fromisoformat(last_guard['started'])<=datetime.fromisoformat(meta['events'][-1]['end'])<=datetime.fromisoformat(last_guard['ended'])
                state['admitted_time_stop']=dict(prefix=last_guard.get('command'),native_status=meta['status']);save()
            elif meta and meta['status']!='time_budget':
                if review_used or not review or meta['status'] not in ('memory_pressure','interrupted_signal'):
                    state.update(status='review_non_time_stop',stage=stage,model_status=meta['status']);save();return
                assert review['meta_sha256']==digest(meta_path) and review['predictions_sha256']==meta['predictions_sha256']
                checked=json.loads(Path(review['audit_file']).read_text());entry=checked['files'][str(meta_path).removesuffix('.meta.json')]
                assert digest(review['audit_file'])==review['audit_sha256'] and entry['meta_sha256']==digest(meta_path) and entry['sha256']==meta['predictions_sha256']
                samples=json.loads(Path(review['recovery_samples']).read_text());assert len(samples)>=12 and all(s['pressure']==1 for s in samples) and review['no_active_prior_model_confirmed']
                review_used=True;state['admitted_review_sha256']=digest(a.review_stop);save()
            if meta:assert digest(str(meta_path).removesuffix('.meta.json'))==meta['predictions_sha256'],'Saved prefix bytes changed'
            budget=min(1180,int((deadline-datetime.now().astimezone()).total_seconds())-360)
            if budget<30:state.update(status='cleanup_reserve',ended=datetime.now().astimezone().isoformat());save();return
            number+=1;call_prefix=f'{prefix}-{stage}-{number:03d}'
            command=[sys.executable,'run_long_context_guarded.py','--suite','ruler500','--stage',stage,'--deadline',deadline.isoformat(),'--prefix',call_prefix,'--seconds',str(budget)]+(['--resume'] if meta else [])
            item=dict(prefix=call_prefix,stage=stage,starting_items=meta['completed_items'] if meta else 0,budget_seconds=budget,started=datetime.now().astimezone().isoformat());state['calls'].append(item);state['status']='running';save();child=subprocess.Popen(command);item['guard_pid']=child.pid;save();code=child.wait()
            guard=json.loads(Path(call_prefix+'-state.json').read_text());item.update(returncode=code,guard_status=guard['status'],ended=datetime.now().astimezone().isoformat())
            current=None
            if meta_path.exists():
                current=json.loads(meta_path.read_text());item.update(ending_items=current['completed_items'],model_status=current['status'])
            save()
            last_guard=guard
            time_stop=verified_time_stop(current or {},guard)
            item['verified_time_stop']=time_stop;save()
            if code or guard['status']!='terminal' or guard['returncode']!=0 or (guard['signals'] and not time_stop):
                state['status']='guard_or_signal_stop_requires_review';save();return
            if current['status'] not in ('time_budget','complete') and not time_stop:state['status']='model_stop_requires_review';save();return
    state.update(status='both_mac_modes_generated_needs_independent_audit',ended=datetime.now().astimezone().isoformat());save()


if __name__=='__main__':main()
