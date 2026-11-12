"""Stream full public benchmark MTP acceptance traces and audit their coverage."""
import argparse
from datetime import datetime, timedelta
import fcntl
import json
import math
import os
from pathlib import Path
import re
import resource

from evaluate_mtp_draft import load_models, score_item
from mtp_public_data import PROFILE, profile_and_tokenizer, public_items
from qwen35_quality_data import digest, load, write_json
from run_dense_text_guarded import native, pressure


def stats_for_trace(result, output, prompt_length, profile):
    assert result['output_ids']==output and result['stop']==('eos' if output[-1]==profile['eos'] else 'max_new_tokens')
    assert math.isfinite(result['seconds']) and result['seconds']>=0
    offset = 1
    rounds = accepted = clipped = draft_tokens = 0
    target_calls = target_forward_calls = 1
    for row in result['rounds']:
        if row['kind']=='target_only_boundary':
            assert row['emitted']==1 and offset+1==len(output)==profile['new_tokens']
            offset += 1
            target_calls += 1
            target_forward_calls += 1
            continue
        proposals, targets = row['proposals'], row['target_tokens']
        drafted = len(proposals)
        assert 1<=drafted==row['drafted']<=profile['draft_tokens']
        assert len(targets)==drafted+1 and drafted<=profile['new_tokens']-offset-1
        assert all(type(x) is int and 0<=x<profile['vocab_size'] for x in (*proposals,*targets))
        matched = next((i for i,(draft,target) in enumerate(zip(proposals,targets)) if draft!=target),drafted)
        if profile['eos'] in proposals[:matched]:
            kept = proposals.index(profile['eos'])+1
            emitted = proposals[:kept]
            kind = 'accepted_eos'
        else:
            kept = matched
            emitted = targets[:kept+1]
            kind = 'bonus' if matched==drafted else 'correction'
        assert row['kind']==kind and row['accepted']==kept and row['emitted']==len(emitted)
        assert row['target_consumed']==kept+1 and row['target_replayed']==(kept<drafted)
        assert row['seed_position']==prompt_length+offset-1
        assert row['mtp_keep']==row['seed_position']
        assert row['mtp_cache_before_rollback']>=row['mtp_keep']
        assert 0<row['mtp_selected_count']<=row['seed_position']
        assert output[offset:offset+len(emitted)]==emitted
        offset += len(emitted)
        rounds += 1
        accepted += kept
        clipped += row['mtp_selected_count'] < row['seed_position']
        draft_tokens += drafted
        target_calls += 1+row['target_replayed']
        target_forward_calls += 1 + drafted + row['target_replayed'] * row['target_consumed']
    assert offset==len(output)
    assert (result['num_drafts'],result['num_accepted_tokens'],result['num_draft_tokens'])==(rounds,accepted,draft_tokens)
    assert result['target_calls']==target_calls and result['draft_calls']==draft_tokens
    assert result['mean_acceptance_length_vllm']==(1+accepted/rounds if rounds else None)
    counts = dict(rounds=rounds,accepted=accepted,clipped_rounds=clipped,draft_tokens=draft_tokens,
                  target_calls=target_calls,seconds=result['seconds'])
    if 'target_forward_calls' in result:
        assert result['target_forward_calls'] == target_forward_calls
        counts['target_forward_calls'] = target_forward_calls
    return counts


def empty_totals(verification="batched"):
    totals = dict(items=0, teacher_seconds=0.0, peak_rss_bytes=0, swap_used_mib=None,
                modes={mode:dict(rounds=0,accepted=0,clipped_rounds=0,draft_tokens=0,
                                target_calls=0,seconds=0.0) for mode in ('initial','trained')})
    if verification == 'serial':
        for mode in totals['modes'].values():
            mode['target_forward_calls'] = 0
    return totals


def add(totals, row, item, tokenizer, profile):
    assert all(row[key]==value for key,value in item.items())
    output = row['output_ids']
    assert 0<len(output)<=profile['new_tokens'] and all(type(x) is int and 0<=x<profile['vocab_size'] for x in output)
    assert profile['eos'] not in output[:-1]
    assert output[-1]==profile['eos'] or len(output)==profile['new_tokens']
    assert row['text']==tokenizer.decode(output)
    assert row['pressure']==1 and math.isfinite(row['teacher_seconds']) and row['teacher_seconds']>=0
    assert set(row['speculative'])==set(totals['modes'])
    for mode,result in row['speculative'].items():
        counts = stats_for_trace(result,output,len(item['input_ids']),profile)
        assert counts.keys() == totals['modes'][mode].keys()
        for key,value in counts.items():
            totals['modes'][mode][key] += value
    totals['items'] += 1
    totals['teacher_seconds'] += row['teacher_seconds']
    totals['peak_rss_bytes'] = max(totals['peak_rss_bytes'],row['process_peak_rss_bytes'])
    swap = float(re.search(r'used = ([\d.]+)M', row['swap'])[1])
    prior = totals['swap_used_mib']
    totals['swap_used_mib'] = [swap,swap] if prior is None else [min(swap,prior[0]),max(swap,prior[1])]


def evaluate(args):
    profile, tokenizer = profile_and_tokenizer()
    items, prompt_sha = public_items(profile,args.job,tokenizer)
    training = load(profile['draft_training_profile'])
    identity = dict(job=args.job,verification=args.verification,profile_sha256=digest(PROFILE),prompt_content_sha256=prompt_sha,
                    draft_checkpoint_sha256=digest(args.checkpoint),target_checkpoint_sha256=digest(training['target']),
                    sources={name:digest(name) for name in ('evaluate_mtp_public.py','mtp_public_data.py',
                            'evaluate_mtp_draft.py','inference.py','speculative_mtp.py','mtp_inference.py',
                            'model.py','mtp.py')})
    out = args.out
    meta = Path(str(out)+'.meta.json')
    if args.verification == 'serial':
        identity['sources']['serial_mtp_verification.py'] = digest('serial_mtp_verification.py')
    totals = empty_totals(args.verification)
    if args.resume or args.audit_out:
        previous = load(meta)
        assert previous['identity']==identity
        with out.open() as stream:
            for line in stream:
                item = items[totals['items']]
                add(totals,json.loads(line),item,tokenizer,profile)
        if args.audit_out:
            complete = totals['items']==len(items)
            assert complete==(previous['status']=='complete') and totals==previous['totals']
            acceptance = {mode:1+r['accepted']/r['rounds'] if r['rounds'] else None
                          for mode,r in totals['modes'].items()}
            write_json(args.audit_out,dict(identity=identity,complete=complete,target=len(items),totals=totals,
                                          acceptance=acceptance,output_sha256=digest(out)))
            return
        if previous['status']=='running':
            # The model lock proves the old writer exited. Raw rows are fsynced before metadata.
            assert totals['items']>=previous['totals']['items']
        else:
            assert previous['status'] in ('memory_pressure','time_budget') and totals==previous['totals']
        if totals['items']==len(items):
            write_json(meta,dict(identity=identity,status='complete',target=len(items),totals=totals))
            return
    else:
        assert not out.exists() and not meta.exists()
    decoder_class = None
    if args.verification == 'serial':
        from serial_mtp_verification import SerialTargetMTP
        decoder_class = SerialTargetMTP
    models = load_models(args.checkpoint)
    assert models[-1]==training and (models[3].eos,models[3].vocab_size)==(profile['eos'],profile['vocab_size'])
    stop = args.deadline-timedelta(seconds=600)
    call_stop = datetime.now().astimezone()+timedelta(seconds=args.seconds)
    status = 'complete'
    with out.open('a' if args.resume else 'x') as stream:
        write_json(meta,dict(identity=identity,status='running',target=len(items),totals=totals))
        for item in items[totals['items']:]:
            if pressure()!=1:
                status='memory_pressure'
                break
            if datetime.now().astimezone()>=min(stop,call_stop):
                status='time_budget'
                break
            teacher, seconds, results = score_item(models, item, profile, True, decoder_class=decoder_class)
            row = dict(item,output_ids=teacher,text=tokenizer.decode(teacher),teacher_seconds=seconds,
                       speculative=results,pressure=1,swap=native(['sysctl','vm.swapusage']).strip(),
                       process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            add(totals,row,item,tokenizer,profile)
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
            write_json(meta,dict(identity=identity,status='running',target=len(items),totals=totals))
    assert (totals['items']==len(items))==(status=='complete')
    write_json(meta,dict(identity=identity,status=status,target=len(items),totals=totals))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True, choices=load(PROFILE)['jobs'])
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--deadline', type=datetime.fromisoformat)
    parser.add_argument('--seconds', type=int, default=900)
    parser.add_argument('--verification', choices=('batched', 'serial'), default='batched')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--resume', action='store_true')
    mode.add_argument('--audit-out', type=Path)
    args = parser.parse_args()
    runs = Path('runs').resolve()
    assert args.out.parent.resolve()==runs
    if args.audit_out:
        assert args.audit_out.parent.resolve()==runs and not args.audit_out.exists()
    else:
        now=datetime.now().astimezone()
        assert args.deadline and now+timedelta(seconds=660)<args.deadline<=now+timedelta(hours=2)
        assert 30<=args.seconds<=1100
    assert not any(key in os.environ for key in ('PYTORCH_MPS_HIGH_WATERMARK_RATIO','PYTORCH_ENABLE_MPS_FALLBACK'))
    with (runs/'hf-mmmlu-global.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert pressure()==1
        if not args.audit_out:
            free = int(re.search(r'free percentage:\s*(\d+)%',native(['memory_pressure','-Q']))[1])
            assert free>=55
        evaluate(args)


if __name__=='__main__':
    main()
