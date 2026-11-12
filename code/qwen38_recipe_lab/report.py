"""Refresh the factual progress report from local metrics, without changing budgets."""
import hashlib,json, statistics,re
from datetime import datetime
from pathlib import Path


def latest_ruler_snapshot():
    """Render a bound historical receipt; never infer completeness from progress."""
    candidates=[]
    for path in Path('runs').glob('round*-final.json'):
        match=re.fullmatch(r'round(\d+)-final.json',path.name)
        if match and int(match[1])>=25:
            final=json.loads(path.read_text())
            if 'ruler500_statistics' in final:candidates.append((int(match[1]),path,final))
    if not candidates:return '', ''
    number,path,final=max(candidates,key=lambda item:item[0]);record=final['ruler500_statistics']
    def sha(name):
        with Path(name).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    assert sha(record['file'])==record['sha256']
    stats=json.loads(Path(record['file']).read_text());audit=final['public500_final_prefix_audit']
    assert stats['audit_sha256']==audit['sha256']==sha(audit['file'])
    text=f"\n## 第{number}轮RULER500已审计快照（{final['confirmed_at']}）\n\n预算{final['round_started']}至{final['deadline']}。这是13配置×4K/8K×500的Mac QA retry-clamp variant；不代表论文128K–1M、生产模型或未修改作者harness。\n\n|模式|状态/完成条数|完整作者宏均值%|累计输出tokens|行内p50秒|输出tokens/行内秒|最后调用RSS峰值bytes|\n|---|---|---|---:|---:|---:|---:|\n"
    bookkeeping=Path(f'runs/round{number}-bookkeeping.json')
    if bookkeeping.exists():
        timing=json.loads(bookkeeping.read_text())
        if timing['documentation_review_exceeded_deadline']:
            text=text.replace('\n\n预算',f"\n\n模型、审计及验收快照在截止前结束；文档/Git收尾于{timing['at']}仍在进行，超过截止，未延长模型执行。\n\n预算",1)
    summary=[]
    for filename,entry in stats['files'].items():
        checked=audit['files'][filename]
        assert entry['sha256']==checked['sha256'] and entry['meta_sha256']==checked['meta_sha256']
        score=str(entry['author_summary']['configuration_macro_percent']) if entry['complete_mac_cohort'] else '未完整，不报宏均值'
        median='—' if entry['row_seconds_p50'] is None else f"{entry['row_seconds_p50']:.4f}"
        rate='—' if entry['output_tokens_per_row_second'] is None else f"{entry['output_tokens_per_row_second']:.2f}"
        text+=f"|{entry['stage']}|{entry['status']} {entry['items']}/13000|{score}|{entry['output_tokens']:,}|{median}|{rate}|{entry['last_call_peak_rss_bytes']}|\n"
        summary.append(f"{entry['stage']} {entry['items']}/13000 {'完整审计' if entry['complete_mac_cohort'] else '部分审计'}")
    text+='\n行内时间含prefill/解码/计分，不含逐行tokenization/写盘、加载及续跑核对；未配对任务/长度的吞吐不能作模式速度比，不宣称fused稀疏kernel加速。\n'
    for entry in stats['files'].values():
        text+=f"\n{entry['stage']}与旧诊断对应{len(entry['matched_old16_questions'])}题输入/输出完全一致；旧审计前缀bytes/SHA保持。\n"
        for example in entry['real_examples'][:1]:text+=f"\n- 真实生成节选 {entry['stage']} {example['source_id']}（raw前160字符，score={example['score']}）：{json.dumps(example['raw'][:160],ensure_ascii=False)}\n"
    proof=final.get('dense3743_resume')
    if proof:text+=f"\nDense3743条旧前缀及64-token未评分草稿实际重放验证通过，证据`{proof['file']}`；不计为额外已完成题目。\n"
    proof=final.get('dense_resume')
    if proof:text+=f"\nDense原{proof['saved_items']}条前缀及首个恢复行已验证，原草稿{proof['draft_tokens']}tokens，非空重放={proof['nonempty_draft_replayed']}；证据`{proof['file']}`。不额外加算题目，也不将0-token停点称为非空重放。\n"
    readiness=final.get('dense_control_readiness')
    if readiness:text+=f"\n后续8000步Dense训练对照的独立起点检查：{readiness['status']}，完整8000步数据RNG重放={readiness['full8000_data_rng_replayed']}，设备记录={readiness['devices']}，峰值RSS={readiness['peak_rss_bytes']}bytes；证据`{readiness['file']}`。这里只验证起点，不是Table2质量结果；后续训练进度见独立训练表及CONTINUE。\n"
    if readiness and readiness.get('reused_from_round25'):text+='\n上述起点检查沿用R25已执行且源码/数据/参数绑定保持的证据，本轮没有重新运行该probe。\n'
    guard=final.get('dense_training_guard_controls')
    if guard:text+=f"\n后续Dense训练守护入口的实际忙锁/过期截止拒绝检查通过，均未创建child；证据`{guard['file']}`。它们只验证拒绝路径，不能证明训练已经执行或完成。\n"
    protocol=final.get('table1_protocol_audit')
    if protocol:text+=f"\nTable1原文协议补核：28层25B-A3B，SWA窗口128；原Mac seq128无法检验SWA裁剪，后续保留W128需更长序列。证据`{protocol['file']}`仅为协议审计，三组独立架构尚未训练。\n"
    recovery=final.get('pressure_stop_recovery')
    if recovery:
        audit_note='初次审计在资源联合断言失败，原日志保留；后续原断言与阈值不变的观测审计通过' if recovery.get('initial_audit_failed') else '初次源/输出审计通过'
        text+=f"\n本轮{recovery['items']}条时guard原生pressure2/SIGINT保存退出，native仍为{recovery['native_status']}；{audit_note}。12次pressure1采样、旧PID不存在及SHA绑定review后，才在同一截止内恢复；证据`{recovery['review_file']}`。不改写资源停点，不归因于其它应用。\n"
    training=final.get('dense_training_checkpoint')
    if training:
        assert sha(training['file'])==training['sha256']
        text+=f"\n独立Dense对照真实checkpoint：{training['step']}/8000步，{training['tokens_saved']:,}tokens，设备{training['actual_training_device_from_config']}，阶段完成={training['phase_complete']}。有限状态、冻结indexer及数据RNG续跑审计通过；固定验证={training['fixed_validation']}。证据`{training['file']}`；这不代替Table2下游质量验收。\n"
    quality=final.get('dense_quality')
    if quality:
        state=quality['state'];text+=f"\n独立Dense质量阶段：{state['status']}，实际保存{len(state['samples'])}个配对样本、{len(state['jobs'])}个评测任务记录；八组完整目标保持，未完整不汇总，代码生成仍须独立判分。证据`{quality['file']}`。\n"
        for sample in state['samples']:
            if 'sha256' not in sample:continue
            assert sha(sample['output'])==sample['sha256']
            actual=json.loads(Path(sample['output']).read_text())
            text+=f"\n- 真实样本 {sample['mode']}/{sample['name']}（seed={actual['seed']}，temperature={actual['temperature']}，{len(actual['ids'])}tokens，滑窗{actual['context_tokens']}）：{json.dumps(actual['decoded'][:240],ensure_ascii=False)}\n"
        for job in state['jobs']:text+=f"\n- Dense质量 {job['name']}：{job['items']}/{job['target']}，{job['status']}，完整生成={job['full_generated']}，判分待完成={job['scoring_pending']}。\n"

    for stage,meta in final['saved_native_resume_metadata'].items():
        draft=meta.get('unscored_incomplete_row')
        if draft:text+=f"\n最终{stage}未评分草稿{draft['source_id']}保存{len(draft['output_ids'])}tokens，未计入完成条数；下轮须从此实际停点验证重放。\n"
    text+='\n|实际资源窗口|样本数|最大原生压力|采样child RSS最大bytes|host swap增量bytes|\n|---|---:|---:|---:|---|\n'
    for window in final['resource_windows']:text+=f"|{window['first_observed']} → {window['last_observed']}|{window['samples']}|{window['max_sampled_pressure']}|{window['sampled_child_rss_max_bytes']}|{window['host_swap_delta_bytes']}|\n"
    text+='\n仅报告实际采样窗口，不归因于其它应用或宣称全轮连续无swap。既有CPU/MPS测试为源码绑定复核，未重跑或加算测试。原权重/核心及冻结源保持；训练MTP/Table4、独立架构/质量对照及其它原目标未完成项仍在。Goal active，准确续跑见CONTINUE.md。\n'
    return text,f"|第{number}轮public500最新已审计快照|{'；'.join(summary)}|`{audit['file']}`、`{record['file']}`|\n"


def completed_dense_pairs():
    """Render only immutable, fully audited matched jobs; no partial aggregate."""
    text=''
    profile=json.loads(Path('profiles/paper_dense_bounded_evaluation.json').read_text())
    names=['gsm8k-v2' if job['name']=='gsm8k' else job['name'] for job in profile['jobs'] if job['suite'] in ('core','extra','mmmlu')]
    for name in names:
        candidates=[]
        for candidate in Path('runs').glob(f'round*-paired-{name}.json'):
            match=re.fullmatch(r'round(\d+)-paired-'+re.escape(name)+r'\.json',candidate.name)
            if match:candidates.append((int(match[1]),candidate))
        if not candidates:continue
        _,path=max(candidates)
        pair=json.loads(path.read_text())
        for filename,record in pair['files'].items():
            with Path(filename).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==record['sha256']
            with Path(record['audit']).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==record['audit_sha256']
            checked=json.loads(Path(record['audit']).read_text())['files'][filename]
            assert checked['full_dataset_generated'] and checked['items']==pair['matched_items']
        if not text:text='\n## 独立Dense对照：已完成的配对质量任务\n\n仅列全部目标生成并独立审计通过、逐题prompt/gold/input IDs一致的任务；其余八组完整目标仍须继续，不作部分总体汇总。运行变体和保护阈值差异保留，不代表官方成绩、数值逐位等价或稀疏kernel加速。\n\n|任务|配对题数|Sparse micro准确率%|Dense micro准确率%|配对正确性计数|证据|\n|---|---:|---:|---:|---|---|\n'
        text+=f"|{pair['job']}|{pair['matched_items']}|{100*pair['sparse_accuracy']:.6f}|{100*pair['dense_accuracy']:.6f}|{pair['paired_correctness']}|{path}|\n"
    return text


def main():
    rows=[]
    stability=[]
    for name in ('qwen-probe','qwen-shakespeare','qwen-cpt','qwen-distill-paper','qwen-distill-paper-lr1e4','qwen-distill-paper-batch8','qwen-distill-hf','qwen-sparse','qwen-sparse-paper',
                 'qwen-control-muon','qwen-control-adamw','qwen-control-ungated','qwen-dense-control','qwen-dense-paper-control'):
        path=Path('runs')/name
        if not (path/'metrics.jsonl').exists(): continue
        records=[json.loads(line) for line in (path/'metrics.jsonl').read_text().splitlines()]
        steps=[r for r in records if 'step_seconds' in r]
        if not steps: continue
        config=json.loads((path/'config.json').read_text()); args=config['args']
        validations=[r for r in records if 'val0/lm' in r]
        state='完成阶段' if any(r.get('stopped')=='phase_complete' for r in records) else '部分/运行中'
        # Distillation acceptance uses its own audit, not merely completed steps.
        audit=Path('runs/audit-qwen-'+name.removeprefix('qwen-distill-')+'.json')
        if name.startswith('qwen-distill-paper') and audit.exists() and not all(v['nondegenerate'] for v in json.loads(audit.read_text())['corpora'].values()):
            state='步数跑完，非退化验收失败'
        times=[r['step_seconds'] for r in steps if r['step']>2] or [r['step_seconds'] for r in steps]
        first,last=validations[0],validations[-1]
        extra=f"LM {first['val0/lm']:.5f}→{last['val0/lm']:.5f}; KL {last['val0/kl']:.6f}"
        if 'val1/lm' in last: extra+=f"; Sh LM {last['val1/lm']:.5f}"
        rows.append(f"|{name}|{steps[-1]['step']}/{args['steps']} ({state})|{steps[-1]['tokens_seen']:,}|{statistics.median(times):.4f}|{max(r['process_peak_rss_bytes'] for r in steps):,}|{max(r['mps_driver_bytes'] for r in steps):,}|{extra}|")
        grads=[r['grad_norm'] for r in steps]
        stability.append(f"|{name}|{max(grads):.4f}|{sum(g>args['clip'] for g in grads)/len(grads):.3f}|{sum(g==0 for g in grads)}|")
    env=json.loads(Path('runs/environment.json').read_text())
    counts=json.loads(Path('profiles/parameter_counts.json').read_text())
    resume=json.loads(Path('runs/qwen-resume-comparison.json').read_text())
    data={name:json.loads((Path('data')/name/'meta.json').read_text())['files'] for name in ('shakespeare','owt')}
    audits=[]
    for name in ('paper','paper-lr1e4','paper-batch8','hf'):
        file=Path(f'runs/audit-qwen-{name}.json')
        if file.exists():
            a=json.loads(file.read_text())
            for corpus,v in a['corpora'].items():
                audits.append(f"|{name}|{corpus}|{v['kl']:.6f}|{v['uniform_student_kl']:.6f}|{v['nonuniform_query_fraction']:.4f}|{v['nondegenerate']}|")
    evaluations=[]
    latency=[]
    for file in sorted(Path('runs').glob('qwen-eval-*.jsonl.meta.json')):
        a=json.loads(file.read_text())
        evaluations.append(f"- `{file}`: "+'; '.join(f"{k} {v['correct']}/{v['n']}（该adapter集合{v['total_dataset']}）" for k,v in a['tasks'].items()))
        predictions=[json.loads(s) for s in Path(str(file).removesuffix('.meta.json')).read_text().splitlines()]
        usable=[r for r in predictions if 'prefill_seconds'in r and len(r['output_ids'])>1]
        if usable:
            latency.append(f"- `{file.name}` / {a['device']}: prompt长度{min(r['prompt_tokens'] for r in usable)}–{max(r['prompt_tokens'] for r in usable)}，prefill中位{statistics.median(r['prefill_seconds'] for r in usable):.4f}s，后续每token中位{statistics.median((r['seconds']-r['prefill_seconds'])/(len(r['output_ids'])-1) for r in usable):.4f}s。")
    controls='独立训练对照尚未完成验证。'
    comparison=Path('runs/qwen-controls-comparison.json')
    if comparison.exists():
        v=json.loads(comparison.read_text())
        controls='初始化权重SHA一致，数据RNG及算法source hashes一致；每组256步/32,768 tokens、lr.0003常数、无warmup。\n\n|对照|起点LM|终点LM|最大clip前梯度norm|\n|---|---:|---:|---:|\n'
        for name,row in v['controls'].items():
            controls+=f"|{name}|{row['initial_lm']:.5f}|{row['final_lm']:.5f}|{row['max_grad_norm']:.5f}|\n"
        q=v['dense_qsa']
        controls+=f"\n独立dense/QSA各200步/25,600 tokens，起始backbone和数据顺序相同；dense OWT LM={q['dense']['val0/lm']:.6f}，QSA={q['sparse']['val0/lm']:.6f}；Shakespeare分别={q['dense']['val1/lm']:.6f}/{q['sparse']['val1/lm']:.6f}。QSA还训练indexer KL，所以对照是两套训练过程，而非只切mask。证据 `runs/qwen-controls-comparison.json`。短程结果不证明任何优化器或GR在官方规模的优劣。"
    diagnostics=[]
    for file in Path('runs').glob('qwen-diagnostics-*.json'):
        d=json.loads(file.read_text())
        for corpus,r in d['corpora'].items():
            experts=r['experts']; unused=[v['unused_experts'] for v in experts.values()]
            entropies=[round(v['normalized_entropy'],4) for v in experts.values()]
            line=f"- `{file}` / {corpus}: 各层未使用专家数{unused}，归一化路由熵{entropies}。"
            if r['qsa']:
                frac=sum(v['pruned_query_fraction'] for v in r['qsa'])/len(r['qsa'])
                line+=f" 实际裁剪query比例{frac:.4f}，最多选择{max(v['max_selected_tokens'] for v in r['qsa'])} tokens。"
            diagnostics.append(line)
    samples=[]
    sample_files=list(Path('runs').glob('qwen-sample-*.json'))
    for suffix in ('romeo','web'):sample_files.extend(Path('runs').glob(f'paper-sparse-*-{suffix}.json'))
    for file in sorted(sample_files):
        sample=json.loads(file.read_text())
        label=f"step {sample['step']}/{sample['target_steps']}，phase_complete={sample['phase_complete']}" if 'step'in sample else '已完成阶段的终点；checkpoint hash见原文件'
        excerpt=sample['decoded'][:240]
        samples.append(f"`{file}`（{label}；仅展示前240字符，完整输出/IDs保留）：\n\n"+'\n'.join('> '+line.rstrip() if line.strip() else '>' for line in excerpt.splitlines()))
    suite_rows=[]
    prediction_metas=list(Path('runs').glob('eval-sparse-*.jsonl.meta.json'))+list(Path('runs').glob('eval-paper-*.jsonl.meta.json'))
    for file in sorted(prediction_metas):
        run=json.loads(file.read_text())
        if run.get('identity',{}).get('suite')=='mrcr':continue
        tasks=run['tasks']
        if run.get('identity', {}).get('suite') == 'gpqa':
            for name, count in tasks.items():
                suite_rows.append(f"|{file.name}/{name}|{run['status']}|{count['n']}/{count['total_dataset']}|{count['correct']}/{count['n']}|{count['capacity_failures']}|独立模板/配置；在途分数未经terminal审计|")
            continue
        n=sum(t['n'] for t in tasks.values());target=sum(t.get('target',t['total_dataset']) for t in tasks.values())
        failures=sum(t['capacity_failures'] for t in tasks.values());right=sum(t['correct'] for t in tasks.values())
        scored=sum(t.get('scored',t['n']) for t in tasks.values())
        result=f'{right}/{scored}' if scored else '未计分'
        scope=run.get('identity',{}).get('scope','合并：原始全量 + 仅容量失败补跑')
        suite_rows.append(f"|{file.name}|{run['status']}|{n}/{target}|{result}|{failures}|{scope}|")
    resource=''
    code_summary='代码生成/判分尚未全部完成；生成文件中的null不是0分。'
    judge=Path('runs/evalplus-score-full-v4.jsonl.meta.json')
    if not judge.exists():judge=Path('runs/evalplus-score-full-v3.jsonl.meta.json')
    if not judge.exists():judge=Path('runs/evalplus-score-full-v2.jsonl.meta.json')
    if not judge.exists():judge=Path('runs/evalplus-score-full.jsonl.meta.json')
    if judge.exists():
        result=json.loads(judge.read_text())
        code_summary=f"官方EvalPlus0.3.1判分：{result['status']}，{result['completed_items']}/{result['target_items']}题；证据对应judge JSONL及meta。"
        if result['status']!='complete' and Path('runs/evalplus-partial-audit.json').exists():
            partial=json.loads(Path('runs/evalplus-partial-audit.json').read_text())
            code_summary+=' 已核验HumanEval base/plus各0/164（该子套完整），MBPP base/plus各0/72（378题中的部分），不报告四套均值。Mbpp/255的全部组合输出对象估算至少1.32GB，超过当前1GB容器，并非证明16GB主机不可执行；后续需安全调整有界profile/逐题调度，不移除测试。证据runs/evalplus-partial-audit.json与runs/evalplus-oracle-resource-audit.json。'
        if result['status']=='complete':
            code_summary+=' '+ '；'.join(f"{name}: base {v['base_correct']}/{v['n']}，plus（base与plus均通过）{v['plus_correct']}/{v['n']}" for name,v in result['tasks'].items())
            code_summary+=f"。四套均值{100*result['four_suite_mean']:.4f}%。这是同一终点缩小模型、768-token贪心base completion的全量结果，不是官方Qwen成绩。"
    batch_resources=[]
    prediction_metas=list(Path('runs').glob('eval-sparse-*.jsonl.meta.json'))+list(Path('runs').glob('eval-paper-*.jsonl.meta.json'))
    for file in sorted(prediction_metas):
        run=json.loads(file.read_text())
        if run.get('resources'):
            batch_resources.append(f"- `{file.name}`：进程峰值RSS {run['resources']['process_peak_rss_bytes']:,} bytes；批次时间均摊记录不等于独立题目延迟。")
    trace=Path('runs/round2-resources.jsonl')
    if trace.exists():
        observed=[json.loads(s) for p in sorted(Path('runs').glob('round*-resources.jsonl')) for s in p.read_text().splitlines()]
        observed=[r for r in observed if r.get('swap',r.get('swapusage')) is not None]
        observed.sort(key=lambda r:r.get('time', r.get('at')))
        used=[float(re.search(r'used = ([\d.]+)M',v.get('swap',v.get('swapusage'))).group(1)) for v in observed]
        resource=f"第二轮曾启动本机已有Docker Desktop，默认8GB VM自动启动7个已有容器，造成系统swap明显跃升；已停止并恢复原先关闭状态。各轮日志观测swap最大{max(used):.2f} MiB，最近{used[-1]:.2f} MiB、压力level {observed[-1]['pressure']}。随后改为独立2CPU的本机Colima VM（常规2GB，完整EvalPlus单题判分3GB），只读共享runs/code-jobs，代码执行与文本生成串行；实时状态以vm.sh status为准。原始证据：runs/docker-default-info.json、runs/code-vm-mounts.txt、runs/round2-resources.jsonl、runs/round3-resources.jsonl、runs/round4-resources.jsonl。不能把本实验称为零swap。"
    inference_proof=Path('runs/evaluation-v2-proof.json')
    cache_proof=''
    if inference_proof.exists():
        proof=json.loads(inference_proof.read_text());t=proof['median_seconds']
        cache_proof=f"同一终点checkpoint、5题×32 tokens，增量KV与完整前缀参考生成IDs完全一致，中断/连续评测IDs一致；中位整题耗时{t['uninterrupted']:.4f}s vs {t['reference']:.4f}s（runs/evaluation-v2-proof.json）。分块128路径另与参考核对5题IDs一致（runs/chunk-trained-ids-proof.json）。这是缓存重用带来的开销下降，不是GPU稀疏kernel加速比。MMLU一次在途进程RSS采样945,176,576 bytes（不是峰值），见runs/eval-memory-samples.jsonl。CPU线程实测1线程快于2/4线程，故保留1线程。"
    text=f'''# Qwen3.8-Flash-Next Mac recipe：实际进度与结果

更新时间：{datetime.now().astimezone().isoformat(timespec='seconds')}。**Goal active，尚未完成。** 本报告由 `uv run python report.py` 读取本机日志生成；阶段完成不等于整个 goal 完成。原 `qwen_next_lab` 没有修改。

## 环境、来源与配置

Apple M4 Mac mini，统一内存 {env['memory_bytes']} bytes；macOS {env['macos']}，arm64 Python {env['python']}，PyTorch {env['torch']}，MPS built/available 均为 true。环境采样：`{env['swap']}`；`{env['pressure']}`；没有设置 MPS 内存保护覆盖变量。训练时另有瞬时压力 level 2，曾观察到swap约117MB，后在稀疏训练中增至约340MB；暂停训练进程后压力恢复1。后续串行执行，`runs/resources.jsonl` 持续记录换入/换出计数与swap增量，不能把这次运行称为全程零swap。退出后的压力与训练峰值不能混为一谈。`runs/environment.json` 是一次采样，不是全程峰值。

{resource}

来源/许可证：`sources.lock.json`、`THIRD_PARTY_NOTICES.md`；逐项差异见 `RECIPE_MATRIX.md`。正式作者代码 lineage 从 commit `cc7a8c3` 开始，checkpoint 内还锁定四个算法/训练文件 SHA256。`profiles/official.json` 保留发布配置；`profiles/mac_qwen.json` 是启动前声明的本机预算，其planned字段保留启动时含义；最新执行状态以本报告和原始metrics为准。

本机文字模型32,978,962参数，width64/4层/4个GR分支，保留完整248,320词表、12Q:1KV、core head256/rotary64、GDN 1key:3value heads/128维、index4Q:1K/head128/rotary64；8个routed experts/top2与gated shared expert。输入/输出各15,892,480参数，Ngram表67,584参数。GR rank8；Ngram base1024、每阶2heads、层2。规模缩小并不保持官方参数占比；词表主导本机参数。

`manifest.py` 不分配官方大权重，按实现公式和发布text_config解析计数，结果在 `profiles/parameter_counts.json`。推导官方文字总量{counts['official_text']['parameters']:,}，其中Ngram表{counts['official_text']['components']['ngram_table']:,}；不含vision/MTP，也未逐个核对官方权重archive。逻辑每token活跃参数（含整个输出词表投影）本机dense={counts['mac']['active_parameters_per_token_dense']:,}、QSA={counts['mac']['active_parameters_per_token_qsa']:,}；官方映射QSA={counts['official_text']['active_parameters_per_token_qsa']:,}。论文约6B的口径不能与含输出head的计数直接比较。该计数不是运行时工作集或FLOPs。

优化器：锁定作者 Canzona `qwen_express` 8步系数；FP32 master gradient归一化后BF16 NS、.95 lerp/Nesterov、eps1e-14、.2√max矩阵形状缩放；按head和gate拆分。Adamβ(.9,.95)/eps1e-8、WD.1沿用作者Muon低层构造器默认，是明确Mac选择。框架层OptimizerConfig默认却为β₂=.999/WD.01；NS默认5步/simple也不同于论文8步。生产启动覆盖值未获证实，不能把任何一层默认当作生产配方。Ngram表Adam无decay；indexer归属为明确Mac选择。主体/Adam状态FP32。没有运行Canzona通信/CUDA graph或GPU fused kernel。

逐张量/逻辑子矩阵优化器、storage/logical shape、分片与实际decay见 `profiles/mac_optimizer_ownership.tsv`，由本轮安全checkpoint的真实param_groups导出，不含权重值。

正式各阶段seq128、batch1、accum1，固定seed42，验证seed1234×8batch，clip.5、MoE辅助系数.001。Shakespeare/CPT各1000步，lr3e-4、50步LR warmup+cosine；蒸馏1000步lr1e-3常数；稀疏8000步lr2.5e-5常数。每次调用1200秒，原子checkpoint，续跑保留optimizer与数据RNG。生产语料、base schedule、完整初始化和部分训练参数未公开。

## 测试与续跑

历史43项测试在CPU和MPS模式通过（最新完整60项及MTP独立4/5项见末尾验收表）：`runs/test-cpu-suite43.log`、`runs/test-mps-suite43.log`。像素处理、数据覆盖/计分与小模型批次恢复fixture实际使用CPU，模型相关测试按device运行。涵盖锁定上游GDN/GR/Ngram/PLE输出、梯度与缓存，QSA真实gather、因果/tie/尾块、KL/冻结，Muon多形状/病态/零矩阵/SVD、fused拆分、混合ownership，以及MTP、视觉encoder、多模态接线、增量/分块KV与原始像素处理。作者NS函数、八步权重及momentum更新分别与同设备作者reference逐位一致，不代表跨设备或整个生产训练逐位复现。

中断/连续4步对照：最大权重差 `{resume['max_parameter_absolute_difference']}`，数据RNG相同，atol2e-5/rtol2e-4通过（BF16轨迹非逐位确定）；证据 `runs/qwen-resume-comparison.json`。评测last-logit优化路径也与完整forward及分段cache人工数值检查通过。

正式训练终点为8000/8000，所有模型/optimizer张量有限、训练核心source hashes匹配，终点checkpoint SHA及证明在 `runs/final-checkpoint.json`。Goal仍active，完整14项评测未完成。

## 数据上限

沿用旧实验已下载的Shakespeare与有界OWT，重新使用锁定官方tokenizer；没有新增OWT下载。OWT来自固定revision的首分片，有界20MB训练/1MB验证、4,198/221文档，文档SHA256 modulo20划分及精确去重。UTF8尾部裁剪本次均0。每份meta记录原始来源、hash及新token hash。训练随机采样连续token窗口，可能跨EOD；Ngram在EOS处重置寻址，attention/GDN不按文档重置，训练窗口之间不携带cache。官方packing/mask语义尚未确认；这不是官方生产数据配比、清洗或contamination保证。

```json
{json.dumps(data,indent=2)}
```

## 作者代码 lineage 的实际训练

下表速度是排除前两步后的训练步中位数，不含validation/checkpoint；tokens为当前阶段，CPT含10%概率Shakespeare混合。RSS与MPS driver是不同统计，**不能相加**。各次进程峰值来自metrics，训练尚在运行的行不是最终峰值。

|run|步数/状态|阶段tokens|秒/步|进程peak RSS B|MPS driver B|固定验证|
|---|---|---:|---:|---:|---:|---|
{chr(10).join(rows)}

验证loss为token级自然对数cross entropy；短序列、小语料、全词表小模型的低loss不能当作通用能力证据。正式checkpoint和完整逐步记录位于各run目录的 `last.pt`、`metrics.jsonl`、`config.json`。

|run|最大clip前梯度norm|超过clip.5的步骤比例|零梯度步骤|
|---|---:|---:|---:|
{chr(10).join(stability)}

训练入口遇到非有限loss/梯度会停止；现有已提交轨迹中未发生此类故障。这个常规梯度裁剪实验并非论文的高学习率stress test，也没有宣称复现生产训练全程无spike。

## QSA非退化与真实裁剪

论文Eq15原始sum(ReLU)与发布HF额外除√128分别实验，绝无head平均或未标记温度。正式r4/K32意味着后段query裁剪，选中多个块后计算KL；gather只生成T×(K+r−1) core权重。dense mask只用于测试oracle。此为数学参考实现，**没有GPU fused QSA kernel加速声明**。

|score profile|验证数据|学习后KL|均匀student KL|非均匀query比例|通过非退化检查|
|---|---|---:|---:|---:|---|
{chr(10).join(audits) if audits else '|待正式蒸馏完成并审计||||||'}

审计入口 `audit_stage.py` 要求KL优于均匀student、至少50%有>=2候选块query的分数非均匀、每批KL梯度为正。其标准是本机诊断门槛，不是官方论文指标。还需验证蒸馏仅indexer变化、backbone冻结和两种profile数据RNG相同。

## 生成与评测

完整数据/生成任务进度如下。**各文件不能相加计为样本总数**：MGSM初次全量含141个容量失败，定向补跑只替换这141条，complete合并文件才是全部2750条实际生成。每次补跑核对同一checkpoint、完整prompt、input IDs、gold、shots及token预算，保留两份原始文件和每条来源；无答案优选。证据 `merge_capacity.py`。

旧HF终点代码成绩（下文不代表新paper终点）：\n\n{code_summary}

完整EvalPlus审计见runs/evalplus-completed-audit.json：全部542条prompt/IDs/solution、执行脚本字节、官方结果、镜像、资源限制及清理一致；完整base/plus输入数与锁定源数据一致。官方fast-check对496题在断言执行前失败，不能将完整判定冒充每条断言全部运行；该早退规则未改。

{chr(10).join(batch_resources)}

|证据meta|调用状态|已处理/任务目标|正确/已处理|容量失败|范围|
|---|---|---:|---:|---:|---|
{chr(10).join(suite_rows)}

MGSM合并结果20/2750=0.7273%，非英语10种语言均值0.76%；这是Mac已声明prompt/64-token贪心预算下的全量结果，不是官方recipe分数。初次context2048，容量补跑context4096/prefill_chunk128，均不截断原prompt。MMLU实测最大prompt3138 tokens，全集14042题采用context4096/chunk128；该任务已完整生成14042题，0题正确、无容量失败。另已准备MMLU-Pro 14类12,032题的5-shot公开validation CoT adapter，最长prompt2,862 tokens；BBH/GSM8K共7,830题最长2,484 tokens，容量审计保存在runs/mmlu-pro-context-audit.json与runs/bbh-gsm-context-audit.json。GSM8K全部1319题已生成，4题正确（0.3033%）；BBH全部6511题已生成，2题正确（micro 0.0307%，23家族macro 0.0348%）。全部prompt/gold/input IDs/raw解码/计分复核通过，证据runs/gsm-bbh-completed-audit.json；MMLU-Pro12032题已按profiles/pro_eval.json启动，实际进度见上表。MATH全部5000题完成，0正确、零容量失败；完整输入/IDs/raw解码/计分复核通过，runs/math-completed-audit.json；profiles/math_eval.json固定4-shot/64-token/greedy/context4096，max prompt3153，保留完整示例；实际进度见上表。另有INCLUDE 44语言22639题、SuperGPQA 26529题、Redux2-ok 57科5330题适配器已准备，实际生成状态见上表；Redux版本不能冒充论文未公开版本。INCLUDE validation/test有重题，已排除全部同语言test题；Estonian/Georgian/Malayalam分别使用5/2/1个固定英语MMLU anatomy dev补足5-shot，其余例子保留同语言。原始重叠与最终示例来源见runs/fewshot-overlap-audit.json、runs/include-fewshot-plan.json；MATH/Pro/Super公开示例未发现规范化后的完全相同测试题（不证明无语义重叠）。

{cache_proof}

MMMLU完整语言的已审计计数见下方自动汇总；未完成全部14种语言时不报告完整均值。后续af011ff按科目缓存，全部196588行prompt/gold摘要不变，真实33题×64生成IDs与冷路径完全相同（runs/mmmlu-prefix-trained-proof.json）；原完成文件保留旧源码。47项CPU/MPS模式测试见runs/test-{{cpu,mps}}-subject-prefix47.log。

可选前缀缓存将真实token共同前缀按128边界复用并克隆到各batch，4个LRU槽；CPU/MPS状态/种子不可变/部分批次重放验证通过，真实终点24题×64新tokens与基线IDs完全相同（runs/prefix-trained-proof.json）。并发负载下的诊断时间不作受控性能声明。只在MultiPL-E完整生成结束后启用，首个正式运行是MATH；冻结helper源码与prefix计划hash，完整prompt/IDs不省略。

可选--grouped-gqa按共享KV分组进行矩阵收缩，避免query-head重复展开；训练四文件不变。CPU/MPS43项通过，独立CPU进程同24题×64原始IDs均一致，峰值RSS3121283072→1158955008 bytes（runs/gqa-comparison.json）。这是小规模单次对照，不推断全benchmark内存或GPU生产kernel加速。

公共GDN chunk scan仅用于prefill，decode仍用原recurrent路径；训练四文件不变。CPU/MPS容差测试覆盖两个stage、EOS、31/128分块、状态和16步argmax。真实终点17题×64新tokens与旧路径IDs全部一致；三组单次总时间分别1.436→1.196s、1.728→1.386s、1.051→0.757s。证据runs/prefill-trained-proof.json；有限单次观测，不是所有输入逐位相等或GPU QSA kernel加速。

`sample.py` 保存实际采样raw IDs/decoded/padded-vocab异常与checkpoint hash；早期pilot入口`evaluate.py`及新的全量/断点入口`evaluate_suite.py`保存完整prompt/token IDs、few-shot、贪心生成、答案抽取和逐题计分。GSM8K固定4-shot、MGSM固定8-shot作者示例；另有MMLU algebra一科5-shot及BBH两任务作者3-shot模板。source revision/hash在 `eval_sources.lock.json`。早期pilot仅每个adapter2题、64新token、context2048；当前全量任务覆盖以上表为准，GSM8K/BBH全集已完成。MMLU/MGSM逐题prompt、gold、重新编码input IDs、解码输出及全部分数另经完成审计验证（runs/completed-evaluation-audit.json）；MMLU未发现以选项字母开头而后接解释的漏计格式。整体**尚未完成14项评测**；超过训练128上下文是外推，容量不足明确失败，不删few-shot。

下列样本固定seed42、temperature.8、最多80新token、seq128滑窗重算；文本仍不连贯，不能当作已获得通用能力。部分稀疏checkpoint单独标注，未称最终训练结果。

{chr(10).join(samples)}

{chr(10).join(evaluations) if evaluations else '正式lineage的生成pilot尚待执行。'}

其余任务及完整benchmark状态见 `EVALUATION.md`。官方prompt/解码等完整工程未发布；自定义SWEBench-Pretrain不能换成标准SWE-bench。EvalPlus已完整判分；MultiPL-E已完成八语言生成、隔离编译/测试和来源审计；原协议三条数据错误与独立Mac修复结果分列，不在宿主执行生成代码。

生成式pilot的分项耗时如下，包含本机eager参考计算和head-last投影优化。第一个token使用prefill logits，其余为带cache的逐token路径；attention cache仍重算prefix，不能据此声称生产KV kernel性能。不同prompt长度的中位数仅描述本机运行，不是dense/QSA kernel对照。

{chr(10).join(latency)}

## 有界独立对照与专家诊断

{controls}

{chr(10).join(diagnostics) if diagnostics else '专家使用率和裁剪诊断待串行验证。'}

## 旧探索与局限

旧通用PolarExpress系数、FP32 NS、WD.01的实验保留为**superseded exploratory**，精确source commit `c7170da`。它的Shakespeare/CPT/蒸馏结果不能移入作者代码lineage，旧 `sparse-hf` 在step800/8000安全中断，详情 `runs/exploratory-lineage-status.json`。旧raw分数indexer非均匀query约.1%且KL差于均匀baseline；HF分数版本通过。该现象说明小规模敏感性，不证明论文公式错误。

MTP `residual_linear_shared`/全局4D norm/双stream/selection复用已有组件测试，主文字训练未加入MTP，生产loss权重/schedule未知。`pixels.py`实现CPU RGB/已解码帧的智能resize、归一化、patch排列、采样和时间戳；`multimodal.py`实现视觉特征替换、三轴位置与无padding的dense/sparse文字forward，数值/梯度对照通过。视觉分支为缩小随机初始化组件，未进行多模态训练；video_decode.py新增有界PyAV16.1.0本地视频解码，processor.py接通prepared-text占位符、时间戳、完整tokenizer和左右padding；真实PNG/MOV、锁定HF方法及像素到视觉/文本梯度链路已验证。完整HF chat/URL/streaming API仍有缺口。新增binary padding mask路径逐样本移除padding，左/右padding的有效logits、三轴位置与梯度在CPU/MPS匹配独立无padding序列，padding logits置零；这是参考row-packed实现，不保证HF padded位置输出或kernel性能。官方image像素预算65536..16777216、video4096..25165824；Mac预算参数分别65536..262144和4096..262144，max_frames64（官方768），上游rounding/temporal padding可使实际面积略超预算参数。不能称完整多模态recipe复刻。

代码评测基础设施：本机独立Colima/Lima VZ VM，常规2GB RAM/2CPU；完整EvalPlus使用3GB VM/2GB单题容器；Python镜像锁定于code_images.lock.json，无网络、非root、只读root/任务挂载、内存/CPU/PID/输出/墙钟限制。九项探针通过（runs/sandbox-probe.json），包含host文件不可见、网络不可达、OOM/CPU/超时/PID/输出限制和清理；进程exit0不被当作benchmark正确性。EvalPlus官方0.3.1包、测试数据与Linux arm64哈希依赖已构建，pip check通过；两数据集共12个正/负例及sanitizer控制通过（runs/evalplus-probe.json），模型全量生成/正式计分进度以本机证据为准，未计分代码题不得报告为0分。MMMLU14语言全部196588题源文件已核验（另列128MB缓存），adapter测试及容量审计通过，max prompt4197，后续context8192保留全部5-shot。Swahili/Yoruba三种异常Subject值共445行只做科目映射，原始CSV/问题/答案未改。单个语言生成不能代表14语言完成。MultiPL-E八语言1277题源数据完整锁定；Python由发布者161个cleaned originals及原translator派生（公开源不含32/38/50），并非直接重用EvalPlus164题。全部1277题实际生成完成，共1307648 tokens；完整输入/IDs/raw/停止后文本审计见runs/multipl-generation-audit.json。原协议有三条不可归因数据错误，另有非官方Mac修复组合；公开入口默认1024新token，Mac每题一次greedy与作者建议20次temperature0.2采样不同。固定预算profiles/multipl_eval.json。原生arm64镜像965026124 bytes已构建，32个正负控制及9个隔离/资源探针通过。原始1277条测试与全量脚手架核验完成，raw pass0；CPP137/JS51/PHP23有发布数据静态错误，不计为模型答错，完整原协议准确率null。其余1274有效脚手架0正确。另有明确标记的Mac修复组合：只重生成/替换上述3条，保留原始文件，1277条八语言macro0%，不是官方Qwen或同一发布协议成绩。证据runs/multipl-final-audit.json；源码归档c12faae。GPQA已于2026-11-09授权成功；Main448/Diamond198/Extended546固定来源已下载并验证（gpqa_sources.lock.json、runs/gpqa-access.json）。作者原始5-shot CoT的示例2、3与测试题规范化后重复，另列clean Mac模板，只替换这两例、保留全部测试行。1292组输入与作者提示/洗牌/gold逐字对照，parser及源漂移控制共5项通过；max prompt4886，加600新token≤8192，无截断。四配置一次greedy、无chat包装，600预算来自作者non-chat公开入口，Qwen私有协议未知。冻结源归档eb48785；实际生成及审计进度见上表/后文，不能把四配置混成单一GPQA准确率。

本轮结束时必须保存准确checkpoint/续跑命令；阶段预算不能为结束goal而缩减。数据、权重、tokenizer缓存及环境均被git忽略。
'''
    choice = Path('runs/choice-output-diagnostics.json')
    if choice.exists():
        text += '\n## 选择题输出诊断\n\n'
        for name, result in json.loads(choice.read_text()).items():
            text += f"- {name}: {result['items']}题，抽取计数{result['extracted_answer_counts']}，正确{result['correct']}。\n"
        text += '\n固定原parser和分数；Pro的命中全部来自I标签，不能解释为推理能力。原始样本与结果文件SHA见runs/choice-output-diagnostics.json。\n'
    performance = Path('runs/round6-eval-performance.json')
    if performance.exists():
        text += '\n## 完整评测实测速度\n\n|集合|题数|生成tokens|活动秒|题/秒|峰值RSS bytes|\n|---|---:|---:|---:|---:|---:|\n'
        for name, result in json.loads(performance.read_text()).items():
            text += f"|{name}|{result['items']}|{result['generated_tokens']}|{result['active_event_seconds']:.1f}|{result['items_per_active_event_second']:.3f}|{result['process_peak_rss_bytes']}|\n"
        text += '\nCPU1的实际全量生成；活动时间排除初始加载/分词及停顿，包含prefill/解码与记录开销。非单题延迟或GPU稀疏kernel加速。原始统计见runs/round6-eval-performance.json。\n'
    if Path('runs/round6-memory-incident.json').exists():
        text += '\n第六轮SuperGPQA续跑期间，系统swap从4802.12升至9490.69 MiB，采样压力曾达4，随后回落。任务VM保持停止，评测进程当时记录峰值RSS3772022784 bytes；现有采样不足以将系统跃升归因于单一进程，也不能以恢复level1称全程无影响。证据runs/round6-memory-incident.json、runs/round6-resources.jsonl。\n'
    super_audit=Path('runs/supergpqa-completed-audit.json')
    if super_audit.exists():
        result=json.loads(super_audit.read_text())['files']['runs/eval-sparse-supergpqa-intern-full.jsonl']
        text += f"\nSuperGPQA全量审计通过：{result['correct']}/{result['items']}，文件SHA {result['file_sha256']}。完整prompt/gold/input IDs/raw解码/计分见runs/supergpqa-completed-audit.json。三段源码/旧前缀字节链见runs/supergpqa-completed-provenance.json；原17654、retention-only18547均为该完整文件前缀，不能相加。存储优化不改模型/输入值/生成预算；完整26529题内容摘要一致，真实8题×64及部分batch恢复IDs对照通过。共享整数版本进程peak RSS2326462464 bytes。\n"
    if Path('runs/test-cpu-suite47.log').exists():
        text += '\n最新47项CPU/MPS模式套件通过（数据、codec和部分fixture实际CPU）；padding改用发布pad token后，3项processor测试在CPU/MPS再次通过。日志runs/test-{cpu,mps}-suite47.log、test-processor-pad-{cpu,mps}.log。video_decode.py采用有界本地PyAV解码与原帧索引/FPS时间戳，processor.py采用锁定HF替换/模态类型规则，真实PNG/MOV与随机缩小模型视觉到文本梯度链路通过；未训练视觉权重，不是完整HF聊天/URL/流式API或官方视频backend。限制见profiles/mac_processor.json，环境见runs/media-environment.json。\n'
    if Path('.cache/references/generation_config.json').exists():
        text += '\n已补锁发布generation_config：采样T1/top_k20/top_p0.95、EOS[248046,248044]，pad248044。Mac base completion固定greedy64/模型文档EOS248044；17个终止文件逐文件扫描未输出额外EOS248046（文件SHA保留、重叠前缀不相加），额外停止符对这些记录无提前截断影响，采样差异仍存在。来源sources.lock.json、profiles/released_generation.json；证据runs/released-eos-output-audit.json。候选generative harness的模板/抽取也不等同本机base模板，具体差异见EVALUATION.md。\n'
    text += '\n## Indexer 参数化数值补充\n\n`audit_indexer_parameterization.py`只复制已训练indexer并使用合成hidden/teacher，在CPU FP32验证：有效RMSNorm增益是`1+weight`，令两侧增益均乘`128**(-1/4)`可将不缩放head求和映射到HF的`/sqrt(128)`分数。128/512长度三组探针有效分数最大绝对差7.63e-6，选块/选中tokens完全一致，KL最大差7.16e-7，按链式法则换算后的梯度最大差1.56e-7。结果runs/indexer-parameterization-audit.json含checkpoint及代码hash。初次探针错误地缩放存储weight而非有效增益，失败日志原样保留；修正后通过。\n\n该参数映射仅验证函数族关系：相同数值优化器、初始化和clip不保证相同训练轨迹；未修改主checkpoint、optimizer或四个训练源文件。原batch1 paper冷启动蒸馏未通过非退化验收；该参数探针不能取代训练。后续batch8新分支的实际通过证据另列。\n'
    text += '\n严格paper重试：profiles/paper_distill_retry.json在启动前固定同一CPT、seed42、数据混合/顺序、128000 tokens及1000步，仅学习率从.001降为.0001。MPS完成1000步，全部非indexer张量与CPT逐位相等、全部模型/optimizer有限、最终batch RNG与原失败run相同；runs/paper-lr1e4-integrity.json。仍失败：OWT KL .172172135 > 均匀 .168170625，Sh .375386741 > .374166284，两语料非均匀query比例均 .00103306；完整门槛和梯度在runs/audit-qwen-paper-lr1e4.json。原失败分支与主HF checkpoint不改；不进入该失败分支的稀疏训练，不降低门槛或用函数参数映射代替训练验收。\n'
    mmmlu=[]
    for file in sorted(Path('runs').glob('mmmlu-*-completed-audit.json')):
        for name,result in json.loads(file.read_text()).get('files', {}).items():
            if result['full_dataset_generated'] and result['items']==result['target_items']==14042 and result['all_source_prompts_gold_input_ids_raw_decoding_and_scores_verified']:
                mmmlu.append((name,result,file))
    if mmmlu:
        text += f'\n旧HF MMMLU完整语言审计：{len(mmmlu)}/14语言；目标196588题保持不变。\n\n'
        for name,result,file in mmmlu:
            text += f"- `{name}`：{result['correct']}/{result['items']}，SHA256 `{result['file_sha256']}`；审计`{file}`。\n"
        if len(mmmlu)<14:text += '\n旧HF完整14语言均值尚未成立；未完成部分不记为零分。\n'
    text += '\n只读塌缩诊断：diagnose_indexer_collapse.py使用CPT真实backbone的每语料8个固定seq128验证batch，将五种indexer应用于相同输入，先核对各run非indexer张量与CPT逐位相同。手算pre-ReLU点积得到的有效分数与模型逐位相同，零分数比例与原audit_stage独立结果一致。原始paper、paper-lr1e4的fully-dead eligible query比例分别为OWT .997934/.998967，Sh .996901/.998967；HF为.002066/.001033。失败分支norm有效增益仍为约.798–1.037及.978–1.012，不能归为norm增益归零或浮点下溢。全部checkpoint hash前后相同；证据runs/indexer-collapse-diagnostic.json含数据meta和实际batch IDs摘要。该结果定位保存状态的死ReLU区域，不证明造成该状态的优化因果过程；该诊断没有修复原失败run；后续新训练的验收另列。\n'
    text += '\n审计入口新增可选`--init runs/qwen-cpt/last.pt`：在原KL/非均匀/梯度门槛之前，核对完整distill阶段、CPT起点、全部非indexer张量逐位冻结、四个训练源hash、模型与optimizer张量有限性，并记录两个checkpoint hash和auditor hash。一个有效控制通过，backbone改动/optimizer NaN/源码错配/未完成四个无效控制均拒绝（runs/distillation-integrity-controls.json）。实际完成的paper-lr1e4分支在新入口验证通过完整性但仍失败非退化，结果runs/audit-qwen-paper-lr1e4-integrity.json；没有修改训练核心四文件或验收阈值。\n'
    text += '\n严格paper batch8新分支通过dense distillation验收：1000/1000步、1024000 tokens，公式保持原始sum(ReLU)，lr.001，microbatch1×accum8；seq128仍是Mac缩放。OWT KL .078189348 < 均匀 .168170625、非均匀query .955578512；Sh KL .102672565 < .374166284、非均匀query .946280992；各固定验证batch梯度为正。完整性审计确认CPT非indexer全部逐位冻结、模型/optimizer有限、四源hash不变。checkpoint SHA256 21b750734f34d8e0b0d21c8637aade929476ff0e75e1fcbf7c95614e8c127f00，证据runs/audit-qwen-paper-batch8.json。此前batch1失败及降学习率失败保留；新实验同时增加每步样本和总tokens，不能单独归因于batch，也没有用归一化参数映射或放宽门槛。\n\n由该通过分支启动独立runs/qwen-sparse-paper：profiles/paper_sparse.json预先固定8000步、1024000阶段tokens、seq128、microbatch1×accum1、lr.000025。沿用原Mac稀疏预算，每步1条仍不同于论文96条；不是生产tokens/性能等价。旧HF终点及所有既有benchmark结果保持其原checkpoint身份；新paper稀疏阶段尚未完成，也尚无新终点benchmark。\n'
    text += '\n同输入诊断扩展到已通过的paper batch8终点，仍逐位核对所有非indexer张量与CPT、有效分数与手算、零分数与原审计一致。fully-dead eligible query比例OWT .044421488、Sh .053719008（各968个eligible queries），对应正pre-ReLU点积比例 .244807168/.243505084；没有改变公式或归一化初值。结果runs/indexer-collapse-diagnostic.json现在包含5个固定indexer状态，全部checkpoint前后SHA相同。它证明新终点保留了更多可导区域，不隔离batch与总tokens的因果作用。\n'
    text += '\npaper稀疏第一段以time_budget正常保存3218/8000步（411904 tokens），checkpoint当时SHA256 ac7e183d95e07905da54df046fa5d918b06f875f944d1b8d55c775d30b6f748c，完整性runs/paper-sparse-round9-first-checkpoint.json：全部模型/optimizer有限，四源hash不变，193个非indexer及3个indexer张量相对通过的蒸馏起点发生更新。诊断runs/paper-sparse-round9-first-pruning.json：两个语料均72.65625% query真实裁剪，最大35 tokens，94.53125% query选至少2块。该步固定验证OWT/Sh LM7.033806/6.379099、训练口径selected-block KL .051930/.074188；同权重dense LM7.034747/6.374977。它是中途结果，不是8000步终点或质量达标。\n\n两条真实中途样本runs/paper-sparse-round9-first-{romeo,web}.json保留raw IDs、相同checkpoint hash、step3218/8000/phase_complete=false；每条100 tokens、温度.8、seed42、滑窗128、未输出未使用词表ID。文本不连贯且有数字/多语言碎片，不能称通用语言能力；这些样本不替换旧HF终点。该checkpoint已从3219步恢复，第二次调用840秒，完整8000步预算未改；rolling last.pt随后会更新，以上hash只对应本次保存/诊断/生成时点。\n'
    text += '\n第九轮安全停点：paper稀疏5481/8000步（701568 tokens，剩2519步），在2026-11-09 01:17:58.750244+08:00写入checkpoint并正常time_budget退出；最终确认/核验延迟到2026-11-09T01:40:03.852239+08:00，不将中间未观测时段称为已核实等待。checkpoint SHA256 215f308fe7f12833581997e153ec4d6420668f95c73911d641ff152cdd348717，模型/optimizer全部有限、四个训练源hash匹配；runs/round9-final.json。最新固定验证OWT/Sh LM6.922365/6.352720，selected-block KL.043831/.061721；同权重dense LM6.919913/6.346442。不是8000步终点。\n\n本轮模型及监测均已确认退出。5秒采样最高压力1，末次采样swap4091.88MiB；系统已有swap不可称零swap。00:47:58–最终核验窗口（包括训练停止后的时间）系统计数新增swap-out0、swap-in36814848 bytes；只限该窗口且不归因于单一进程。原HF终点hash未变，MMMLU完整3/14语言、德语停3040题，GPQA最近授权仍GatedRepoError。所有完整目标保持；goal active。\n'
    text += '\npaper严格打分分支完成8000/8000步、1024000阶段tokens：终点SHA256 1b0127c8d93ae879f041f202e789623de837a8fe7e56a557d06da98aa3afcb5e。训练四源hash及全部模型/optimizer有限性通过，runs/paper-final-checkpoint.json。固定验证OWT/Sh LM6.827170/6.343225、selected-block KL.043174/.059828；同权重dense LM6.828505/6.345107。全分布teacher审计是另一个口径：KL.059274/.087818 < uniform.318780/.427636，非均匀query .995868/.987603，各batch梯度正；runs/audit-paper-sparse-final.json。真实裁剪与两个终点100-token样本见runs/paper-sparse-final-{pruning,romeo,web}.json，不能把训练完成当作通用能力。batch/seq/代理数据/工程差异仍在profiles/paper_sparse.json。\n\n新终点推理33题×64，reference recurrent、chunk/grouped冷路径、subject-prefix缓存IDs一致，最大logits差8.59e-6；runs/paper-inference-equivalence.json。与旧HF比较只复用了固定输入fixture，未将旧HF输出充作新模型预测。新paper全量评测独立使用eval-paper-*前缀、profiles/paper_evaluation.json；旧eval-sparse-*及其分数均属于HF终点，不混合。新模型仍须完整评测，旧HF MMMLU剩余完整目标也保留；Goal active。\n'
    text += '\n第十轮最终停点：paper训练8000步及终点验收完成，paper MMLU保存11111/14042，剩2931题，状态interrupted_signal；runs/paper-mmlu-round10-prefix-audit.json对全部保存行的source/prompt/gold/IDs/raw/计分核验。首段02:55:00–03:15:01正常time_budget到11048；之后存在未观测空档，不称连续执行或verified wait。05:07:56未先检查截止时间即错误启动一次续跑，05:08:08已在完整batch SIGINT保存退出，新增63题原样保留，不隐瞒超预算执行。监测于03:40:14正常退出，所有模型/监测session均已确认terminal。准确下一条命令仍为上方同文件同源码--resume；complete训练不重跑，不缩减任何已声明目标。详见runs/round10-final.json。\n'
    text += '\n执行预算修复：run_with_deadline.py在exec前原生检查本轮带时区绝对截止，并按剩余时间保留180秒收尾、单调用最多1200秒；过期则拒绝启动。正负控制runs/deadline-launch-controls.json确认可用时间传入、过期时子进程完全未执行。第十一轮起使用此入口，避免只在较早的工具调用观察时钟而后无条件启动。该包装不改模型/训练/评测源码或总步数、数据、token预算。\n'
    paper_audits=[]
    for file in sorted(Path('runs').glob('paper-*-completed-audit.json')):
        for name,result in json.loads(file.read_text()).get('files', {}).items():
            meta=Path(name+'.meta.json')
            if not meta.exists():continue
            identity=json.loads(meta.read_text()).get('identity',{})
            if identity.get('checkpoint_sha256')!='1b0127c8d93ae879f041f202e789623de837a8fe7e56a557d06da98aa3afcb5e':continue
            if result['full_dataset_generated'] and result['all_source_prompts_gold_input_ids_raw_decoding_and_scores_verified']:
                paper_audits.append((name,result,file))
    if paper_audits:
        text += '\n严格paper终点的完整文字评测审计（不包括旧HF预测）：\n\n'
        for name,result,file in paper_audits:
            text += f"- `{name}`：{result['correct']}/{result['items']}，SHA256 `{result['file_sha256']}`；`{file}`。\n"
        text += '\n逐集合真实失败样例（取全量审计中保存的首个错误样例，不代表错误类型的频率；raw仅展示前240字符，完整输入/IDs/输出见原文件）：\n\n'
        for name,result,file in paper_audits:
            example=next((row for row in result.get('examples', []) if row['correct'] is False), None)
            if example is None:
                continue
            raw=example['raw_prediction'][:240].rstrip()
            fence='`'*max(3, 1+max((len(s) for s in re.findall(r'`+', raw)), default=0))
            text += f"`{name}`：task `{example['task']}`，index {example['index']}，gold {json.dumps(example['gold'], ensure_ascii=False)}，固定计分为错误。\n\n{fence}text\n{raw}\n{fence}\n\n"
    text += '\npaper GSM8K全1319题完成、固定数字抽取器命中6题，审计runs/paper-gsm8k-completed-audit.json，SHA256 8bfe9314200b9fd9de80d33dddb422a1785cbcd69e84dc700e06bcef15380f63。逐一检查六个命中，其raw都是重复短语并以10或100等数字命中，没有展示正确计算过程；runs/paper-gsm8k-output-diagnostic.json。保留原parser和6/1319分数，不把它解释为推理能力或算法优越性。\n\n代码评测只读审计已参数化：audit_code_results.py接受predictions/judged/checkpoint/source-archive/out，audit_multipl_generation.py接受predictions/checkpoint/source-archive/out。audit_generation_sources.py同时核对生成器、四个训练核心、chunk/公开缓存引用和prefix源码归档；旧EvalPlus542和MultiPL-E1277完整控制的实质字段与原审计完全一致，14条现代生成源指纹正控制通过，三类伪造源指纹均拒绝。证据runs/{evalplus,multipl}-parameterized-audit-control.json、generation-source-controls.json。未执行新模型代码生成/判分，旧HF预测、判分、归档和成绩不改。\n'
    mgsm_summary=Path('runs/paper-mgsm-summary.json')
    if mgsm_summary.exists():
        summary=json.loads(mgsm_summary.read_text())
        text += f"\npaper MGSM全11语言{summary['items']}题审计完成：{summary['correct']}正确，全部语言macro {summary['all_11_language_macro_accuracy']:.8f}、非英语10语言macro {summary['non_english_10_language_macro_accuracy']:.8f}；`{mgsm_summary}`，独立于旧HF结果。\n"
    text += '\nMultiPL-E执行审计也支持显式predictions/checkpoint/raw/scaffolds/summary/repaired-predictions/repaired-scores/source-archive/out。除原1277原始+1277脚手架的2554条完整执行记录，还独立核对3条修复预测输入IDs/原始解码、正确checkpoint、实际driver payload/program SHA、官方返回、镜像/资源/清理；按每题raw/修复结果重算八语言n、correct、修复数和macro。汇总必须绑定传入的同一raw/scaffold/修复预测文件，代码hash对固定Git归档、数据hash对原文件；不会因当前只读审计改动而篡改原历史源版本。完整旧记录的原结论一致，原协议full accuracy仍null，Mac修复macro0%。两个负控制wrong-checkpoint和wrong-summary均拒绝且不产出审计结果，runs/multipl-binding-controls.json；完整正控制runs/multipl-results-parameterized-control.json。只读取原执行产物，无VM启动、无候选代码宿主执行、未运行新paper代码评测。\n'
    text += '\n三修复题评分入口score_multipl_repairs.py已显式参数化predictions/checkpoint/raw/scaffolds/controls/scores-out/out/deadline。任何已有输出拒绝覆盖；在启动Sandbox前完整核对三题输入、ID/解码、原始1277行和scaffold集合、同一checkpoint、1024预算及六个控制；要求带时区deadline且至少500秒余量（3×90秒任务加收尾）。过期/错checkpoint/已有产物三个负控制均在Sandbox构造前拒绝，runs/multipl-scorer-preflight-controls.json。旧三条已审计执行的纯回放验证参数传递与原macro一致，且原score/summary哈希未变；runs/multipl-score-path-control.json明确标记no-new-sandbox、not-new-paper-score。修改入口后旧完整2557执行归档审计仍通过runs/multipl-legacy-after-scorer-cli.json。没有把mock回放计作新模型代码成绩，paper代码生成和隔离判分仍未执行。\n'
    bbh_audit=Path('runs/paper-bbh-completed-audit.json')
    if bbh_audit.exists():
        bbh=next(iter(json.loads(bbh_audit.read_text())['files'].values()))
        text += f"\npaper BBH全{bbh['items']}题审计完成：{bbh['correct']}正确，micro {bbh['micro_accuracy']:.10f}、27配置macro {bbh['bbh_27_configuration_macro_accuracy']:.10f}、23任务族macro {bbh['bbh_23_family_macro_accuracy']:.10f}；`{bbh_audit}`，独立于旧HF成绩。\n"
    text += '\npaper/HF分支可比性额外核验：compare_recipe_variants.py证明两次sparse阶段均8000步/1024000 tokens、四个训练源hash/语料身份/全部实际训练与验证超参数一致、最终独立数据RNG逐位相同；同seed和未变纯数据采样源码支持相同数据窗口顺序。两个蒸馏后起点的全部193个非indexer张量逐位相等。差异是score_mode和indexer初始状态，且HF蒸馏128000 tokens、paper batch8蒸馏1024000 tokens，完整训练历史并非matched-budget，不能称单变量打分消融或推断准确率/速度优势。证据runs/paper-hf-comparability.json含四checkpoint SHA和终点固定验证结果；没有重训模型或更改既有分数。\n'
    timing=Path('runs/paper-prefill-decode-ungrouped.json')
    if timing.exists():
        bench=json.loads(timing.read_text())
        text += '\n严格paper同权重CPU参考路径计时：warm weights、每次fresh cache、每形状3次、固定64输出（63次decode forward，不按EOS早停）；不是质量评分。表为双方关闭grouped GQA的对照，单位秒。\n\n|batch×prompt|Dense prefill|QSA prefill|Dense decode63|QSA decode63|\n|---|---:|---:|---:|---:|\n'
        for case in range(3):
            rows={r['stage']:r for r in bench['summaries'] if r['case']==case};d,q=rows['pretrain'],rows['sparse']
            text += f"|{d['batch_size']}×{d['prompt_tokens']}|{d['prefill_median_seconds']:.4f}|{q['prefill_median_seconds']:.4f}|{d['decode_median_seconds']:.4f}|{q['decode_median_seconds']:.4f}|\n"
        text += '\n18/18 timed trials完成；raw IDs、源码/输入fixture/checkpoint SHA在runs/paper-prefill-decode-ungrouped.json。另一个grouped配置仅优化sparse路径，输出IDs跨两配置一致，runs/paper-inference-timing-controls.json；第一版probe源码归档d7dbfc1。当前参考实现这三组QSA prefill慢而decode快，包含GR/MoE/head、Python及ungrouped sparse prefill的unused KL开销；不是孤立attention成本、纯算法优势或生产GPU sparse kernel加速。setup/tokenization不计入，主机其它应用不受控，累计RSS不是分stage内存比较。\n'
    text += '\n第十一轮最终停点：模型驱动07:07:00、监测07:07:47均正常exit0，原定07:09:46截止保护生效。人工最终确认较晚，按实际记录，不称连续人工等待。新增Pro26/12032、Redux2-ok0/5330、MATH0/5000均完整审计；INCLUDE保存10880/22639、剩11759，前缀source/IDs/raw/scoring审计通过。runs/round11-final.json及paper-include-round11-prefix-audit.json。paper/HF checkpoint及旧qwen_next_lab不变，Goal active，其余原完整评测目标未缩减。\n'
    text += '\n第十二轮按实际资源试运行两个CPU单线程推理进程：原文字队列之外，独立EvalPlus完整542题/768-token代码生成写eval-paper-code-full.jsonl，预声明profiles/paper_evalplus_generation.json，source archive15e8acc，stop-pressure2（文字仍4），相同paper终点，不启动VM。首段约110秒压力1、代码/文字各峰值RSS约0.906/1.838GB，新增swap-out0、swap-in131072B；只限该观察窗口，不相加RSS或倒推全轮无swap。生成正确性仍None，后续必须隔离判分及完整审计；该并行不改变任何完整数据或token目标。\n'
    code_generation_audit=Path('runs/paper-evalplus-generation-audit.json')
    if code_generation_audit.exists():
        generation=json.loads(code_generation_audit.read_text())
        text += f"\npaper EvalPlus完整生成已预检：{generation['items']}题/{generation['generated_tokens']} tokens、无未使用词表ID；`{code_generation_audit}`。实际judge原始输入预检通过后在Sandbox构造前退出，未执行任何候选代码，正确性全部pending，不能借用旧HF判分。\n"
    paper_code_audit=Path('runs/paper-evalplus-completed-audit.json')
    if paper_code_audit.exists():
        result=json.loads(paper_code_audit.read_text())
        text += '\n后续严格paper终点实际EvalPlus隔离判分及独立审计已完成（取代上述预检时的pending状态）：\n\n'
        for name, task in result['scores'].items():
            text += f"- {name}：base {task['base_correct']}/{task['n']}；plus（base与plus均通过）{task['plus_correct']}/{task['n']}。\n"
        text += f"\n四套均值{100*result['four_suite_mean']:.4f}%；证据`{paper_code_audit}`。完整542题、官方0.3.1测试输入、源码/镜像/执行记录及清理均独立核验；一题一次768-token贪心输出，属于本机缩小模型，不是官方Qwen成绩。\n"
        failed={}
        with Path(result['judged_path']).open() as stream:
            for line in stream:
                row=json.loads(line)
                if not row['plus_pass'] and not any(key[0]==row['dataset'] for key in failed):
                    failed[row['dataset'], row['task_id']]=row
                if len(failed)==2:
                    break
        examples={}
        with Path(result['predictions_path']).open() as stream:
            for line in stream:
                row=json.loads(line);key=(row['task'].removeprefix('evalplus-'), row['gold'])
                if key in failed:examples[key]=row
        for key,row in examples.items():
            raw=row['raw_prediction'][:240].rstrip();score=failed[key]
            fence='`'*max(3, 1+max((len(s) for s in re.findall(r'`+', raw)), default=0))
            text += f"\n实际代码失败样例`{key[0]}/{key[1]}`：base={score['base']['status']}，plus={score['plus']['status']}；base实际执行断言输入数{score['base']['tested_inputs']}。以下是原始生成前240字符，未用sanitizer输出替代：\n\n{fence}text\n{raw}\n{fence}\n"
    text += '\n第十二轮已正常有界停止：新增paper INCLUDE0/22639、SuperGPQA44/26529、MMMLU中文0/14042完整审计；paper MMMLU1/14，Arabic2771/14042前缀审计通过。模型09:27:27、监测09:28:15正常退出，早于09:30:15截止；人工最终确认较晚，按原时间线记录。两组checkpoint及旧lab保持不变；所有其它完整目标、代码隔离判分和gated/private缺口仍在。runs/round12-final.json。\n'
    multipl_generation=Path('runs/paper-multipl-generation-audit.json')
    if multipl_generation.exists():
        generation=json.loads(multipl_generation.read_text())
        text += f"\n严格paper MultiPL-E八语言完整生成：{generation['items']}题、{generation['generated_tokens']} tokens；全部原prompt/gold/input IDs/raw解码/官方stop处理及生成源码归档15e8acc已独立核对，`{multipl_generation}`。生成完成不代表测试通过，实际判分见runs/paper-multipl-raw-full.jsonl及meta；未完成全量执行审计前不报告新模型完整准确率。三个独立Mac修复题的3072 tokens另存eval-paper-multipl-repaired.jsonl，输入/IDs/raw/停止规则/源码检查见paper-multipl-repaired-generation-audit.json，不替换原1277题。原始三个无效fixture导致的full accuracy null与Mac修复组合仍须分开报告。\n"
    multipl_execution=Path('runs/paper-multipl-executions-audit.json')
    if multipl_execution.exists():
        execution=json.loads(multipl_execution.read_text())
        assert execution['all_program_inputs_generated_ids_official_returns_images_limits_driver_bytes_and_cleanups_verified']
        result=json.loads(Path(execution['summary_path']).read_text())
        text += f"\n后续严格paper MultiPL-E实际执行已全部通过独立审计：1277条新模型原始执行、1277条模型无关脚手架控制及3条新模型修复执行；`{multipl_execution}`。有效原始fixture {result['original_valid_correct']}/{result['original_valid_scaffolds']}，三个公开数据错误单列，原协议full accuracy仍为null。独立Mac修复组合八语言macro {100*result['mac_repaired_eight_language_macro']:.4f}%，不是官方Qwen分数或原发布协议。只复用模型无关scaffold/六个正负控制，没有复用旧HF预测或分数。任务VM已于11:01:46关闭。\n"
        invalid=set(map(tuple, result['original_ungradable_scaffolds']));failed={}
        with Path(execution['raw_path']).open() as stream:
            for line in stream:
                row=json.loads(line);key=(row['language'], row['name'])
                if not row['passed'] and key not in invalid:
                    failed.setdefault(row['language'], row)
        text += '\nMultiPL-E逐语言真实失败样例：取实际执行顺序中首个有效fixture的错误输出，非错误频率估计；raw仅展示前240字符。\n'
        with Path(execution['predictions_path']).open() as stream:
            for line in stream:
                row=json.loads(line);lang=row['task'].removeprefix('multipl-');score=failed.get(lang)
                if score is None or row['gold']!=score['name']:
                    continue
                raw=row['raw_prediction'][:240].rstrip()
                fence='`'*max(3, 1+max((len(s) for s in re.findall(r'`+', raw)), default=0))
                text += f"\n`{lang}/{row['gold']}`：官方执行状态`{score['status']}`，passed=false；不是基础设施异常或上述无效fixture。\n\n{fence}text\n{raw}\n{fence}\n"
    swap_window=Path('runs/round13-parallel-swap-window.json')
    if swap_window.exists():
        window=json.loads(swap_window.read_text());delta=window['system_swap_counter_delta_bytes']
        text += f"\n第十三轮CPU文字+MultiPL-E生成实测资源窗口：{window['start']}至{window['end']}，系统新增swap-out {delta['Swapouts']} bytes、swap-in {delta['Swapins']} bytes；文字/代码各自进程峰值RSS {window['text']['resources']['process_peak_rss_bytes']}/{window['code']['resources']['process_peak_rss_bytes']} bytes，进程采样压力最高均为1。这里只报告该窗口的host-wide计数，不归因于单个模型，也不代表全轮或已有swap为零；`{swap_window}`。\n"
    round_final=Path('runs/round13-final.json')
    if round_final.exists():
        final=json.loads(round_final.read_text())
        text += f"\n第十三轮已正常有界停止：新版EvalPlus全542及MultiPL-E全1277+3修复实际执行审计完成，paper MMMLU3/14完整，德语保存{final['paper_german_saved']}/14042、剩{final['paper_german_remaining']}，time_budget；完整保存前缀source/IDs/raw/score审计paper-mmmlu-de-round13-prefix-audit.json通过。驱动{final['model_driver_terminal_at']}、监测{final['monitor_terminal_at']}退出，早于11:44:27截止；人工确认较晚，不能称空档为连续人工verified wait。恢复后监测窗口系统新增swap-out0、swap-in1753088 bytes、压力最高1，仅限该窗口。两个checkpoint与旧lab未改；其余完整目标保留，Goal active；`{round_final}`。\n"
    hf_guard=Path('runs/round14-hf-german-second-state.json')
    if hf_guard.exists():
        guard=json.loads(hf_guard.read_text())
        if guard.get('early_pressure_stop'):
            text += f"\n第十四轮旧HF第二个并行调用在{guard['early_pressure_stop']}观测到系统压力>=2，外部保护对自己的模型进程SIGINT，完整batch保存13190/14042，正常exit0。该停点不是全量完成，剩852题；随后串行完成并通过mmmlu-de-completed-audit.json全量审计，旧HF德语0/14042，当前旧HF4/14。警告时没有关闭MPS保护或用户应用，也不把其它时段压力1误写为全程无警告。证据`{hf_guard}`；当时停止该分支，随后用串行方式完成其剩余题并恢复paper法语；当前保持单模型。\n"
    round14=Path('runs/round14-final.json')
    if round14.exists():
        final=json.loads(round14.read_text());window=final['resource_window']
        text += f"\n第十四轮已正常有界结束：新版MMMLU6/14、旧HF4/14分别完整审计，不相加；新版印地语time_budget保存12780/14042、剩1262，保存前缀逐题审计paper-mmmlu-hi-round14-prefix-audit.json通过。模型驱动{final['model_driver_terminal_at']}、监测{final['monitor_terminal_at']}退出，最终核验{final['confirmed_at']}，均早于14:01:07截止。系统采样窗口{window['start']}至{window['end']}新增swap-out {window['host_swap_delta_bytes']['Swapouts']} bytes、swap-in {window['host_swap_delta_bytes']['Swapins']} bytes，压力最高{window['max_sampled_pressure']}；保留并行警告及外部保护后串行的过程，不能称全轮压力始终1或零swap。两checkpoint及旧lab未改，VM关闭，所有剩余目标保留；`{round14}`，Goal active。\n"
    mmmlu_collection = Path('runs/paper-mmmlu14-collection-audit.json')
    if mmmlu_collection.exists():
        collection = json.loads(mmmlu_collection.read_text())
        assert collection['full_languages'] == 14 and collection['items'] == 196588 and collection['full_dataset_generated']
        text += f"\n## 新paper MMMLU完整14语言\n\n`{mmmlu_collection}`将每个预测文件hash与完整逐题source/prompt/gold/input IDs/raw/score审计绑定，14种语言各14042题，共{collection['items']:,}题，{collection['correct']}正确。语言macro={collection['language_macro_accuracy']:.6%}，micro={collection['micro_accuracy']:.6%}。全部属于paper checkpoint1b0127c8…，不混入旧HF。采用明确Mac协议：英语同科dev5-shot、greedy64、context8192；不是未公开Qwen私有harness。\n"
    component_counts = Path('profiles/component_counts.json')
    if component_counts.exists():
        components=json.loads(component_counts.read_text()); official=components['published_config_component_shapes']; mac=components['mac_untrained_component_defaults']
        text += f"\n## 视觉与MTP补充参数清单\n\n`component_inventory.py`仅用meta形状补齐主文字之外的组件：官方视觉{official['vision_parameters']:,}参数，各组件全部state形状与锁定HF上游一致；一层MTP含fusion/norm/QSA/MoE/GR的增量{official['incremental_mtp_parameters']:,}参数，与独立公式一致。主文字+视觉+MTP在明确共享主embedding/head假设下的逻辑重建总量{official['reconstructed_parameters_with_shared_mtp_vocab_tables']:,}。standalone vLLM draft另创建的输入embedding/untied head副本有{official['standalone_draft_input_and_untied_output_replica_elements']:,}元素，单列运行时副本，不自动算成新增学习参数。这个总量是公开架构形状及表共享假设下的重建，不是已逐tensor核验的发布权重archive或生产内存。\n\nMac默认视觉/MTP组件分别{mac['vision_parameters']:,}/{mac['incremental_mtp_parameters']:,}参数，仍未训练，未并入已完成32,978,962参数文字checkpoint。生产MTP/视觉训练loss、schedule、optimizer归属仍未知，不据此虚构训练或状态内存。完整配置、来源SHA和边界见`profiles/component_counts.json`，核对日志`runs/component-inventory.log`；官方config原始字节与MTP/HF源码先校验锁定SHA，profiles官方配置JSON值也逐项相等。\n"
    optimizer_counts = Path('profiles/optimizer_counts.json')
    if optimizer_counts.exists():
        values = json.loads(optimizer_counts.read_text()); mac=values['mac_trained_text']; official=values['official_text_shapes_with_declared_mac_ownership']
        text += f"\n## 主文字模型优化器状态核算\n\n`optimizer_inventory.py`读取8000步checkpoint的实际分片/缓冲区，并与冻结的`mac_optimizer_ownership.tsv`、meta模块形状和解析参数总量交叉核对。Mac全231个逻辑分片均已初始化：Muon753664、AdamW32157714、Adam67584个参数元素；FP32状态tensor共{mac['actual_state_tensor_bytes']:,} bytes，权重+梯度+状态核算小计{mac['replicated_fp32_weights_gradients_and_all_initialized_state_bytes']:,} bytes。这个小计不是RSS/MPS driver或统一内存实测，不含activation/临时buffer/Python对象。\n\n官方主文字形状按本机明确选择的相同ownership规则，仅用meta模块（零实际权重storage）核算：FP32状态{official['all_initialized_fp32_state_tensor_bytes']:,} bytes，FP32权重+梯度+状态小计{official['replicated_fp32_weights_gradients_and_all_initialized_state_bytes']:,} bytes。不是官方生产optimizer归属、BF16/master/sharded布局或实际工作集；MTP/vision与原parameter_counts一样单列而未加入主文字计数。完整分项/源hash见`profiles/optimizer_counts.json`，实际核对日志`runs/optimizer-inventory.log`；没有训练或推理官方大模型。\n"
    gpqa_audit = Path('runs/paper-gpqa-completed-audit.json')
    if not gpqa_audit.exists():
        gpqa_audit = Path('runs/paper-gpqa-saved-audit.json')
    if gpqa_audit.exists():
        result = json.loads(gpqa_audit.read_text())
        text += f"\n## GPQA已审计保存结果\n\n`{gpqa_audit}`：{result['items']}/{result['target_items']}，full={result['full_dataset_generated']}；所有来源prompt/gold、输入IDs、真实输出解码与作者计分复核通过。原始数据和生成只保留本机，不在报告中公开题目。\n\n|配置|已生成/目标|正确|准确率（仅已生成）|\n|---|---:|---:|---:|\n"
        for name, score in result['tasks'].items():
            accuracy = f"{score['accuracy']:.6%}" if score['accuracy'] is not None else '未生成'
            text += f"|{name}|{score['n']}/{score['target_items']}|{score['correct']}|{accuracy}|\n"
        text += '\nMain与Diamond有测试重叠，author与clean也不是独立题集；分别报告，不合并均值。未全量时只代表保存子集。\n'
    gpqa_performance = Path('runs/paper-gpqa-performance.json')
    if gpqa_performance.exists():
        p = json.loads(gpqa_performance.read_text()); window=p['resource_window']
        text += f"\nGPQA实际{p['actual_generated_tokens']:,}生成tokens，全部1292条都达到600-token上限，作者parser均未抽取到A–D答案；0分主要反映输出/格式失败，不能解释为可用科学推理能力。墙钟{p['wall_seconds']:.0f}s（含两段间保存/恢复），合计输出{p['aggregate_output_tokens_per_wall_second']:.2f} tokens/s，属于本机batch执行吞吐，非单请求延迟或生产kernel速度。进程峰值RSS {p['resources']['process_peak_rss_bytes']:,} bytes。资源采样窗口{window['start']}至{window['end']}，压力最高{window['max_sampled_pressure']}，host新增swap-out {window['host_swap_counter_delta_bytes']['Swapouts']} bytes、swap-in {window['host_swap_counter_delta_bytes']['Swapins']} bytes；不代表已有swap为零，也不归因于某个进程。\n"

    round16 = Path('runs/round16-final.json')
    if round16.exists():
        final = json.loads(round16.read_text()); window=final['resource_window']
        text += f"\n## 第十六轮有界执行结果\n\nGPQA四配置1292条已全部生成和审计；新增paper意大利语/日语两套完整MMMLU，paper共10/14语言、旧HF4/14独立保留。韩语time_budget保存{final['paper_korean_saved']}/14042、剩{final['paper_korean_remaining']}，保存前缀逐题source/IDs/raw/score审计通过，不能记为完整语言。模型子进程{final['model_child_terminal_at']}、原生监测{final['monitor_terminal_at']}均早于截止{final['deadline']}；最终快照确认{final['confirmed_at']}稍晚，不能称全部人工确认均在截止前。全部任务模型/驱动/监测已terminal；两checkpoint、核心源码和旧教学目录未变。\n\n本轮host采样窗口{window['start']}至{window['end']}，压力最高{window['max_sampled_pressure']}，新增swap-out {window['host_swap_delta_bytes']['Swapouts']} bytes、swap-in {window['host_swap_delta_bytes']['Swapins']} bytes；这不是零已有swap或进程归因。资源/完整性/原始续跑证据`runs/round16-final.json`、`runs/round16-integrity.json`、`runs/paper-mmmlu-ko-round16-prefix-audit.json`；准确续跑命令见CONTINUE。所有剩余完整目标和公开资料缺口保持，Goal active。\n"
    round17 = Path('runs/round17-final.json')
    if round17.exists():
        final = json.loads(round17.read_text()); window=final['resource_window']
        text += f"\n## 第十七轮有界执行结果\n\n本轮新增优化器实际状态/官方meta形状核算，并完成韩语、葡萄牙语及斯瓦希里语的全量审计。paper MMMLU完整语言{final['paper_mmmlu_full_languages']}/14；约鲁巴语状态{final['paper_yoruba_status']}，保存{final['paper_yoruba_saved']}/14042、剩{final['paper_yoruba_remaining']}，对应审计`{final['yoruba_audit']}`。完整14语言汇总只有通过独立collection门槛才在上方报告；旧HF完整{final['hf_mmmlu_full_languages']}/14及其剩余目标独立保留，不能借用paper预测。\n\n原生模型子进程{final['model_child_terminal_at']}、监测{final['monitor_terminal_at']}结束，原绝对截止{final['deadline']}；最终快照确认{final['confirmed_at']}，人工观测有间隔，不称持续人工监控或把确认时间改写成更早时刻。驱动/模型/监测均已确认terminal；两训练checkpoint、四个核心源码及旧教学目录未变。host采样窗口{window['start']}至{window['end']}，压力最高{window['max_sampled_pressure']}，新增swap-out {window['host_swap_delta_bytes']['Swapouts']} bytes、swap-in {window['host_swap_delta_bytes']['Swapins']} bytes；不是零已有swap或单进程归因。真实证据`runs/round17-final.json`、`runs/round17-integrity.json`及CONTINUE续跑命令；Goal active，尚未完成。\n"
        if final.get('hf_spanish_saved') is not None:
            text += f"\n同轮串行旧HF西班牙语保存{final['hf_spanish_saved']}/14042，剩{14042-final['hf_spanish_saved']}；保存前缀审计单列，未完成不能记作完整语言。\n"
    round18 = Path('runs/round18-final.json')
    if round18.exists():
        final=json.loads(round18.read_text()); window=final['resource_window']
        text += f"\n## 第十八轮有界执行结果\n\npaper约鲁巴语最后4条已补完，完整14语言196588条通过独立collection绑定审计，micro/语言macro均0%。原主paper驱动已结束，未重复训练或借用旧HF预测。旧HF现完整{final['hf_mmmlu_full_languages']}/14语言，最后{final['hf_last_language']}状态{final['hf_last_status']}，保存{final['hf_last_saved']}/14042、剩{final['hf_last_remaining']}；全部保存source/prompt/gold/输入IDs/raw/score核验`{final['hf_last_audit']}`通过，尚未完成的语言不记为全量或零分。\n\n原绝对截止{final['deadline']}；HF模型event结束{final['hf_model_event_end']}、子进程{final['hf_child_terminal_at']}、原生监测{final['monitor_terminal_at']}结束。最终快照确认{final['confirmed_at']}，人工观测有间隔，不称连续人工监控或把确认时间改早。本任务驱动/模型/监测均已实际确认terminal；两checkpoint、四训练核心和旧教学目录未变。host采样窗口{window['start']}至{window['end']}，压力最高{window['max_sampled_pressure']}，新增swap-out {window['host_swap_delta_bytes']['Swapouts']} bytes、swap-in {window['host_swap_delta_bytes']['Swapins']} bytes；不代表已有swap为零，也不归因于单个进程。原始证据`runs/round18-final.json`、`runs/round18-integrity.json`及CONTINUE准确续跑命令。原旧HF14语言目标与公开资料缺口保留，Goal active。\n"
    round19 = Path('runs/round19-final.json')
    if round19.exists():
        final=json.loads(round19.read_text()); window=final['resource_window']
        text += f"\n## 第十九轮有界执行结果\n\n既有paper14语言196588条完整结果保持独立。旧HF现完整{final['hf_mmmlu_full_languages']}/14语言，最后{final['hf_last_language']}状态{final['hf_last_status']}，保存{final['hf_last_saved']}/14042、剩{final['hf_last_remaining']}；全部保存输入/IDs/raw/score核验`{final['hf_last_audit']}`通过，未完成语言不算全量或零分。\n\n本轮21:18:51原生监测曾捕捉pressure2并安全SIGINT，保存9449条；停点完整审计/元数据/append-only前缀bytes+hash单列保存。最近60秒原生采样恢复1且实际启动门槛free56%/pressure1后，才在同一绝对deadline内手动恢复；新wrapper每5秒pressure>=2只停自己的driver，没有自动盲重启。最终已核对append-only前缀hash保持一致，不将原hash误当后续增长文件的整文件hash。警告不归因于某个用户应用，也不声称全轮压力始终1。\n\n原绝对截止{final['deadline']}；HF模型event结束{final['hf_model_event_end']}、子进程{final['hf_child_terminal_at']}、原生监测{final['monitor_terminal_at']}结束，最终快照确认{final['confirmed_at']}。本任务驱动/模型/恢复守护/监测均确认terminal；两checkpoint、四训练核心和旧教学目录未变。host窗口{window['start']}至{window['end']}压力最高{window['max_sampled_pressure']}，新增swap-out {window['host_swap_delta_bytes']['Swapouts']} bytes、swap-in {window['host_swap_delta_bytes']['Swapins']} bytes；不代表已有swap为零或单进程归因。原始证据`runs/round19-final.json`、`runs/round19-integrity.json`、`runs/round19-recovery.json`及CONTINUE准确续跑命令。Goal active，原完整目标与公开资料缺口保持。\n"
    pressure19=Path('runs/round19-pressure-stop.json')
    if pressure19.exists() and not round19.exists():
        warning=json.loads(pressure19.read_text())
        text += f"\n第十九轮进行中已有资源警告：{warning['at']}原生监测捕捉pressure{warning['pressure']}并SIGINT本任务，9449条保存前缀已审计；随后在60秒原生采样均1、启动实测free56%/pressure1后，在原23:16:09截止内手动同身份恢复。新wrapper保留5秒pressure>=2早停，原警告/前缀bytes+hash/元数据均保留；不称本轮压力全程1。证据`runs/round19-pressure-stop.json`、`runs/hf-mmmlu-round19-pressure-prefix-proof.json`、`runs/round19-recovery.json`；实际存活状态需另外核对进程。\n"
    hf_collection=Path('runs/hf-mmmlu14-collection-audit.json')
    if hf_collection.exists():
        full_hf=json.loads(hf_collection.read_text())
        assert full_hf['full_languages']==14 and full_hf['items']==196588 and full_hf['full_dataset_generated']
        assert full_hf['checkpoint_sha256']=='996f1abf6540637b63da742077db4bd7d49f2f13bbe247f8aca2710c3a072cb8'
        text += f"\n## 旧HF打分分支完整14语言\n\n`{hf_collection}`绑定全部14语言各14042条，共{full_hf['items']:,}条实际预测与逐题审计、原文件hash和各自历史生成源码归档。micro={full_hf['micro_accuracy']:.6%}，语言macro={full_hf['language_macro_accuracy']:.6%}；仅属于CP996f1abf…的HF indexer /√128变体，不混入paper CP1b0127c8…预测。前两语言的旧缓存版本与之后版本按原SHA保留，未强行迁移；Mac英语同科dev5/greedy64/context8192不是Qwen私有harness。\n"
    device_records={}
    for device in ['cpu','mps']:
        path=Path(f'runs/device-final-{device}.json')
        if path.exists():device_records[device]=json.loads(path.read_text())
    if set(device_records)=={'cpu','mps'} and all(item['status']=='passed' for item in device_records.values()):
        text += '\n## 明确设备与源码绑定的数值验收\n\n|模式|实际探针|用例|失败/错误/skip|秒|进程峰值RSS bytes|\n|---|---|---:|---|---:|---:|\n'
        for device,item in device_records.items():
            text += f"|{device}|{item['actual_probe_device']}|{item['tests_run']}|0/0/0|{item['seconds']:.3f}|{item['process_peak_rss_bytes']:,}|\n"
        text += '\n两模式均在评测模型已停的串行窗口完成，环境、LAB_TEST_DEVICE、源码SHA及实际MPS探针见runs/device-final-{cpu,mps}.json与对应.log。数据/codec/GPQA及FP64 oracle按设计使用CPU；模型数值测试使用选定设备，不称所有fixture均在GPU执行。MPS内存保护默认；这是数值验收，不替代剩余完整评测。\n'
    recovered19=Path('runs/round19-recovery.json')
    if recovered19.exists() and json.loads(recovered19.read_text()).get('early_pressure_stop'):
        event=json.loads(recovered19.read_text())['early_pressure_stop']
        text += f"\n第十九轮第二次资源早停：{event['at']}恢复wrapper捕捉pressure{event['pressure']}，安全保存韩语5160条并退出。停点元数据/前缀bytes+hash单列，之后先串行完成CPU/MPS52项验收；23:05恢复启动端实际free58%/pressure1，仍用原23:16:09截止和5秒pressure>=2早停。第二恢复状态runs/round19-recovery2.json；不将两次警告归因于某个用户应用。\n"
    text += '\n## 论文逐表验收边界\n\n完整11表核对见`PAPER_EXPERIMENT_AUDIT.md`。Table3 RULER/8-needle MRCR尚未实际评测，普通任务context8192不替代长上下文；Table4的MTP四步接受长度未测，组件正确性不替代训练后的speculative实验。其它官方规模架构/AttnRes/N-gram/batch-LR多组扫描也未复现。新增原作者小源码单独锁在long_context_sources.lock.json；锁源/读取元数据不算模型评测。全goal完成仍须逐项核对公开可行工作与缺口，不能从主任务完成直接推导。\n'
    round20=Path('runs/round20-final.json')
    if round20.exists():
        final=json.loads(round20.read_text())
        text += f"\n## 第二十轮有界执行结果\n\n旧HF完整{final['hf_mmmlu_full_languages']}/14语言，最后{final['hf_last_language']} {final['hf_last_status']}保存{final['hf_last_saved']}/14042、剩{final['hf_last_remaining']}；保存记录的原始输入/IDs/raw/score审计`{final['hf_last_audit']}`通过。11588条葡萄牙语压力停点前缀未改，完整14042审计覆盖原前缀。paper14/14及旧HF分支独立。全14旧HF集合审计：{final['hf_collection_audit']}；未完整时不输出集合均值。\n\n23:59压力2触发原HF与数据转换停止，随后串行CPU/MPS57各通过，在一分钟12次原生压力1及恢复启动实测free60%/pressure1后只恢复一个模型，数据转换保持停止；未关闭MPS保护。\n\n|采样窗口|样本数|最高采样压力|host swap-in bytes|host swap-out bytes|\n|---|---:|---:|---:|---:|\n"
        for window in final['resource_windows']:
            text += f"|{window['start']}至{window['end']}|{window['samples']}|{window['max_sampled_pressure']}|{window['host_swap_delta_bytes']['Swapins']}|{window['host_swap_delta_bytes']['Swapouts']}|\n"
        text += f"\n两个分离采样窗口不冒充跨停止/测试间隔连续观测，不作单进程/用户应用因果归因。HF最后模型event结束{final['hf_model_event_end']}，child结束{final['hf_child_terminal_at']}，实际最终确认{final['confirmed_at']}；原截止{final['deadline']}。本任务驱动/模型/监测/数据准备与设备验收均已停止，checkpoint和训练核心/旧lab未改，直接证据round20-final.json / round20-integrity.json。原CLI1200秒从加载后生成循环开始，先前调用有装载开销，不冒称所有进程墙钟≤20分钟；上下界见round20-call-wall-observations.json。00:42起增加外部1180秒SIGINT守护，信号和实际进程链验证在round20-call-wallguard.json。Goal active，MRCR实际生成、RULER及逐表缺口未由这些结果代替。\n"
    v2=[]
    for device in ('cpu','mps'):
        path=Path(f'runs/device-final-v2-{device}.json')
        if path.exists():
            result=json.loads(path.read_text())
            v2.append(f"|{device}|{result['status']}|{result['tests_run']}|{result['actual_probe_device']}|{len(result.get('imported_local_source_hashes',{}))}|{result['process_peak_rss_bytes']}|")
    if v2:
        text += '\n## 导入模块与依赖锁绑定的设备验收\n\n|模式|状态|实际执行数|探针设备|已绑定导入模块数|峰值RSS bytes|\n|---|---|---:|---|---:|---:|\n'+'\n'.join(v2)+'\n\n新device-final-v2-{cpu,mps}.json/.log记录原52项和MRCR5项、实际导入模块源码及uv.lock/pyproject指纹，串行且无评测模型；旧52项证据保留。数据/codec/FP64 oracle按设计CPU。该套件没有执行MRCR模型评测，新runner只有构造模型前的预检负控制；真实生成/续跑与独立审计仍待完成。\n'
    mrcr_progress=Path('data/mrcr-mac/progress.json')
    if mrcr_progress.exists():
        data=json.loads(mrcr_progress.read_text())
        text += f"\n## MRCR公开协议补充进度\n\nprofiles/mrcr_mac.json预声明raw≤500MB，固定8needle两分片461459875bytes哈希验证通过；不是完整MRCR全部needle配置下载。完整800小列审计及19份作者/模板/编译器小来源锁保留。数据准备当前{data['status']}，已处理{data['processed']}/800、Mac8K可容纳已选{data['selected']}，尚未完整冻结时不能启动模型或给MRCR分数。第一次0行索引解释失败保留，修正为原用户请求的下一条assistant后核对；没有修改原答案或删去needle。23:59的系统pressure2停点使数据和HF评测都停止，后续串行；原128K至1M长度、MTP接受长度与其它表边界见PAPER_EXPERIMENT_AUDIT。\n"
    text += '\n## 第二十一轮长上下文实际进度\n\n旧HF全14种语言已完成时，以上collection审计单独汇总，不与paper分支混用。MRCR原始800条全部核对，固定8K/1024输出的80条Mac-fit cohort；720条以精确token数或严格字节下界排除。独立全800重算见runs/mrcr-cohort-audit.json，峰值RSS1,713,061,888bytes。旧400判断回归一致，准备峰值1,749,696,512bytes，原2GB门槛保持。原失败/资源停点保留，不修改gold/needle或截断输入。\n'
    ruler_meta=Path('data/ruler-mac/meta.json')
    if ruler_meta.exists():
        data=json.loads(ruler_meta.read_text())
        text += f"\nRULER准备状态{data['status']}，已接受{len(data['cases'])}/26配置长度组合、{sum(x['rows'] for x in data['cases'].values())}/416条。原13任务、4K/8K、每组合16样本、seed42、作者base模板+answer_prefix/逐任务输出预算保持；局部tokenizer后端是已声明Mac适配。218份文章在锁定作者脚本中使用已验SHA本地bytes完整转换，derived3,108,621bytes；核心3文件74,001,217bytes、原文章4,492,086bytes和额外NLTK资源分别留账。120MB约束记录为落盘数据上限；首次Hotpot按60MB单文件门槛中断，已经读取60,000,001网络bytes另记在ruler-hotpot-intake-correction.json，不能称总网络下载≤120MB。英语NLTK zip下载4,319,076bytes只展开237,757bytes，主环境/训练依赖锁未改。生成器每子2GB、pressure2与原deadline保护，串行且与模型隔离。该416题诊断不是原128K–1M或官方full-sample RULER。\n"
    def file_sha(path):
        with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    table=[]
    for suite in ('mrcr','ruler'):
        audit_path=Path(f'runs/{suite}-paper-completed-audit.json')
        audit=json.loads(audit_path.read_text()) if audit_path.exists() else {}
        for stage in ('sparse','dense'):
            prediction=f'runs/{suite}-paper-{stage}.jsonl';path=Path(prediction+'.meta.json')
            if not path.exists():continue
            data=json.loads(path.read_text());entry=audit.get('files',{}).get(prediction)
            verified=bool(entry and audit['auditor_sha256']==file_sha(f'audit_{suite}_results.py') and entry['sha256']==file_sha(prediction) and entry['meta_sha256']==file_sha(path))
            if verified and entry['complete_mac_cohort']:
                score=f"{entry['mean_author_score']:.6f}" if suite=='mrcr' else str(entry['author_summary']['configuration_macro_percent'])
            else:score='不报完整分数；等待完整生成/绑定审计'
            table.append(f"|{suite}/{stage}|{data['status']} {data['completed_items']}/{data['target_items']}|{verified}|{score}|{data.get('process_peak_rss_bytes','运行中')}|")
    if table:text += '\n|套件/同权重模式|状态|独立完整绑定审计|作者指标（MRCR0–1；RULER每长度13配置宏均值%）|调用峰值RSS bytes|\n|---|---|---|---|---:|\n'+'\n'.join(table)+'\n'
    proof=Path('runs/mrcr-sparse-resume-proof.json')
    if proof.exists():
        data=json.loads(proof.read_text());text += f"\nMRCR实际20秒停点保存3条完整记录和288-token未评分草稿；续跑后原{data['completed_prefix_bytes']}bytes前缀SHA不变，草稿完整288 IDs与续跑对应行前缀相同（mrcr-sparse-resume-proof.json）。原copy-load两条sparse长输入32-token等价检查通过，但同进程整体dense检查及独立dense/mapped尝试曾在pressure2退出，日志完整保留，不能说所有长输入数值检查通过。真实生成只采用同身份frozen runner，所有原始IDs/raw/response均保存；没有fused稀疏kernel或加速比声明。\n"
    v3=[]
    for device in ('cpu','mps'):
        path=Path(f'runs/device-final-v3-{device}.json')
        if path.exists():
            data=json.loads(path.read_text());v3.append(f"|{device}|{data['status']}|{data.get('tests_run','—')}|{data['actual_probe_device']}|{data['process_peak_rss_bytes']}|")
    if v3:text += '\n新增CapacityBound两项后当前完整设备验收（与模型串行）：\n\n|模式|状态|测试数|实际设备|峰值RSS bytes|\n|---|---|---:|---|---:|\n'+'\n'.join(v3)+'\n\n59项含旧57与两项byte-bound数值/不安全配置拒绝控制；实际导入模块及source/dependency SHA在device-final-v3 JSON。套件不替代长上下文真实生成和逐条审计。\n'
    shared=Path('runs/shared-dense-long-proof.json');longest=Path('runs/shared-dense-longest-attention-proof.json')
    if shared.exists() and longest.exists():
        first=json.loads(shared.read_text());second=json.loads(longest.read_text())
        assert first['sources']['shared_kv_inference.py']==second['sources']['shared_kv_inference.py']==file_sha('shared_kv_inference.py')
        text += '\nDense额外使用shared_kv_inference.py按查询head分组直接收缩共享K/V，避免原head-expanded参考路径的KV复制；模型、权重和dense数学不变。真实6496-token输入的末prefill块attention及32步完整模型缓存续解码与原实现逐位相同；7163-token最长输入末块attention最大差0（longest-attention-proof.json，峰值RSS1,927,430,144bytes，结束pressure1）。同进程两长样本/最长32-token原参考续解码曾触发pressure2，均保留失败/部分证据；不能把它们改称全通过，也不能宣称该CPU收缩就是fused稀疏kernel。原MRCR sparse入口和已生成meta指纹保持，dense独立入口evaluate_mrcr_dense.py；RULER同一入口按stage选择。\n'
    v4=[]
    for device in ('cpu','mps'):
        path=Path(f'runs/device-final-v4-{device}.json')
        if path.exists():
            data=json.loads(path.read_text());v4.append(f"|{device}|{data['status']}|{data.get('tests_run','—')}|{data['actual_probe_device']}|{data['process_peak_rss_bytes']}|")
    if v4:text += '\n共享KV新增缓存/因果mask/多KV head oracle后最新完整套件（旧59/57/52证据保留）：\n\n|模式|状态|测试数|实际设备|峰值RSS bytes|\n|---|---|---:|---|---:|\n'+'\n'.join(v4)+'\n\n60项在CPU/MPS分别执行，实际source/dependency哈希见device-final-v4 JSON；没有用CPU替代MPS，也未关闭默认内存保护。\n'
    ruler_resume=Path('runs/ruler-sparse-resume-proof.json')
    if ruler_resume.exists():
        data=json.loads(ruler_resume.read_text());text += f"\nRULER实际20秒预算调用保存17条，续跑后原{data['completed_prefix_bytes']}bytes前缀SHA不变；第一未完行只有prefill、0草稿token，所以不宣称非空token草稿重放。真实MRCR两模式的288/768-token重放证据独立保留。统一守护run_long_context_guarded.py在真实global-lock忙碌和expired deadline下拒绝造第二模型（longctx-guard-*-control-v2-state.json），并在RULER正向调用保留source SHA/资源/terminal状态。原03:19整轮截止和每call1180s SIGINT、1198s强制墙钟余量保持；不关闭MPS保护，不自动重启资源停点。\n"
    public=Path('data/ruler-public500-repair/meta.json') if Path('data/ruler-public500-repair/meta.json').exists() else Path('data/ruler-public500/meta.json')
    if public.exists():
        data=json.loads(public.read_text());text += f"\n## 公开默认500样本的独立继续评测\n\n原profiles/ruler_public500.json先声明全500目标，后发现原作者4K Hotpot在index251处used_docs27→17→7导致negative sample无限重试，原12文件失败停点保留。独立profiles/ruler_public500_repair.json只把QA重试下限钳到10，原10个文档、原题、长度及预算保持；前500题10文档含输出最大3416tokens，修复版与未改原作者必须分开。修复配置仍为：13配置×4K/8K×500=13000条/模式；与旧16条诊断分别保存。准备当前{data['status']}，已保存{len(data['cases'])}/26个500-row文件；压缩数据{sum(c['compressed_bytes'] for c in data['cases'].values()):,}bytes、对应原始JSONL {sum(c['uncompressed_bytes'] for c in data['cases'].values()):,}bytes，逐文件前16条与旧数据原始bytes一致。原始公开输入仍沿用固定core3/218文章，无新增下载；无损gzip压缩数据上限64MB、临时未压缩数据32MB、准备RSS2GB保持；每轮统一截止见CONTINUE及各轮原始state，不跨调用重置。不能把压缩后的bytes说成未压缩数据量。\n"
        public_table=[]
        for stage in ('sparse','dense'):
            meta_path=Path(f'runs/ruler500-paper-{stage}.jsonl.meta.json')
            if meta_path.exists():
                item=json.loads(meta_path.read_text());peak=item.get('process_peak_rss_bytes','未记录');peak=f'上次停点 {peak}' if item['status']=='running' else peak;score='完整13000及独立审计前不报宏均值'
                if item['status']=='complete':
                    for audit_path in Path('runs').glob('ruler500-round*-final-prefix-audit.json'):
                        checked=json.loads(audit_path.read_text());entry=checked.get('files',{}).get(str(meta_path).removesuffix('.meta.json'))
                        if entry and entry['complete_mac_cohort'] and entry['meta_sha256']==file_sha(meta_path):
                            assert checked['auditor_sha256']==file_sha('audit_ruler_public500.py') and entry['sha256']==file_sha(str(meta_path).removesuffix('.meta.json'))
                            score='Mac作者宏均值% '+str(entry['author_summary']['configuration_macro_percent']);break
                public_table.append(f"|{stage}|{item['status']}|{item['completed_items']}/13000|{peak}|{score}|")
            else:public_table.append(f"|{stage}|未启动|0/13000|—|无分数|")
        text += '\n|模式|状态|实际保存/目标|调用峰值RSS bytes|指标边界|\n|---|---|---|---:|---|\n'+'\n'.join(public_table)+'\n\n使用流式逐行编码/生成/续跑验证，不把13000条prompt/input IDs一起装入内存。新的公共默认样本量仍只覆盖Mac4K/8K，不是论文128K至1M分桶或未公开Qwen私有harness。Goal保持active；部分进度不算完成。\n'
    mtp_records=[]
    for device in ('cpu','mps'):
        path=Path(f'runs/mtp-incremental-v2-{device}.json')
        if path.exists():
            data=json.loads(path.read_text());mtp_records.append(f"|{device}|{data['status']}|{data.get('tests_run','—')}|{data['actual_probe_device']}|{data['process_peak_rss_bytes']}|")
    if mtp_records:
        text += '\n## 第二十二轮MTP增量组件\n\n`mtp_inference.py`补充dense/sparse增量KV、冻结step0选择时仍写indexer缓存、回滚后重填、等长请求压缩及packed valid-count检查。完整proposer、拒绝/bonus/EOS流程及MTP训练仍未完成，没有接受长度成绩。Table4与Figure6设置不能混用，见MTP_PROTOCOL_AUDIT。\n\n|模式|状态|新增针对性测试数|实际设备|峰值RSS bytes|\n|---|---|---:|---|---:|\n'+'\n'.join(mtp_records)+'\n\n旧完整60项与新增4项是不同执行记录，不合称一次完整64项。v1因argsort参数写法失败的四错误保留，v2修正dim关键字后通过；失败源码按原记录SHA恢复并留档。新增源码固定同一vLLM revision的88720bytes，仅源码，不是数据/大权重下载。\n'
    pressure_audits=[]
    for name in ('runs/ruler500-round22-pressure-prefix-audit.json','runs/ruler500-round22-pressure2-prefix-audit.json'):
        path=Path(name)
        if path.exists():
            audit=json.loads(path.read_text());entry=audit['files']['runs/ruler500-paper-sparse.jsonl'];pressure_audits.append(f"{entry['items']}条（{name}）")
    if pressure_audits:text += '\n本轮public500在native pressure2分别保存84/555条，两个模型及驱动均退出；已独立审计保存前缀：'+ '、'.join(pressure_audits)+'。各恢复均绑定当时meta/predictions/audit SHA、12次压力1采样及旧PID不存在证据，资源停点不自动重启。它们不代表全轮压力始终1，也不归因于用户应用；主训练权重未变。\n'
    final_path=Path('runs/round22-final.json')
    if final_path.exists():
        final=json.loads(final_path.read_text());stats=json.loads(Path('runs/ruler500-round22-prefix-statistics.json').read_text())
        entry=final['public500_final_prefix_audit']['files']['runs/ruler500-paper-sparse.jsonl']
        assert entry['sha256']==stats['prediction_sha256']
        assert entry['meta_sha256']==stats['meta_sha256']
        assert final['public500_final_prefix_audit']['sha256']==file_sha(final['public500_final_prefix_audit']['file'])
        text += f"\n## 第二十二轮收尾验收快照（{final['confirmed_at']}）\n\n主训练权重和训练核心SHA复核保持。全13000条源预算独立验证通过；实际sparse前缀{entry['items']}/13000条全部逐条绑定prompt、gold、input/output IDs、raw/response、stop及作者计分，宏均值仍为{entry['author_summary']}。完整dense尚未开始。与旧16条诊断对应的{len(stats['matched_old16_diagnostic_questions'])}个问题输入及输出完全一致；84/555/3062条停点原bytes保持，3062时间停点64-token草稿重放一致。\n\n保存输出共{stats['output_tokens']:,}tokens；行内耗时（含prefill/解码/计分）中位数{stats['row_seconds_p50']:.4f}s、prefill中位数{stats['prefill_seconds_p50']:.4f}s，输出tokens/记录行内秒{stats['output_tokens_per_generation_second']:.2f}。此速率属于本机CPU串行当前前缀，不是稀疏kernel加速比，也不包含调用加载和源验证等墙钟成本。\n"
        text += '\n|资源采样文件|窗口起止|样本数|最大原生压力|采样child RSS最大bytes|host Swapins/Swapouts增量bytes|\n|---|---|---:|---:|---:|---|\n'
        for window in final['resource_windows']:
            if window['file'].startswith(('runs/round22-ruler500','runs/ruler500-sparse-round22')):
                text += f"|{window['file']}|{window['first_observed']} → {window['last_observed']}|{window['samples']}|{window['max_sampled_pressure']}|{window['sampled_child_rss_max_bytes']}|{window['host_swap_delta_bytes']}|\n"
        text += '\n只对上述实际采样窗口报告host swap，不归因于用户应用或称全轮连续无压力。两次pressure2保存及后续逐次恢复证据独立保留。真实输出示例（原始全部IDs/raw在ignored预测文件）：\n'
        for example in stats['real_examples']:
            text += f"\n- {example['source_id']}，{example['output_tokens']}tokens，作者score={example['score']}：{json.dumps(example['response'][:160],ensure_ascii=False)}\n"
        text += '\nGoal保持active；完整13000×2、原长上下文协议、MTP训练及接受长度等缺口仍在。精确后续命令见CONTINUE.md，不能以收尾或局部审计代替完成。\n'
    greedy_rows=[]
    for device in ('cpu','mps'):
        path=Path(f'runs/speculative-mtp-v1-{device}.json')
        if path.exists():
            data=json.loads(path.read_text());assert data['status']=='passed' and all(file_sha(name)==sha for name,sha in data['source_hashes'].items())
            greedy_rows.append(f"|{device}|{data['status']}|{data['tests_run']}|{data['actual_probe_device']}|{data['process_peak_rss_bytes']}|")
    if greedy_rows:
        text += '\n## 第二十三轮MTP贪心解码验收\n\n新增speculative_mtp.py实现B1贪心提案、一次批量target verification、拒绝/bonus/EOS/输出预算处理。拒绝后恢复GDN cache并重放已接受前缀，MTP回滚后以真实target hidden重填；额外成本不称CUDA/paged kernel加速。五项双设备测试涵盖独立full Model.forward teacher、全接受及每个拒绝位置、hidden/token对齐、EOS/预算与非有限draft拒绝。\n\n|设备|状态|新增测试数|实际device|峰值RSS bytes|\n|---|---|---:|---|---:|\n'+'\n'.join(greedy_rows)+'\n\n旧完整60项、增量4项及新增贪心5项是三套独立执行记录，不合称一次完整69项。固定vLLM proposer、runner、hook与stats的11份源码294146bytes、通用Canzona MTP训练文档/模块/测试61127bytes以及loader/config/作者测试98488bytes独立锁定；隔离执行四个Qwen配置适配分支通过，hc_count→hc_mult有来源；通用loss默认不是Qwen3.8生产配方。本机sparse明确启用论文讨论的索引复用；固定vLLM还有独立option/PCP门控，发布config无该字段。公开vLLM日志mean=1+accepted drafts/rounds包含bonus，EOS/预算实际输出长度单独记录，不能据此断言Table4原harness一致。\n'
        for device in ('cpu','mps'):
            data=json.loads(Path(f'runs/mtp-real-integration-{device}.json').read_text());assert data['status']=='passed' and data['draft_trained'] is False
            matches=all(row['exact_teacher_sequence_match'] for row in data['stages'].values())
            text += f"\n真实接线{device}（实际{data['actual_probe_device']}）：主8000步checkpoint+未训练draft；dense/sparse各24tokens与独立teacher完全匹配={matches}；sparse首轮64历史选32，draft接受数均0。峰值RSS{data['process_peak_rss_bytes']}bytes，结束压力{data['ending_pressure']}；主参数RAM前后SHA保持。原始IDs/raw、提案/验证和额外重放次数在mtp-real-integration-{device}.json。它是接线证明，不是Table4、训练MTP或加速成绩。\n"
    round23_path=Path('runs/round23-final.json')
    if round23_path.exists():
        latest=json.loads(round23_path.read_text());progress=json.loads(Path('runs/ruler500-round23-prefix-statistics.json').read_text());latest_entry=latest['public500_final_prefix_audit']['files']['runs/ruler500-paper-sparse.jsonl']
        assert latest_entry['sha256']==progress['prediction_sha256'] and latest_entry['meta_sha256']==progress['meta_sha256']
        assert latest['public500_final_prefix_audit']['sha256']==file_sha(latest['public500_final_prefix_audit']['file'])
        dense_latest=latest['public500_evaluations']['dense'];latest_score=latest_entry['author_summary']
        text += f"\n## 第二十三轮收尾快照（{latest['confirmed_at']}）\n\n原05:15:13至07:15:13预算保持；实际sparse {progress['items']}/13000，dense {dense_latest['saved']}/13000，sparse完整author_summary={latest_score}。全13000源及当前全部输出前缀独立逐条审计通过；真实输出{progress['output_tokens']:,}tokens，行内耗时（含prefill/解码/计分）p50={progress['row_seconds_p50']:.4f}s、prefill p50={progress['prefill_seconds_p50']:.4f}s，输出token/行内秒={progress['output_tokens_per_generation_second']:.2f}。这不包括装载和续跑审计墙钟，也不是kernel加速比。与旧16诊断对应{len(progress['matched_old16_diagnostic_questions'])}个问题输入/输出一致。原6612及7368条bytes保持；两个停点均没有新的非空草稿，原3062处64-token重放证据独立保留。\n"
        text += '\n|本轮资源采样文件|窗口|样本数|最大压力|采样child RSS最大bytes|host Swapins/Swapouts增量bytes|\n|---|---|---:|---:|---:|---|\n'
        for window in latest['resource_windows']:
            if window['file'].startswith('runs/round23-ruler500'):
                text += f"|{window['file']}|{window['first_observed']} → {window['last_observed']}|{window['samples']}|{window['max_sampled_pressure']}|{window['sampled_child_rss_max_bytes']}|{window['host_swap_delta_bytes']}|\n"
        text += '\n只报告实际窗口，MTP测试/probe的峰值另列；不把采样间隙称为连续观测。所有模型及guard实际退出、核心/主权重及原教学lab保持、数据/权重未追踪。Goal active，后续精确命令见CONTINUE.md；完整13000×2、训练MTP及原harness/论文实验缺口仍未完成。\n'
    round24_path=Path('runs/round24-final.json')
    if round24_path.exists():
        final24=json.loads(round24_path.read_text());snapshot=json.loads(Path('runs/ruler500-round24-statistics.json').read_text())
        assert snapshot['audit_sha256']==file_sha(snapshot['audit_file'])
        assert final24['ruler500_statistics']['sha256']==file_sha('runs/ruler500-round24-statistics.json')
        text += f"\n## 第二十四轮收尾快照（{final24['confirmed_at']}）\n\n本轮统一预算07:13:31至09:13:31。以下两种模式分别统计，不混入旧16诊断；只在完整13000且独立逐条审计通过后显示作者宏均值。它仍是4K/8K的显式QA retry-clamp Mac修复variant，不是论文128K–1M或未修改作者harness。\n\n|模式|实际状态/目标|独立审计后的作者宏均值%|累计输出tokens|行内耗时p50秒|输出tokens/行内秒|最后调用peak RSS bytes|\n|---|---|---|---:|---:|---:|---:|\n"
        stage_details=[]
        for filename,mode24 in snapshot['files'].items():
            assert mode24['sha256']==final24['public500_final_prefix_audit']['files'][filename]['sha256']
            score=str(mode24['author_summary']['configuration_macro_percent']) if mode24['complete_mac_cohort'] else '未完整，不报宏均值'
            row_time='—' if mode24['row_seconds_p50'] is None else f"{mode24['row_seconds_p50']:.4f}"
            rate='—' if mode24['output_tokens_per_row_second'] is None else f"{mode24['output_tokens_per_row_second']:.2f}"
            text += f"|{mode24['stage']}|{mode24['status']} {mode24['items']}/13000|{score}|{mode24['output_tokens']:,}|{row_time}|{rate}|{mode24['last_call_peak_rss_bytes']}|\n"
            stage_details.append(f"{mode24['stage']}与旧16诊断对应的{len(mode24['matched_old16_questions'])}个问题输入/输出完全一致；此前审计前缀按原bytes保持：{ {key:value['items'] for key,value in mode24['preserved_audited_prefixes'].items()} }。")
        text += '\n'+'\n\n'.join(stage_details)+'\n'
        text += '\n本轮真实生成节选（raw前160字符；完整raw与output IDs在对应统计/预测文件）：\n'
        for mode24 in snapshot['files'].values():
            for example in mode24['real_examples'][:1]:
                text += f"\n- {mode24['stage']} {example['source_id']}，{len(example['output_ids'])}tokens，作者score={example['score']}：{json.dumps(example['raw'][:160],ensure_ascii=False)}\n"
        text += '\nDense首16相同问题真实输出回归通过；首1025条时间停点55,293,302bytes原SHA保持，草稿token为0，不宣称非空重放。\n'
        draft24=final24['saved_native_resume_metadata'].get('dense',{}).get('unscored_incomplete_row')
        if draft24:
            text += f"\n本轮最终Dense未评分草稿为{draft24['source_id']}，已保存{len(draft24['output_ids'])}个token；它不计入完成条数，下一轮须从最终native元数据验证重放。本轮尚未执行该草稿的续跑验证。\n"
        text += '\n行内时间包含prefill/解码/计分，不含每行tokenization/JSON写盘、模型装载及续跑检查；两模式当前可能覆盖不同任务/长度，未配对吞吐不能直接作速度比，也不能据此称fused稀疏kernel加速。ruler500_statistics.py对正在写入的真实模型结果的拒绝控制通过，正向统计使用本轮完整绑定审计。原始输入/IDs/raw/score与生成示例保留在ignored预测/统计文件，不上传。\n\n|本轮资源窗口|样本数|最大压力|采样child RSS最大bytes|host Swapins/Swapouts增量bytes|\n|---|---:|---:|---:|---|\n'
        for window in final24['resource_windows']:
            if window['file'].startswith('runs/round24-ruler500'):
                text += f"|{window['first_observed']} → {window['last_observed']} ({window['file']})|{window['samples']}|{window['max_sampled_pressure']}|{window['sampled_child_rss_max_bytes']}|{window['host_swap_delta_bytes']}|\n"
        text += '\n资源只覆盖上述实际采样窗口，不归因于用户应用或覆盖间隙；MTP训练/原Table4、原长上下文和其它论文实验缺口仍保留。Goal active，准确后续命令见CONTINUE。\n'
        if 'pressure_stop_recovery' in final24:
            recovery=final24['pressure_stop_recovery'];text += f"\n本轮{recovery['items']}条时native memory_pressure/guard pressure2保存退出；初次审计在资源联合断言处失败，原log保留（{recovery['initial_failed_audit_log']}）。12次恢复采样全pressure1、旧PID不存在，并重新完成完整源/前缀审计和统计后，才用{recovery['review_file']}一次SHA绑定review恢复。没有改写原status或重置09:13:31截止；不归因于用户应用。\n"

    latest_text,latest_row=latest_ruler_snapshot();text+=latest_text
    overview = '\n## 当前验收摘要\n\n|范围|实际状态|直接证据|\n|---|---|---|\n'
    endpoint=Path('runs/paper-final-checkpoint.json')
    if endpoint.exists():
        phase=json.loads(endpoint.read_text())
        overview += f"|严格paper主文字训练|{phase['step']}/{phase['target_steps']}步；非退化与实际历史裁剪审计通过|`runs/paper-final-checkpoint.json`、`runs/audit-paper-sparse-final.json`|\n"
    collection=Path('runs/paper-mmmlu14-collection-audit.json')
    if collection.exists():
        full=json.loads(collection.read_text())
        overview += f"|主paper MMMLU|{full['full_languages']}/14语言，{full['items']:,}条；micro/语言macro均{full['micro_accuracy']:.2%}|`runs/paper-mmmlu14-collection-audit.json`|\n"
    old_files={name:result for name,result,_ in mmmlu}
    overview += f"|旧HF MMMLU|{len(old_files)}/14完整语言|`runs/hf-mmmlu14-collection-audit.json`及下方独立审计表|\n"
    if table:overview += '|MRCR/RULER Mac长上下文|见末尾逐模式真实生成/独立审计表；与原128K–1M不同|第二十一轮长上下文实际进度|\n'
    if final_path.exists():overview += f"|第二十二轮public500 Mac QA修复variant快照|sparse {entry['items']}/13000逐条审计；dense0/13000；完整均值null|`runs/ruler500-round22-final-prefix-audit.json`|\n|MTP新增增量组件|增量4项及贪心5项双设备分别通过；训练/Table4成绩未完成|`runs/mtp-incremental-v2-cpu.json`、`runs/mtp-incremental-v2-mps.json`|\n"
    if round23_path.exists():overview += f"|第二十三轮public500快照|sparse {progress['items']}/13000逐条审计；dense {dense_latest['saved']}/13000；完整分数见审计表|`runs/ruler500-round23-final-prefix-audit.json`|\n"
    if round24_path.exists():
        state_summary='；'.join(f"{entry['stage']} {entry['items']}/13000 {'完整审计' if entry['complete_mac_cohort'] else '部分审计'}" for entry in snapshot['files'].values())
        overview += f"|第二十四轮public500快照|{state_summary}|`runs/ruler500-round24-final-prefix-audit.json`、`runs/ruler500-round24-statistics.json`|\n"
    overview += latest_row+'\n两分支不混用预测或成绩。MTP/视觉组件未训练；生产loss/schedule、分布式/CUDA工程和私有SWEBench-Pretrain等缺口见矩阵。所有准确率属于已声明Mac规模/代理训练数据/公开候选协议，不是官方Qwen成绩。下文保留历史阶段与失败证据，当前进度以本表和末尾最新轮次为准。\n\n'
    text=text.replace('## 环境、来源与配置',overview+'## 环境、来源与配置',1)
    text+=completed_dense_pairs()
    # Keep the explicitly dated independent-control results when refreshing the
    # original recipe report. Their immutable receipts live outside this renderer.
    previous=Path('RESULTS.md').read_text() if Path('RESULTS.md').exists() else ''
    control_results=re.search(r'<!-- QWEN35_RESULTS_START -->.*?<!-- QWEN35_RESULTS_END -->',previous,re.S)
    if control_results:
        heading,body=text.split('\n',1)
        text=heading+'\n\n'+control_results.group(0)+'\n'+body
    Path('RESULTS.md').write_text(text)


if __name__=='__main__': main()
