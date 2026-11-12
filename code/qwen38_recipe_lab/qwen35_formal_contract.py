"""Fixed Mac formal budgets; progress lives in receipts, not this declaration."""
from pathlib import Path
from run_dense_text_guarded import load,digest

BUDGET='profiles/qwen35_formal_budget.json'


def contract(profile,phase):
    assert phase in ['formal_short','formal_long'] and profile['stage']==phase
    assert profile['run_class']=='formal_predeclared_continuation'
    assert profile['formal_budget_manifest']==BUDGET and digest(BUDGET)==profile['formal_budget_sha256']
    budget=load(BUDGET);entry=next(row for row in budget['arms'] if row['arm']==profile['arm']);stage=budget['short' if phase=='formal_short' else 'long']
    assert profile['steps']==stage['total_steps'] and profile['seq']==stage['seq'] and profile['batch']==stage['batch']
    assert profile['device']=='cpu' and profile['config']==budget['config'] and profile['data']==budget['data']
    for key,value in budget['mac_declared_settings'].items():assert profile[key]==value,key
    assert profile['source_hashes'][BUDGET]==profile['formal_budget_sha256']
    assert all(digest(p)==h for p,h in profile['source_hashes'].items())
    assert Path(profile['out']).resolve().is_relative_to(Path('runs').resolve())
    return budget,entry
