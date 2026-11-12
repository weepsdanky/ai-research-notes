"""Pinned author RULER configuration and local fast-tokenizer adapter."""
import hashlib,json,runpy
from pathlib import Path
from tokenizers import Tokenizer


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


class LocalTokenizer:
    def __init__(self,path):self.codec=Tokenizer.from_file(str(path))
    def text_to_tokens(self,text):return self.codec.encode(text,add_special_tokens=False).tokens
    def tokens_to_text(self,tokens):return self.codec.decoder.decode(tokens)


def configurations():
    import yaml
    for item in json.loads(Path('ruler_sources.lock.json').read_text()):assert digest(item['file'])==item['sha256']
    custom=yaml.safe_load(Path('.cache/long-context-sources/ruler-tasks.yaml').read_text())
    base=runpy.run_path('.cache/long-context-sources/ruler-scripts-data-synthetic-constants.py')['TASKS']
    templates=runpy.run_path('.cache/long-context-sources/ruler-scripts-data-template.py')['Templates']
    result={}
    for name,item in custom.items():
        config=dict(item,**base[item['task']]);config['template']=templates['base'].format(task_template=config['template'])+config.get('answer_prefix','');result[name]=config
    assert len(result)==13
    return result


def metrics():return runpy.run_path('.cache/long-context-sources/ruler-constants.py')['TASKS']
