"""Pinned MRCR author scoring and released Qwen chat rendering; no API calls."""
import ast
from contextlib import contextmanager
from datetime import datetime
from difflib import SequenceMatcher
from functools import lru_cache
import hashlib,json,re
from pathlib import Path
from typing import no_type_check


def locked_text(name):
    spec=next(item for item in json.loads(Path('long_context_sources.lock.json').read_text()) if Path(item['file']).name==name)
    data=Path(spec['file']).read_bytes()
    assert len(data)==spec['bytes'] and hashlib.sha256(data).hexdigest()==spec['sha256'],'Locked long-context source drift'
    return data.decode('utf-8')


@lru_cache
def chat_template():
    import jinja2, jinja2.ext
    from jinja2.ext import Extension
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from packaging import version
    source=locked_text('hf-chat-template-utils.py')
    names={'_compile_jinja_template','_cached_compile_jinja_template'}
    nodes=[node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes}==names
    namespace=dict(lru_cache=lru_cache,no_type_check=no_type_check,is_jinja_available=lambda:True,
                   jinja2=jinja2,Extension=Extension,ImmutableSandboxedEnvironment=ImmutableSandboxedEnvironment,
                   version=version,contextmanager=contextmanager,json=json,datetime=datetime)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'locked HF chat compiler','exec'),namespace)
    return namespace['_compile_jinja_template'](locked_text('qwen-chat-template.jinja'))


def render(messages):
    assert messages and all(type(m)==dict and m['role'] in ('user','assistant','system') and type(m['content'])==str for m in messages)
    # Plain JSON text only; no objects, tools, media, or private chat serialization.
    return chat_template().render(messages=messages,add_generation_prompt=True,enable_thinking=False,tools=None)


@lru_cache
def author_grade():
    block=re.search(r'```python\n(.*?)```',locked_text('mrcr-readme.md'),re.S).group(1)
    nodes=[node for node in ast.parse(block).body if isinstance(node,ast.FunctionDef) and node.name=='grade']
    assert len(nodes)==1
    namespace=dict(SequenceMatcher=SequenceMatcher)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'locked MRCR author grade','exec'),namespace)
    return namespace['grade']


def validate_source_row(row,messages):
    """Bind the requested answer to its actual assistant message and all8 asks."""
    assert row['n_needles']==8 and messages[-1]['role']=='user'
    index=row['desired_msg_index'];assert 0<=index<len(messages)-2
    assert messages[index]['role']=='user' and messages[index+1]['role']=='assistant'
    assert row['answer']==row['random_string_to_prepend']+messages[index+1]['content'],'Source answer/message mismatch'
    needle=messages[index]['content']
    assert sum(m['role']=='user' and m['content']==needle for m in messages[:-1])==8,'Original source has unexpected needle count'
