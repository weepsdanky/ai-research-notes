"""Explicit Mac-only repairs; original publisher data/predictions stay intact."""
import copy
import hashlib
import json
import re
from pathlib import Path
from eval_multipl_tasks import datasets


def sha(text):return hashlib.sha256(text.encode()).hexdigest()


def main():
    targets={'cpp':'HumanEval_137_compare_one','js':'HumanEval_51_remove_vowels','php':'HumanEval_23_strlen'}
    rows=[];changes=[]
    for language,name in targets.items():
        original=next(r for r in datasets()[language] if r['name']==name);row=copy.deepcopy(original)
        if language=='cpp':
            prompt=row['prompt'];doc=prompt[prompt.index('// Create a function'):prompt.index('Union_long_float_std_string_std_nullopt compare_one')]
            row['prompt']='#include<assert.h>\n#include<bits/stdc++.h>\nusing Input = std::variant<long,float,std::string>;\nusing Output = std::optional<Input>;\n'+doc+'Output compare_one(Input a, Input b) {\n'
            def literal(value):
                if isinstance(value,str):return 'std::string('+json.dumps(value)+')'
                return repr(value)+'f' if isinstance(value,float) else str(value)+'L'
            cases=[(1,2,2),(1,2.5,2.5),(2,3,3),(5,6,6),(1,'2,3','2,3'),('5,1','6','6'),('1','2','2'),('1',1,None)]
            assert original['tests'].count('assert(candidate(')==len(cases)
            assertions=[]
            for a,b,value in cases:
                expected='std::nullopt' if value is None else 'Output(Input('+literal(value)+'))'
                assertions.append('assert(candidate(Input('+literal(a)+'),Input('+literal(b)+')) == '+expected+');')
            row['tests']='}\nint main() {\nauto candidate=compare_one;\n'+'\n'.join(assertions)+'\n}\n'
            row['canonical_control']='''auto number=[](const Input &x)->long double {
return std::visit([](const auto &v)->long double {
using T=std::decay_t<decltype(v)>;
if constexpr(std::is_same_v<T,std::string>) { auto s=v; std::replace(s.begin(),s.end(),',','.'); return std::stold(s); }
else return static_cast<long double>(v);
},x); };
auto a_value=number(a),b_value=number(b);
if(a_value==b_value)return std::nullopt;
return a_value>b_value ? Output(a) : Output(b);
'''
            reason='Replace malformed untagged unions/nullopt type with tagged variant/optional, explicit typed literals; preserve all eight published input/expected-value pairs and return type distinctions. Not the official C++ prompt.'
        elif language=='js':
            for old,new in [('"abcdef\nghijklm"','"abcdef\\nghijklm"'),('"bcdf\nghjklm"','"bcdf\\nghjklm"')]:
                assert old in row['tests'];row['tests']=row['tests'].replace(old,new)
            row['canonical_control']="return text.replace(/[aeiou]/gi,'');\n}\n"
            reason='Escape literal newlines inside JS string literals, preserving the actual newline-valued input/expected strings; prompt unchanged.'
        else:
            assert 'function strlen(' in row['prompt']
            row['prompt']=re.sub(r'\bstrlen\b','multipl_strlen',row['prompt']);row['tests']=re.sub(r'\bstrlen\b','multipl_strlen',row['tests'])
            row['canonical_control']='return strlen($string);\n}\n'
            reason='Rename the candidate to avoid redeclaring PHP builtin strlen; preserve all three inputs/expected lengths. Not the official PHP prompt.'
        row['language']=language;rows.append(row)
        changes.append(dict(language=language,name=name,reason=reason,original_prompt_sha256=sha(original['prompt']),original_tests_sha256=sha(original['tests']),
                            repaired_prompt_sha256=sha(row['prompt']),repaired_tests_sha256=sha(row['tests'])))
    root=Path('data/multipl-repaired');root.mkdir(exist_ok=True)
    data=''.join(json.dumps(r)+'\n' for r in rows);(root/'repairs.jsonl').write_text(data)
    Path('multipl_repairs.lock.json').write_text(json.dumps(dict(version='mac-repair-v1',not_official_protocol=True,selection='Only independently identified invalid scaffolds; no answer-quality selection',original_source_lock_sha256=sha(Path('multipl_sources.lock.json').read_text()),data_sha256=sha(data),changes=changes),indent=2)+'\n')


if __name__=='__main__':main()
