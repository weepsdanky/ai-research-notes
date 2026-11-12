"""Positive/negative wiring controls, not model benchmark scores."""
import json
from pathlib import Path
from eval_multipl_tasks import datasets
from multipl_judge import execute
from sandbox import Sandbox


SOLUTIONS = {
 'py': '    for i in range(len(numbers)):\n        for j in range(i+1,len(numbers)):\n            if abs(numbers[i]-numbers[j]) < threshold: return True\n    return False\n',
 'cpp':'    for (size_t i=0;i<numbers.size();i++) for(size_t j=i+1;j<numbers.size();j++) if(std::abs(numbers[i]-numbers[j])<threshold) return true;\n    return false;\n',
 'java':'        for(int i=0;i<numbers.size();i++) for(int j=i+1;j<numbers.size();j++) if(Math.abs(numbers.get(i)-numbers.get(j))<threshold) return true;\n        return false;\n',
 'cs':'        for(int i=0;i<numbers.Count;i++) for(int j=i+1;j<numbers.Count;j++) if(Math.Abs(numbers[i]-numbers[j])<threshold) return true;\n        return false;\n',
 'php':'    for($i=0;$i<count($numbers);$i++) for($j=$i+1;$j<count($numbers);$j++) if(abs($numbers[$i]-$numbers[$j])<$threshold) return true;\n    return false;\n}\n',
 'ts':'    for(let i=0;i<numbers.length;i++) for(let j=i+1;j<numbers.length;j++) if(Math.abs(numbers[i]-numbers[j])<threshold) return true;\n    return false;\n}\n',
 'js':'    for(let i=0;i<numbers.length;i++) for(let j=i+1;j<numbers.length;j++) if(Math.abs(numbers[i]-numbers[j])<threshold) return true;\n    return false;\n}\n',
 'sh': '''    awk -v xs="$1" -v threshold="$2" 'BEGIN {n=split(xs,a," "); for(i=1;i<=n;i++) for(j=i+1;j<=n;j++) {d=a[i]-a[j]; if(d<0)d=-d; if(d<threshold){print "true";exit}} print "false"}'
'''
}
WRONG = dict(py='    return False\n',cpp='return false;',java='return false;',cs='return false;',php='return false;\n}',ts='return false;\n}',js='return false;\n}',sh='echo false')
EARLY = dict(py='    raise SystemExit(0)\n',cpp='std::exit(0); return false;',java='System.exit(0); return false;',cs='System.Environment.Exit(0); return false;',php='exit(0);\n}',ts="require('node:process').exit(0); return false;\n}",js='process.exit(0);\n}',sh='exit 0')


def main():
    sandbox = Sandbox('multipl');results = []
    for language,rows in datasets().items():
        row = next(r for r in rows if r['name']=='HumanEval_0_has_close_elements')
        for label,value in [('canonical',SOLUTIONS[language]),('wrong',WRONG[language]),('early_exit',EARLY[language]),('syntax_error','    @@@\n')]:
            result = execute(sandbox,language,row,value);result.update(language=language,control=label)
            results.append(result)
            Path('runs/multipl-probe.json').write_text(json.dumps(results,indent=2)+'\n')
            print(language,label,result['status'],result['passed'],flush=True)
            if result['passed'] != (label=='canonical'):raise AssertionError('Control failed: '+language+' '+label)
    print('All 32 isolated language controls passed')


if __name__=='__main__':main()
