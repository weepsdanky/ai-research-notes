"""Measure historical JSON record retention without running model inference."""
import sys,json,resource,time,hashlib
from pathlib import Path
path,mode,out=sys.argv[1:]
if mode not in ('full','minimal'):raise ValueError('Choose full or minimal')
started=time.perf_counter();rows=[];h=hashlib.sha256()
with Path(path).open() as stream:
 for line in stream:
  r=json.loads(line)
  h.update(json.dumps((r['output_ids'],r['prediction']),ensure_ascii=False).encode())
  rows.append(r if mode=='full' else {'output_ids':r['output_ids'],'prediction':r['prediction']})
r={'mode':mode,'rows':len(rows),'replay_fields_sha256':h.hexdigest(),'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),'seconds':time.perf_counter()-started,'scope':'JSON record retention only, no model/inference; source engine unchanged'}
Path(out).write_text(json.dumps(r,indent=2)+'\n');print(r)
