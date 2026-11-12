"""Record device and source provenance for the complete local numerical test suite."""
import argparse
from datetime import datetime
import hashlib,json,os,platform,resource,sys,time,unittest
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','mps'],required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args();out=Path(args.out)
    if out.exists():raise FileExistsError('Preserve existing device evidence; choose a new output')
    os.environ['LAB_TEST_DEVICE']=args.device
    import torch
    torch.set_num_threads(1)
    local_before={str(path.resolve()):digest(path) for path in Path('.').glob('*.py')}
    protocol=json.loads(Path('runs/final-device-test-protocol.json').read_text())
    hashes={name:digest(name) for name in protocol['source_hashes']}
    assert hashes==protocol['source_hashes'],'Test/core sources changed since protocol; review before accepting'
    if args.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('Actual MPS required; no CPU substitution')
    probe=torch.ones(1,device=args.device)
    assert probe.device.type==args.device
    if args.device=='mps':torch.mps.synchronize()
    record=dict(started=datetime.now().astimezone().isoformat(),device=args.device,selected_LAB_TEST_DEVICE=os.environ['LAB_TEST_DEVICE'],
                actual_probe_device=str(probe.device),mps_built=torch.backends.mps.is_built(),mps_available=torch.backends.mps.is_available(),
                python=platform.python_version(),executable=sys.executable,platform=platform.platform(),torch=str(torch.__version__),
                threads=torch.get_num_threads(),source_hashes=hashes,runner_sha256=digest(__file__),sources_lock_sha256=digest('sources.lock.json'),
                uv_lock_sha256=digest('uv.lock'),pyproject_sha256=digest('pyproject.toml'),
                mps_high_watermark_ratio=os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO','default'),
                mps_fallback=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','default'),status='running',
                scope='Full discovery in selected mode; file/codec/GPQA and FP64 numerical oracle fixtures may deliberately use CPU; model math tests select LAB_TEST_DEVICE.')
    print(json.dumps(record,indent=2),flush=True)
    tick=time.monotonic()
    try:
        suite=unittest.defaultTestLoader.discover('.',pattern='test_*.py',top_level_dir='.')
        discovered=suite.countTestCases();assert discovered>=52,'Expected original47 plus GPQA5 controls'
        result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(suite)
        passed=result.wasSuccessful() and result.testsRun==discovered and not result.skipped
        loaded={str(Path(module.__file__).resolve()) for module in tuple(sys.modules.values()) if getattr(module,'__file__',None)}
        imported={str(Path(name).relative_to(Path.cwd())):sha for name,sha in local_before.items() if name in loaded}
        assert all(digest(name)==sha for name,sha in imported.items()),'Imported sources changed during tests'
        record['imported_local_source_hashes']=imported
        record.update(status='passed' if passed else 'failed_or_skipped',tests_discovered=discovered,tests_run=result.testsRun,
                      failures=[str(test) for test,_ in result.failures],errors=[str(test) for test,_ in result.errors],
                      skipped=[(str(test),reason) for test,reason in result.skipped])
    except BaseException as error:
        record.update(status='interrupted_or_error',error_type=type(error).__name__)
        raise
    finally:
        record.update(ended=datetime.now().astimezone().isoformat(),seconds=time.monotonic()-tick,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if args.device=='mps':record['mps_driver_bytes_at_end']=torch.mps.driver_allocated_memory()
        out.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({k:v for k,v in record.items() if k not in ('source_hashes',)},indent=2),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
