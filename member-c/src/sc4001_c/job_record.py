"""Record live Slurm evidence when historical accounting is unavailable.

An EXIT event is a shell-observed exit, not a fabricated Slurm terminal record.
Keep failed/aged-out scontrol responses so the evidence boundary is explicit.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time


def snapshot(job,event,exit_code=None):
    result=subprocess.run(['scontrol','show','job',str(job),'-o'],capture_output=True,text=True)
    return {'job_id':str(job),'event':event,'unix_time':time.time(),
        'observed_utc':datetime.now(timezone.utc).isoformat(),'shell_exit_code':exit_code,
        'scontrol_returncode':result.returncode,'scontrol_stdout':result.stdout,'scontrol_stderr':result.stderr,
        'caveat':'Live scheduler snapshot; shell exit observation is not historical accounting.'}


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--job',default=os.getenv('SLURM_JOB_ID'))
    p.add_argument('--event',choices=('start','exit','observed'),required=True)
    p.add_argument('--exit-code',type=int);p.add_argument('--out',type=Path,default=Path('member-c/logs/job-evidence'))
    a=p.parse_args(argv)
    if not a.job or not a.job.isdigit():raise ValueError('A numeric GP job ID is required')
    if (a.event=='exit')!=(a.exit_code is not None):raise ValueError('Only exit events require an exit code')
    record=snapshot(a.job,a.event,a.exit_code)
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/f'{a.job}.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    print('JOB_EVIDENCE',a.job,a.event,record['scontrol_returncode'],flush=True)


if __name__=='__main__':main()
