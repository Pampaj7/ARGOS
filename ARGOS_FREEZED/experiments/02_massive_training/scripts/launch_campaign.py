#!/usr/bin/env python3
"""Fail-fast launcher for disjoint ARGOS v2 one-GPU campaign slots."""
from __future__ import annotations
import argparse, fcntl, json, os, queue, socket, subprocess, sys, threading, time
from datetime import datetime, timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from campaign_common import *

RUNS=[(1,20260722),(1,20260723),(1,20260724),(3,20260722),(3,20260723),(3,20260724),(6,20260722),(6,20260723),(6,20260724)]

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--slot",type=int,choices=(0,1)); args=parser.parse_args()
    verify_frozen_core(); verify_training_dependencies()
    slot_lock=None
    if args.slot is not None:
        slot_lock=(CAMPAIGN/f"logs/slot_{args.slot}.lock").open("a+"); fcntl.flock(slot_lock,fcntl.LOCK_EX)
    for review in (CAMPAIGN/"protocol_audit/review_scientific.md",CAMPAIGN/"protocol_audit/review_engineering.md"):
        if not review.is_file() or "CRITICAL ISSUES: 0" not in review.read_text(): raise SystemExit(f"prelaunch review not cleared: {review}")
    if not (CAMPAIGN/"protocol_audit/tests_passed.json").is_file(): raise SystemExit("targeted tests have not passed")
    runs=RUNS if args.slot is None else RUNS[args.slot::2]
    pending=queue.Queue(); [pending.put(run) for run in runs]; lock=threading.Lock(); failures=[]; jobs=[]
    manifest_path=CAMPAIGN/"launch_manifest.json" if args.slot is None else CAMPAIGN/f"logs/launch_slot_{args.slot}.json"
    environment_base=os.environ.copy(); environment_base["PYTHONPATH"]=f"{ROOT/'src'}:{HERE}"
    local_tmp=Path("/tmp")/f"argos_v2_{os.environ.get('LSB_JOBID',os.getpid())}"; local_tmp.mkdir(exist_ok=True); environment_base["TMPDIR"]=str(local_tmp)
    def worker(gpu):
        while True:
            try: budget,seed=pending.get_nowait()
            except queue.Empty: return
            output=run_directory(budget,seed); state=output/"state.json"
            if state.is_file() and json.loads(state.read_text()).get("status")=="completed": pending.task_done(); continue
            log=output/"logs/train.log"; log.parent.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,str(HERE/"train_one_run.py"),"--budget",str(budget),"--seed",str(seed),"--device","cuda:0","--workers","48","--flow-batch-size","32","--batch-size","12"]
            environment=environment_base.copy()
            if args.slot is None: environment["CUDA_VISIBLE_DEVICES"]=str(gpu)
            record={"budget":f"{budget}x","seed":seed,"gpu_slot":gpu,"host":socket.gethostname(),"pid":None,"started_at":datetime.now(timezone.utc).isoformat(),"command":command,"log":str(log)}
            with log.open("a") as handle:
                process=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,env=environment,start_new_session=False); record["pid"]=process.pid
                with lock: jobs.append(record); atomic_json(manifest_path,{"project":"ARGOS v2","status":"RUNNING","slot":args.slot,"lsf_job_id":os.environ.get("LSB_JOBID"),"host":socket.gethostname(),"jobs":jobs,"failures":failures,"dataset_7_locked":True})
                code=process.wait()
            record["ended_at"]=datetime.now(timezone.utc).isoformat(); record["exit_code"]=code
            if code:
                with lock: failures.append(record); atomic_json(manifest_path,{"project":"ARGOS v2","status":"FAILED","slot":args.slot,"lsf_job_id":os.environ.get("LSB_JOBID"),"jobs":jobs,"failures":failures,"dataset_7_locked":True})
                return
            pending.task_done()
    threads=[threading.Thread(target=worker,args=(gpu,),daemon=False) for gpu in ((0,1) if args.slot is None else (0,))]
    [thread.start() for thread in threads]; [thread.join() for thread in threads]
    status="FAILED" if failures or not pending.empty() else "TRAINING_COMPLETE"
    atomic_json(manifest_path,{"project":"ARGOS v2","status":status,"slot":args.slot,"lsf_job_id":os.environ.get("LSB_JOBID"),"host":socket.gethostname(),"jobs":jobs,"failures":failures,"dataset_7_locked":True})
    if status!="TRAINING_COMPLETE": raise SystemExit(1)
if __name__=="__main__": main()
