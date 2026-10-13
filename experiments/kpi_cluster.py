#!/usr/bin/env python3
"""Plan, execute, package and merge KPI experiments across single-GPU nodes."""
import argparse
from contextlib import contextmanager
import fcntl
import socket
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE_FILES = ["SemSphere_method.py", "models/SemSphere/network/model.py",
                "models/SemSphere/trainer/trainer.py", "dataloader/dataset.py",
                "dataloader/data_preprocessing.py", "models/anomaly_predict.py",
                "experiments/datasets.py", "experiments/kpi_data.py", "experiments/kpi_runtime.py", "experiments/cluster_results.py", "experiments/kpi_cluster.py",
                "conf/semsphere/IOpsCompetition_Configs.py", "models/SemSphere/trainer/early_stopping.py"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def assign_jobs(costs, seeds, nodes):
    if nodes < 1 or not costs or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Require positive nodes, nonempty costs and distinct seeds')
    jobs = [{'seed': seed, 'entity_index': i, 'cost': int(cost)}
            for seed in seeds for i,cost in enumerate(costs)]
    loads = [0]*nodes
    assigned = []
    # Largest entities first; assign to the currently lightest node.
    for job in sorted(jobs, key=lambda item: (-item['cost'], item['seed'], item['entity_index'])):
        node = min(range(nodes), key=lambda index: (loads[index],index))
        assigned.append(dict(job, node_id=node+1))
        loads[node] += job['cost']
    return sorted(assigned,key=lambda item: (item['node_id'],item['seed'],item['entity_index'])), loads


def create_plan(args):
    from experiments.kpi_data import KPISeries
    dataset = KPISeries(args.data_root)
    overrides = json.loads(Path(args.config).read_text())
    if getattr(args, 'use_infonce', None) is not None:
        overrides['use_infonce'] = args.use_infonce
    from SemSphere_method import load_configs
    unknown = set(overrides) - set(vars(load_configs('KPI')))
    if unknown:
        raise ValueError(f'Unknown config fields: {sorted(unknown)}')
    seeds = [int(value) for value in args.seeds.split(',')]
    jobs, loads = assign_jobs(dataset.costs(), seeds, args.nodes)
    scheduler = getattr(args, 'scheduler', 'dynamic')
    if scheduler == 'dynamic':
        jobs = sorted(({key:value for key,value in job.items() if key!='node_id'} for job in jobs),
                      key=lambda job: (-job['cost'], job['seed'], job['entity_index']))
    plan = {
        'schema_version':2, 'scheduler':scheduler, 'dataset':'IOpsCompetition', 'study_name':args.study,
        'nodes':args.nodes, 'seeds':seeds, 'entity_ids':list(map(str,dataset.kpi_ids)),
        'config':overrides, 'num_workers':args.num_workers,
        'data_sha256':{name:sha256(dataset.rootdir/name) for name in ('phase2_train.csv','phase2_test.csv')},
        'source_sha256':{name:sha256(ROOT/name) for name in SOURCE_FILES},
        'jobs':jobs, 'estimated_node_loads':loads if scheduler=='fixed' else None,
    }
    plan['plan_id'] = hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    output = Path(args.output)
    if output.exists():
        old=json.loads(output.read_text())
        if old != plan:
            raise ValueError(f'{output} already contains a different plan; use a new study/output name')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(plan,indent=2,ensure_ascii=False))
    print(f'Plan: {output}; entities={len(dataset)}, seeds={seeds}, jobs={len(jobs)}, nodes={args.nodes}')
    if scheduler=='dynamic':
        print('Shared queue: workers claim longest remaining entities first; no fixed node assignments')
    else:
        for node in range(1,args.nodes+1):
            count=sum(job['node_id']==node for job in jobs)
            print(f'node {node:02d}: {count} jobs; load proxy={loads[node-1]}')


def read_plan(path):
    plan=json.loads(Path(path).read_text())
    payload={key:value for key,value in plan.items() if key!='plan_id'}
    if hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()!=plan['plan_id']:
        raise ValueError('Plan content was modified; generate a new plan')
    return plan


def node_jobs(plan,node_id):
    if not 1<=node_id<=plan['nodes']:
        raise ValueError(f"node-id must be between 1 and {plan['nodes']}")
    if plan.get('scheduler','fixed')=='dynamic':
        return plan['jobs']
    return [job for job in plan['jobs'] if job['node_id']==node_id]


def record_directory(plan,node_id):
    return ROOT/'results/kpi_cluster'/plan['study_name']/f'node_{node_id:02d}'/'entities'


def score_path(plan,job):
    return ROOT/'scores/SemSphere/IOpsCompetition'/plan['study_name']/'trial_manual'/f"seed_{job['seed']}"/f"diagnostics_{job['entity_index']}_{job['seed']}.npz"


def _run_node_impl(args):
    plan=read_plan(args.plan)
    jobs=node_jobs(plan,args.node_id)
    dynamic=plan.get('scheduler','fixed')=='dynamic'
    label = 'shared queue jobs' if dynamic else 'assigned jobs'
    print(f"Worker {args.node_id:02d}: {len(jobs)} {label}; plan={plan['plan_id'][:12]}",flush=True)
    if args.dry_run:
        for job in jobs:
            print(f"seed={job['seed']} entity={job['entity_index']} KPI={plan['entity_ids'][job['entity_index']]}")
        return
    if not jobs:
        return
    if dynamic and all(completed_record(plan,job) is not None for job in jobs):
        print('Shared queue is already complete; no data/model initialization needed',flush=True)
        return
    for name,expected in plan['source_sha256'].items():
        if sha256(ROOT/name)!=expected:
            raise ValueError(f'Source differs from plan: {name}. Deploy identical code to every node.')
    from experiments.kpi_data import KPISeries
    from experiments.cluster_results import read_entity_record
    dataset=KPISeries(args.data_root)
    if list(map(str,dataset.kpi_ids))!=plan['entity_ids']:
        raise ValueError('Entity order differs from plan')
    for name,expected in plan['data_sha256'].items():
        if sha256(dataset.rootdir/name)!=expected:
            raise ValueError(f'Data differs from plan: {name}')
    import torch
    from models.SemSphere.network import model as network
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; first-run worker requires one GPU')
    if network.SentenceTransformer is None:
        raise RuntimeError(f'sentence-transformers cannot load: {network.SENTENCE_TRANSFORMERS_IMPORT_ERROR}')
    print(f'GPU: {torch.cuda.get_device_name(0)}; CPU evaluation cap={os.environ.get("LOKY_MAX_CPU_COUNT","unset")}',flush=True)
    import SemSphere_method as method
    # Reuse parsed CSVs and the text cache across all jobs on this GPU.
    method.load_dataset=lambda name:dataset
    directory=record_directory(plan,args.node_id)
    directory.mkdir(parents=True,exist_ok=True)
    profile=directory.parent/'resolved_profile.json'
    profile.write_text(json.dumps(plan['config'],indent=2))
    if dynamic:
        return run_dynamic_queue(args, plan, method, profile)
    pending=[]
    for job in jobs:
        record=directory/f"seed_{job['seed']}_entity_{job['entity_index']:03d}.json"
        if record.exists():
            saved=read_entity_record(record)
            if saved['plan_id']!=plan['plan_id']:
                raise ValueError(f'Record belongs to another plan: {record}')
            if (saved['seed'],saved['entity_index'],saved['entity_id']) != (job['seed'],job['entity_index'],plan['entity_ids'][job['entity_index']]):
                raise ValueError(f'Record identity mismatch: {record}')
            if score_path(plan,job).is_file():
                print(f"Resume: skip seed={job['seed']} entity={job['entity_index']} (record and score exist)",flush=True)
                continue
        pending.append(job)
    started=time.time()
    for seed in plan['seeds']:
        indices=[job['entity_index'] for job in pending if job['seed']==seed]
        if not indices:
            continue
        argv=['SemSphere_method.py','--selected_dataset','KPI','--device','cuda',
              '--seed',str(seed),'--study_name',plan['study_name'],
              '--experiment_description',plan['study_name'],'--num_workers',str(plan['num_workers']),
              '--config_json',str(profile),'--entity_indices',','.join(map(str,indices)),
              '--entity_records_dir',str(directory),'--summary_suffix',f'_node_{args.node_id:02d}',
              '--cluster_plan_id',plan['plan_id'],'--cluster_worker_id',str(args.node_id),'--disable_timing']
        sys.argv=argv
        method.run_semsphere(method.parse_args())
    (directory.parent/'node_status.json').write_text(json.dumps({
        'plan_id':plan['plan_id'],'node_id':args.node_id,'completed':True,
        'wall_seconds_this_invocation':time.time()-started,'jobs_total':len(jobs)},indent=2))
    print(f'Node {args.node_id:02d} complete; score files remain in scores/SemSphere/...',flush=True)


def queue_record_directory(plan):
    return ROOT/'results/kpi_cluster'/plan['study_name']/'entities'


def queue_record_path(plan, job):
    return queue_record_directory(plan)/f"seed_{job['seed']}_entity_{job['entity_index']:03d}.json"


def job_lock_path(plan, job):
    return ROOT/'results/kpi_cluster'/plan['study_name']/'queue'/f"seed_{job['seed']}_entity_{job['entity_index']:03d}.lock"


def completed_record(plan, job):
    path=queue_record_path(plan,job)
    if not path.is_file():
        return None
    record=json.loads(path.read_text())
    identity=(record.get('plan_id'),record.get('seed'),record.get('entity_index'),record.get('entity_id'))
    expected=(plan['plan_id'],job['seed'],job['entity_index'],plan['entity_ids'][job['entity_index']])
    if identity!=expected:
        raise ValueError(f'Record identity differs from current plan: {path}')
    return record if score_path(plan,job).is_file() else None


@contextmanager
def claim_job(plan, node_id):
    """Hold an entity lock throughout training; recheck completion after locking."""
    stream=None
    claimed=None
    for job in plan['jobs']:
        if completed_record(plan,job) is not None:
            continue
        path=job_lock_path(plan,job)
        path.parent.mkdir(parents=True,exist_ok=True)
        candidate=path.open('a+')
        try:
            fcntl.flock(candidate.fileno(),fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            candidate.close()
            continue
        try:
            if completed_record(plan,job) is not None:
                fcntl.flock(candidate.fileno(),fcntl.LOCK_UN)
                candidate.close()
                continue
            candidate.seek(0);candidate.truncate()
            json.dump({'hostname':socket.gethostname(),'pid':os.getpid(),'worker_id':node_id,
                       'seed':job['seed'],'entity_index':job['entity_index'],'plan_id':plan['plan_id']},candidate)
            candidate.flush()
        except BaseException:
            candidate.close()
            raise
        stream=candidate
        claimed=job
        break
    try:
        yield claimed
    finally:
        if stream is not None:
            fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
            stream.close()


def job_is_running(plan,job):
    path=job_lock_path(plan,job)
    if not path.is_file():
        return False
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
    return False


def show_queue_status(plan):
    done=0;running=0
    counts={node:0 for node in range(1,plan['nodes']+1)}
    for job in plan['jobs']:
        record=completed_record(plan,job)
        if record is not None:
            done+=1
            worker=record.get('worker_id')
            if worker in counts:
                counts[worker]+=1
        elif job_is_running(plan,job):
            running+=1
    for worker,count in counts.items():
        print(f'worker {worker:02d}: {count} jobs complete')
    print(f"Total: {done}/{len(plan['jobs'])} jobs complete; running={running}; pending={len(plan['jobs'])-done-running}")
    if done==len(plan['jobs']):
        print('All entity records and score files exist; run scripts/merge_kpi.sh')


def run_dynamic_queue(args,plan,method,profile):
    if getattr(args,'poll_seconds',5.)<=0:
        raise ValueError('poll-seconds must be positive')
    directory=queue_record_directory(plan)
    directory.mkdir(parents=True,exist_ok=True)
    completed=0
    started=time.time()
    waiting=False
    while True:
        with claim_job(plan,args.node_id) as job:
            if job is None:
                if all(completed_record(plan,item) is not None for item in plan['jobs']):
                    break
                if not waiting:
                    print('Remaining jobs are held by other workers; waiting for completion or released tasks',flush=True)
                waiting=True
            else:
                waiting=False
                print(f"Worker {args.node_id:02d} claimed seed={job['seed']} entity={job['entity_index']} KPI={plan['entity_ids'][job['entity_index']]}",flush=True)
                sys.argv=['SemSphere_method.py','--selected_dataset','KPI','--device','cuda',
                    '--seed',str(job['seed']),'--study_name',plan['study_name'],
                    '--experiment_description',plan['study_name'],'--num_workers',str(plan['num_workers']),
                    '--config_json',str(profile),'--entity_indices',str(job['entity_index']),
                    '--entity_records_dir',str(directory),'--summary_suffix',f'_node_{args.node_id:02d}',
                    '--cluster_plan_id',plan['plan_id'],'--cluster_worker_id',str(args.node_id),'--disable_timing']
                method.run_semsphere(method.parse_args())
                if completed_record(plan,job) is None:
                    raise RuntimeError('Training returned without a complete entity record and score file')
                completed+=1
        if waiting:
            time.sleep(getattr(args,'poll_seconds',5.))
    status=record_directory(plan,args.node_id).parent/'node_status.json'
    temporary=status.with_suffix('.json.tmp')
    temporary.write_text(json.dumps({'plan_id':plan['plan_id'],'worker_id':args.node_id,
        'hostname':socket.gethostname(),'completed':True,'jobs_completed_this_invocation':completed,
        'wall_seconds_this_invocation':time.time()-started},indent=2))
    temporary.replace(status)
    print(f'Worker {args.node_id:02d} finished; all queue jobs are complete',flush=True)


@contextmanager
def node_lock(plan, node_id):
    """Shared-filesystem lock released by the OS even after a process crash."""
    directory = record_directory(plan, node_id).parent
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'.run.lock').open('a+') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Node {node_id} already runs in this shared study; choose another node ID") from error
        try:
            stream.seek(0)
            stream.truncate()
            json.dump({'hostname': socket.gethostname(), 'pid': os.getpid(),
                       'node_id': node_id, 'plan_id': plan['plan_id']}, stream)
            stream.flush()
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run_node(args):
    plan = read_plan(args.plan)
    node_jobs(plan, args.node_id)
    if args.dry_run:
        return _run_node_impl(args)
    with node_lock(plan, args.node_id):
        return _run_node_impl(args)


def show_status(args):
    plan = read_plan(args.plan)
    if plan.get('scheduler','fixed')=='dynamic':
        return show_queue_status(plan)
    done_total = 0
    for node in range(1, plan['nodes']+1):
        jobs = node_jobs(plan, node)
        done = 0
        for job in jobs:
            path = record_directory(plan, node)/f"seed_{job['seed']}_entity_{job['entity_index']:03d}.json"
            if not path.is_file() or not score_path(plan, job).is_file():
                continue
            record = json.loads(path.read_text())
            if (record.get('plan_id'), record.get('seed'), record.get('entity_index')) == (plan['plan_id'], job['seed'], job['entity_index']):
                done += 1
        done_total += done
        print(f'node {node:02d}: {done}/{len(jobs)} jobs complete')
    print(f"Total: {done_total}/{len(plan['jobs'])} jobs complete")
    if done_total == len(plan['jobs']):
        print('All entity records and score files exist; run scripts/merge_kpi.sh')


def pack_node(args):
    plan=read_plan(args.plan)
    jobs=node_jobs(plan,args.node_id)
    if plan.get('scheduler','fixed')=='dynamic':
        raise ValueError('Dynamic workers share all records and scores; merge directly instead of packaging per node')
    directory=record_directory(plan,args.node_id)
    paths=set(directory.parent.rglob('*')) if directory.parent.exists() else set()
    for job in jobs:
        paths.add(score_path(plan,job))
        checkpoint=ROOT/'best_network/SemSphere/IOpsCompetition'/plan['study_name']/'trial_manual'/f"seed_{job['seed']}"/f"{job['entity_index']:02d}_best_network.pth"
        if checkpoint.is_file():
            paths.add(checkpoint)
    paths.add(ROOT/'logs'/plan['study_name']/f'node_{args.node_id:02d}.log')
    paths.add(ROOT/'results'/f"SemSphere_IOpsCompetition_summary_{plan['study_name']}_node_{args.node_id:02d}.csv")
    paths.add(Path(args.plan).resolve())
    for job in jobs:
        if not score_path(plan,job).is_file() or not (directory/f"seed_{job['seed']}_entity_{job['entity_index']:03d}.json").is_file():
            raise ValueError(f'Incomplete job: {job}; rerun the node before packaging')
    destination=ROOT/'exports'/plan['study_name']/f'node_{args.node_id:02d}.tar'
    destination.parent.mkdir(parents=True,exist_ok=True)
    # Uncompressed tar: avoid spending CPU on recompressing existing NPZ data.
    with tarfile.open(destination,'w') as archive:
        for path in sorted(paths):
            if path.is_file():
                archive.add(path,arcname=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else 'configs/'+path.name,recursive=False)
    print(destination)


def merge_results(args):
    plan=read_plan(args.plan)
    from experiments.cluster_results import read_entity_record,aggregate_records,write_summary
    records={}
    records_root = Path(args.records_root) if args.records_root else ROOT/'results/kpi_cluster'/plan['study_name']
    for path in records_root.rglob('seed_*_entity_*.json'):
        record=read_entity_record(path)
        if record.get('plan_id')!=plan['plan_id']:
            raise ValueError(f'Record from another plan: {path}')
        if record['dataset']!=plan['dataset'] or record['entity_count']!=len(plan['entity_ids']):
            raise ValueError(f'Dataset identity mismatch: {path}')
        key=(record['seed'],record['entity_index'])
        if key in records:
            raise ValueError(f'Duplicate record: {key}')
        if not 0<=record['entity_index']<len(plan['entity_ids']) or record['entity_id']!=plan['entity_ids'][record['entity_index']]:
            raise ValueError(f'KPI ID mismatch: {path}')
        records[key]=record
    expected={(job['seed'],job['entity_index']) for job in plan['jobs']}
    if set(records)!=expected:
        missing=sorted(expected-set(records));extra=sorted(set(records)-expected)
        raise ValueError(f'Incomplete results; missing={missing}; extra={extra}. No full-dataset summary written.')
    results=[aggregate_records([records[(seed,i)] for i in range(len(plan['entity_ids']))]) for seed in plan['seeds']]
    output=Path(args.output) if args.output else ROOT/'results'/f"SemSphere_IOpsCompetition_summary_{plan['study_name']}.csv"
    write_summary(output,results)
    print(f'Full original-format CSV: {output}')
    for result in results:
        print(f"seed={result['seed']} | VUS-ROC={result['Test VUS-ROC']:.5f} | VUS-PR={result['Test VUS-PR']:.5f} | RPA F1={result['RPA F1']:.5f}")
    if len(results)>1:
        import numpy as np
        for label,key in [('VUS-ROC','Test VUS-ROC'),('VUS-PR','Test VUS-PR'),('RPA F1','RPA F1')]:
            values=[result[key] for result in results]
            print(f'{label}: {np.nanmean(values):.5f} ± {np.nanstd(values):.5f}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    plan=sub.add_parser('plan',help='Generate identical entity/seed assignments for all nodes')
    plan.add_argument('--data-root',default=None)
    plan.add_argument('--config',default='configs/kpi_original_v1.json')
    plan.add_argument('--nodes',type=int,default=20)
    plan.add_argument('--scheduler',choices=['dynamic','fixed'],default='dynamic')
    plan.add_argument('--seeds',default='1')
    alignment = plan.add_mutually_exclusive_group()
    alignment.add_argument('--enable_infonce', dest='use_infonce', action='store_true')
    alignment.add_argument('--disable_infonce', dest='use_infonce', action='store_false')
    plan.set_defaults(use_infonce=None)
    plan.add_argument('--num-workers',type=int,default=2)
    plan.add_argument('--study',default='kpi_original_v1')
    plan.add_argument('--output',default='configs/kpi_original_plan.json')
    for command in ('run','pack','merge','status'):
        item=sub.add_parser(command)
        item.add_argument('--plan',default='configs/kpi_original_plan.json')
        if command in ('run','pack'):
            item.add_argument('--node-id',type=int,required=True)
        if command=='run':
            item.add_argument('--data-root',default=None)
            item.add_argument('--dry-run',action='store_true')
            item.add_argument('--poll-seconds',type=float,default=5.0)
        if command=='merge':
            item.add_argument('--records-root',default=None)
            item.add_argument('--output',default=None)
    args=parser.parse_args()
    os.chdir(ROOT)
    {'plan':create_plan,'run':run_node,'pack':pack_node,'merge':merge_results,'status':show_status}[args.command](args)


if __name__=='__main__':
    main()
