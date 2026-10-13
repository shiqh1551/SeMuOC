#!/usr/bin/env python3
"""Shared GPU task queue for multi-dataset, multi-seed SemSphere experiments."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.datasets import canonical_dataset,dataset_label,load_dataset,data_files,entity_sizes
from experiments.kpi_cluster import sha256,read_plan

SOURCE_FILES=['SemSphere_method.py','experiments/cluster.py','experiments/datasets.py',
 'experiments/cluster_results.py','experiments/kpi_data.py','experiments/kpi_runtime.py',
 'experiments/kpi_cluster.py','models/SemSphere/network/model.py',
 'models/SemSphere/trainer/trainer.py','models/SemSphere/trainer/early_stopping.py',
 'dataloader/dataset.py','dataloader/data_preprocessing.py','utils/ad_utils.py','models/anomaly_predict.py']

# These queue revisions have identical model/data/config/score semantics.
# Accept their existing plans so the CPU/idle update does not discard results.
COMPATIBLE_QUEUE_SHA256={'d001b314fe7371e26231bfbfa9c64022d73bc83b2fc4179e47a20fddeba58a08','3ea060a6b4425f9dee79e86be0eb73c1d4a38358f421544a7037bd83d8eb536a','85633fa71a36d30ee2d7576404c208acce89378d9408ea44c791ce5133833a9d'}


def validate_sources(plan):
    for name,expected in plan['source_sha256'].items():
        if sha256(ROOT/name)==expected:
            continue
        if name=='experiments/cluster.py' and expected in COMPATIBLE_QUEUE_SHA256:
            print('Compatible queue update: device, idle exit, and dataset selection; existing plan retained',flush=True)
            continue
        raise ValueError(f'Code differs from plan: {name}')


def select_device(requested):
    import torch
    if requested=='cpu':
        return 'cpu'
    available=torch.cuda.is_available()
    if requested=='cuda' and not available:
        raise RuntimeError('CUDA is unavailable; select --device cpu or --device auto')
    return 'cuda' if available else 'cpu'


def selected_datasets(plan,args):
    requested=getattr(args,'datasets',None)
    if requested is None:
        return list(plan['datasets'])
    names=[canonical_dataset(name) for name in requested]
    if not names or len(set(names))!=len(names):
        raise ValueError('Selected datasets must be nonempty and distinct')
    missing=set(names)-set(plan['datasets'])
    if missing:
        raise ValueError(f'Datasets absent from plan: {sorted(missing)}')
    return [name for name in plan['datasets'] if name in names]


def selected_jobs(plan,args):
    names=set(selected_datasets(plan,args))
    return [job for job in plan['jobs'] if job['dataset'] in names]


def write_json(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(payload,indent=2,ensure_ascii=False))
    temporary.replace(path)


def prepare(args):
    from SemSphere_method import load_configs
    names=[canonical_dataset(name) for name in args.datasets]
    if len(set(names))!=len(names):
        raise ValueError('Duplicate datasets')
    seeds=[int(value) for value in args.seeds.split(',')]
    if not seeds or len(set(seeds))!=len(seeds) or any(seed<0 or seed>=2**32 for seed in seeds):
        raise ValueError('Seeds must be distinct integers in [0, 2**32)')
    if args.nodes<1:
        raise ValueError('nodes must be positive')
    if args.data_root and len(names)!=1:
        raise ValueError('--data-root is for a single dataset; use --data-base for multiple datasets')
    custom=json.loads(Path(args.config).read_text()) if args.config else {}
    if custom and len(names)>1:
        custom={canonical_dataset(key):value for key,value in custom.items()}
        if set(custom)-set(names):
            raise ValueError('Configuration contains an unselected dataset')
    roots={}
    entries={}
    jobs=[]
    source_files=list(SOURCE_FILES)
    for name in names:
        config=vars(load_configs(name)).copy()
        overrides=custom if len(names)==1 else custom.get(name,{})
        if set(overrides)-set(config):
            raise ValueError(f'Unknown {name} configuration fields: {set(overrides)-set(config)}')
        config.update(overrides)
        # Completed queue tasks require persisted scores for review and resuming.
        config['save_diagnostics']=True
        if args.use_infonce is not None:
            config['use_infonce']=args.use_infonce
        if not isinstance(config['use_infonce'],bool):
            raise ValueError('use_infonce must be a boolean')
        if config['dataset']!=name:
            raise ValueError('Configuration cannot rename a dataset')
        root=args.data_root
        if args.data_base:
            base=Path(args.data_base)
            choices=[base/('iops_competition' if name=='IOpsCompetition' else name.lower()),base/name,base/dataset_label(name)]
            root=next((path for path in choices if path.is_dir()),choices[0])
        dataset=load_dataset(name,root)
        roots[name]=str(dataset.rootdir.resolve())
        fingerprints={str(path.relative_to(dataset.rootdir)):sha256(path) for path in data_files(dataset)}
        entries[name]={'config':config,'entity_ids':list(map(str,dataset.entity_ids)),
                       'data_sha256':fingerprints,'data_root':roots[name]}
        for index in range(len(dataset)):
            train,test,channels=entity_sizes(dataset,index)
            if channels!=config['input_channels']:
                raise ValueError(f'{name}/{dataset.entity_ids[index]} has {channels} channels, config specifies {config["input_channels"]}')
            window,step=config['window_size'],config['time_step']
            train_windows=max(0,(train-window)//step+1)
            test_windows=max(0,(test-window)//step+1)
            if train_windows<2 or test_windows<1:
                raise ValueError(f'{name}/{dataset.entity_ids[index]} too short for original window settings')
            cost=int((train_windows*config['num_epoch']+test_windows)*window*channels*config['d_model']**2*config['tcn_layers'])
            jobs.extend({'dataset':name,'seed':seed,'entity_index':index,'cost':cost} for seed in seeds)
        source_files.extend([f'conf/semsphere/{name}_Configs.py',
                             f'ts_datasets/ts_datasets/anomaly/{"iops_competition" if name=="IOpsCompetition" else name.lower()}.py'])
    study=args.study or '_'.join(dataset_label(name).lower() for name in names)+f'_{len(seeds)}seed'
    if Path(study).name!=study or study in ('.','..'):
        raise ValueError('study must be a single directory name')
    plan={'schema_version':1,'kind':'semsphere_multi_dataset','study_name':study,'nodes':args.nodes,
          'seeds':seeds,'datasets':entries,'num_workers':args.num_workers,
          'jobs':sorted(jobs,key=lambda job:(-job['cost'],job['dataset'],job['seed'],job['entity_index'])),
          'source_sha256':{name:sha256(ROOT/name) for name in source_files}}
    plan['plan_id']=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    output=Path(args.output) if args.output else ROOT/'configs'/f'{study}_plan.json'
    if output.exists() and json.loads(output.read_text())!=plan:
        raise ValueError(f'{output} contains a different plan; use another study/output')
    write_json(output,plan)
    print(f'Plan: {output}; workers≤{args.nodes}; seeds={seeds}; tasks={len(jobs)}')
    for name,entry in entries.items():
        print(f'{dataset_label(name)}: {len(entry["entity_ids"])} entities × {len(seeds)} seeds; original config unless explicitly overridden')
    print('Any available worker claims the longest remaining task; no fixed assignment')


def plan_root(plan):
    return ROOT/'results/cluster'/plan['study_name']


def record_dir(plan,name):
    return plan_root(plan)/'entities'/name


def record_path(plan,job):
    return record_dir(plan,job['dataset'])/f"seed_{job['seed']}_entity_{job['entity_index']:03d}.json"


def score_path(plan,job):
    return ROOT/'scores/SemSphere'/job['dataset']/plan['study_name']/'trial_manual'/f"seed_{job['seed']}"/f"diagnostics_{job['entity_index']}_{job['seed']}.npz"


def lock_path(plan,job):
    return plan_root(plan)/'queue'/job['dataset']/f"seed_{job['seed']}_entity_{job['entity_index']:03d}.lock"


def completed(plan,job):
    path=record_path(plan,job)
    if not path.is_file():
        return None
    record=json.loads(path.read_text())
    expected=(plan['plan_id'],job['dataset'],job['seed'],job['entity_index'],plan['datasets'][job['dataset']]['entity_ids'][job['entity_index']])
    actual=tuple(record.get(key) for key in ('plan_id','dataset','seed','entity_index','entity_id'))
    if actual!=expected:
        raise ValueError(f'Record identity mismatch: {path}')
    return record if score_path(plan,job).is_file() else None


@contextmanager
def worker_lock(plan,worker_id):
    if not 1<=worker_id<=plan['nodes']:
        raise ValueError(f'worker-id must be 1–{plan["nodes"]}')
    path=plan_root(plan)/'workers'/f'worker_{worker_id:02d}'/'.run.lock'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Worker {worker_id} already runs in this study') from error
        try:
            stream.seek(0);stream.truncate()
            json.dump({'plan_id':plan['plan_id'],'hostname':socket.gethostname(),'pid':os.getpid()},stream)
            stream.flush()
            yield
        finally:
            fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


@contextmanager
def claim(plan,worker_id,jobs=None):
    selected=None;stream=None
    for job in (plan['jobs'] if jobs is None else jobs):
        if completed(plan,job) is not None:
            continue
        path=lock_path(plan,job);path.parent.mkdir(parents=True,exist_ok=True)
        candidate=path.open('a+')
        try:
            fcntl.flock(candidate.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            candidate.close();continue
        try:
            if completed(plan,job) is not None:
                candidate.close();continue
            candidate.seek(0);candidate.truncate()
            json.dump(dict(job,worker_id=worker_id,hostname=socket.gethostname(),pid=os.getpid(),plan_id=plan['plan_id']),candidate)
            candidate.flush()
        except BaseException:
            candidate.close();raise
        stream=candidate;selected=job;break
    try:
        yield selected
    finally:
        if stream is not None:
            fcntl.flock(stream.fileno(),fcntl.LOCK_UN);stream.close()


def running(plan,job):
    path=lock_path(plan,job)
    if not path.is_file():
        return False
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
    return False


def worker_loop(args,plan,execute):
    done=0;started=time.time();waiting=False;scope_complete=False
    jobs=selected_jobs(plan,args)
    names=selected_datasets(plan,args)
    while True:
        with claim(plan,args.worker_id,jobs) as job:
            if job is None:
                if all(completed(plan,item) is not None for item in jobs):
                    scope_complete=True
                    break
                if getattr(args,'exit_when_idle',True):
                    break
                if not waiting:
                    print('Other workers hold remaining tasks; waiting for completion or released tasks',flush=True)
                waiting=True
            else:
                waiting=False
                print(f"Worker {args.worker_id:02d}: {dataset_label(job['dataset'])} seed={job['seed']} entity={job['entity_index']}",flush=True)
                execute(job)
                if completed(plan,job) is None:
                    raise RuntimeError('Training did not produce both an entity record and score')
                done+=1
        if waiting:
            time.sleep(args.poll_seconds)
    queue_complete=all(completed(plan,item) is not None for item in plan['jobs'])
    write_json(plan_root(plan)/'workers'/f'worker_{args.worker_id:02d}'/'status.json',{
        'plan_id':plan['plan_id'],'hostname':socket.gethostname(),'completed':queue_complete,
        'scope_completed':scope_complete,'datasets':names,
        'exit_reason':'queue_complete' if queue_complete else ('datasets_complete' if scope_complete else 'idle'),
        'device':getattr(args,'device','unknown'),
        'jobs_this_invocation':done,'wall_seconds':time.time()-started})
    message='All tasks complete' if queue_complete else ('Selected dataset tasks complete; worker exits' if scope_complete else 'Worker exits: other workers hold selected remaining tasks; experiment still in progress')
    print(message,flush=True)


def run(args):
    plan=read_plan(args.plan)
    if plan.get('kind')!='semsphere_multi_dataset':
        raise ValueError('Use a plan generated by scripts/prepare_experiment.sh')
    if not 1<=args.worker_id<=plan['nodes'] or args.poll_seconds<=0:
        raise ValueError('Invalid worker-id or poll-seconds')
    jobs=selected_jobs(plan,args)
    names=selected_datasets(plan,args)
    if args.dry_run:
        for job in jobs:
            print(f"{dataset_label(job['dataset'])} seed={job['seed']} entity={job['entity_index']}")
        return
    with worker_lock(plan,args.worker_id):
        if all(completed(plan,job) is not None for job in jobs):
            print('Selected dataset tasks already complete');return
        validate_sources(plan)
        from models.SemSphere.network import model as network
        requested=getattr(args,'device','auto')
        args.device=select_device(requested)
        print(f'Worker device: {args.device} (requested {requested}); datasets:', ', '.join(dataset_label(name) for name in names),flush=True)
        needs_e5=any(entry['config'].get('text_model','intfloat/e5-base-v2') not in ('hash','fallback','none')
                     for entry in (plan['datasets'][name] for name in names))
        if needs_e5 and network.SentenceTransformer is None:
            raise RuntimeError(f'sentence-transformers cannot load: {network.SENTENCE_TRANSFORMERS_IMPORT_ERROR}')
        import SemSphere_method as method
        datasets={};profiles={}
        def execute(job):
            name=job['dataset'];entry=plan['datasets'][name]
            if name not in datasets:
                data=load_dataset(name,entry['data_root'])
                if list(map(str,data.entity_ids))!=entry['entity_ids']:
                    raise ValueError(f'{name} entity order differs from plan')
                for relative,expected in entry['data_sha256'].items():
                    if sha256(data.rootdir/relative)!=expected:
                        raise ValueError(f'{name} data differs from plan: {relative}')
                datasets[name]=data
                profiles[name]=plan_root(plan)/'workers'/f'worker_{args.worker_id:02d}'/f'{name}_config.json'
                write_json(profiles[name],entry['config'])
            method.load_dataset=lambda selected:datasets[selected]
            sys.argv=['SemSphere_method.py','--selected_dataset',name,'--device',args.device,
                '--seed',str(job['seed']),'--study_name',plan['study_name'],
                '--experiment_description',plan['study_name'],'--num_workers',str(plan['num_workers']),
                '--config_json',str(profiles[name]),'--entity_indices',str(job['entity_index']),
                '--entity_records_dir',str(record_dir(plan,name)),'--summary_suffix',f'_worker_{args.worker_id:02d}',
                '--cluster_plan_id',plan['plan_id'],'--cluster_worker_id',str(args.worker_id),'--disable_timing']
            method.run_semsphere(method.parse_args())
        worker_loop(args,plan,execute)


def status(args):
    plan=read_plan(args.plan)
    total=0
    names=selected_datasets(plan,args)
    for name in names:
        jobs=[job for job in plan['jobs'] if job['dataset']==name]
        done=sum(completed(plan,job) is not None for job in jobs)
        active=sum(completed(plan,job) is None and running(plan,job) for job in jobs)
        total+=done
        print(f'{dataset_label(name)}: {done}/{len(jobs)} complete; running={active}; pending={len(jobs)-done-active}')
    print(f'Total selected: {total}/{len(selected_jobs(plan,args))} complete')


def merge(args):
    from experiments.cluster_results import read_entity_record,aggregate_records,write_summary
    import numpy as np
    import pandas as pd
    plan=read_plan(args.plan)
    paths=plan_root(plan)/'entities'
    records={}
    jobs=selected_jobs(plan,args)
    names=selected_datasets(plan,args)
    expected={(job['dataset'],job['seed'],job['entity_index']) for job in jobs}
    for path in paths.rglob('seed_*_entity_*.json'):
        record=read_entity_record(path)
        if record['dataset'] not in names:
            continue
        key=(record['dataset'],record['seed'],record['entity_index'])
        if key not in expected or record['plan_id']!=plan['plan_id']:
            raise ValueError(f'Record from another plan: {path}')
        if key in records:
            raise ValueError(f'Duplicate record: {key}')
        records[key]=record
    missing=[job for job in jobs if completed(plan,job) is None]
    if missing or set(records)!=expected:
        raise ValueError(f'Incomplete experiment: {len(missing)} tasks lack complete records/scores. No final summaries written.')
    results={}
    for name in names:
        entry=plan['datasets'][name]
        results[name]=[]
        for seed in plan['seeds']:
            group=[records[(name,seed,index)] for index in range(len(entry['entity_ids']))]
            if any(r['entity_count']!=len(entry['entity_ids']) for r in group):
                raise ValueError('Entity count differs from plan')
            results[name].append(aggregate_records(group))
    directory=Path(args.output_dir) if args.output_dir else ROOT/'results'
    for name,values in results.items():
        output=directory/f"SemSphere_{name}_summary_{plan['study_name']}.csv"
        write_summary(output,values)
        rows=[]
        print(dataset_label(name))
        for value in values:
            print(f"seed={value['seed']} | VUS-ROC={value['Test VUS-ROC']:.5f} | VUS-PR={value['Test VUS-PR']:.5f} | RPA F1={value['RPA F1']:.5f}")
        for label,key in [('VUS-ROC','Test VUS-ROC'),('VUS-PR','Test VUS-PR'),('RPA F1','RPA F1')]:
            samples=np.asarray([value[key] for value in values],dtype=float)
            valid=samples[~np.isnan(samples)]
            mean=float(valid.mean()) if len(valid) else float('nan')
            std=float(valid.std(ddof=0)) if len(valid) else float('nan')
            rows.append({'Metric':label,'Mean':mean,'Std':std,'Valid Seeds':len(valid),'Total Seeds':len(samples)})
            print(f'{label}: {mean:.5f} ± {std:.5f} (valid seeds {len(valid)}/{len(samples)})')
        statistics=directory/f"SemSphere_{name}_statistics_{plan['study_name']}.csv"
        pd.DataFrame(rows).set_index('Metric').to_csv(statistics)
        print(f'Full CSV: {output}; statistics CSV: {statistics}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    plan=commands.add_parser('prepare')
    plan.add_argument('--datasets','--dataset',nargs='+',default=['KPI'])
    plan.add_argument('--seeds',default='1,2,3,4,5')
    plan.add_argument('--nodes',type=int,default=20)
    plan.add_argument('--num-workers',type=int,default=2)
    plan.add_argument('--study',default=None)
    plan.add_argument('--output',default=None)
    plan.add_argument('--config',default=None)
    plan.add_argument('--data-root',default=None)
    plan.add_argument('--data-base',default=None)
    toggle=plan.add_mutually_exclusive_group()
    toggle.add_argument('--enable_infonce',dest='use_infonce',action='store_true')
    toggle.add_argument('--disable_infonce',dest='use_infonce',action='store_false')
    plan.set_defaults(use_infonce=None)
    for command in ('run','status','merge'):
        item=commands.add_parser(command)
        item.add_argument('--plan',required=True)
        item.add_argument('--datasets','--dataset','--selected_dataset',nargs='+',default=None,
                          help='Only run, inspect, or merge these datasets from the existing plan')
        if command=='run':
            item.add_argument('--worker-id',type=int,required=True)
            item.add_argument('--device',choices=['auto','cpu','cuda'],default='auto',
                              help='auto uses available CUDA, otherwise CPU; model config is preserved')
            idle=item.add_mutually_exclusive_group()
            idle.add_argument('--exit-when-idle',dest='exit_when_idle',action='store_true',
                              help='Exit when no unfinished job can be claimed (default)')
            idle.add_argument('--wait-when-idle',dest='exit_when_idle',action='store_false',
                              help='Wait for completion or a released job when others hold remaining tasks')
            item.set_defaults(exit_when_idle=True)
            item.add_argument('--poll-seconds',type=float,default=5.)
            item.add_argument('--dry-run',action='store_true')
        if command=='merge':
            item.add_argument('--output-dir',default=None)
    args=parser.parse_args()
    os.chdir(ROOT)
    {'prepare':prepare,'run':run,'status':status,'merge':merge}[args.command](args)


if __name__=='__main__':
    main()
