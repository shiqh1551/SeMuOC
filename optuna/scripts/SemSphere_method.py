"""Run fixed configurations across independent random seeds."""
import argparse
import copy
import logging
import subprocess
import sys
import json
import os
from pathlib import Path

import pandas as pd
import torch
import optuna_SemSphere as search


def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--selected_dataset',default='ASD',choices=['KPI','IOpsCompetition','SMD','ASD','Exathlon','SWaT','WADI','ESA','UCR'])
    parser.add_argument('--study_name',default='manual')
    parser.add_argument('--experiment_description',default='test')
    parser.add_argument('--trial_id',default='manual')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--internal_seed',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--seeds',default=None)
    parser.add_argument('--njobs',type=int,default=1)
    parser.add_argument('--torch_threads',type=int,default=1)
    parser.add_argument('--num_workers',type=int,default=0)
    parser.add_argument('--text_model',default=None)
    parser.add_argument('--data_root',default=None)
    for name in ('num_epoch','d_model','tcn_layers','attention_heads','project_dim','batch_size','window_size','time_step','vus_window','oc_hidden_dim'):
        parser.add_argument('--'+name,type=int,default=None)
    for name in ('lambda_align','lambda_oc','score_beta'):
        parser.add_argument('--'+name,type=float,default=None)
    parser.add_argument('--lambda_recon',type=float,choices=[1.0],default=1.0)
    parser.add_argument('--score_alpha',type=float,choices=[1.0],default=1.0)
    parsed=parser.parse_args()
    args=search.parse_args([])
    vars(args).update(vars(parsed))
    args.seed_values=[int(v) for v in args.seeds.split(',')] if args.seeds else [args.seed]
    if not args.seed_values or len(set(args.seed_values))!=len(args.seed_values):
        parser.error('Distinct seeds required')
    if args.njobs<1 or args.torch_threads<1 or args.num_workers<0:
        parser.error('Positive njobs/torch_threads and nonnegative num_workers required')
    for name in ('study_name','trial_id','experiment_description'):
        value=getattr(args,name)
        if not value or Path(value).name!=value or value in ('.','..'):
            parser.error(name+' must be a filename component')
    args.seeds=','.join(map(str,args.seed_values))
    if args.selected_dataset=='KPI':
        args.selected_dataset='IOpsCompetition'
    return args


def run_semsphere(args):
    torch.set_num_threads(args.torch_threads)
    ctx=search.build_context(copy.deepcopy(args))
    ctx.configs.trial_id=args.trial_id
    ctx.configs.study_name=f'{ctx.dataset_name}_{args.study_name}'
    for field in ('lambda_align','lambda_oc','score_beta','oc_hidden_dim'):
        value=getattr(args,field)
        if value is not None:
            setattr(ctx.configs,field,value)
    data=search.load_dataset(ctx.dataset_name,args.data_root)
    # Match the Optuna entry: honor an explicit VUS tolerance independently of the training window.
    with search.fixed_vus(ctx.configs):
        result=search.run_one_seed(ctx,args.seed,data)
    result['Hyper-parameter']=str(vars(ctx.configs))
    return result


def main():
    args=parse_args()
    os.chdir(search.ROOT)
    output_dir=Path('results')/'SemSphere'/args.selected_dataset/args.study_name/f'trial_{args.trial_id}'
    output_dir.mkdir(parents=True,exist_ok=True)
    if args.internal_seed:
        logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
            handlers=[logging.FileHandler(output_dir/f'training_{args.experiment_description}_seed_{args.seed}.log',mode='w'),
                      logging.StreamHandler()],force=True)
        logging.info('Starting SemSphere dataset=%s seed=%s',args.selected_dataset,args.seed)
        result=run_semsphere(args)
        (output_dir/f'metrics_{args.experiment_description}_seed_{args.seed}.json').write_text(json.dumps(result,indent=2))
        logging.info('Finished seed=%s',args.seed)
        return
    children=[]
    active=[]
    try:
        for seed in args.seed_values:
            if len(active)>=args.njobs:
                code=active.pop(0).wait()
                if code:
                    raise RuntimeError(f'Seed process failed: exit code {code}')
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),
                *sys.argv[1:],'--seeds',str(seed),'--seed',str(seed),'--njobs','1','--internal_seed'])
            active.append(child);children.append(child)
        for child in active:
            code=child.wait()
            if code:
                raise RuntimeError(f'Seed process failed: exit code {code}')
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            child.wait()
    results=[json.loads((output_dir/f'metrics_{args.experiment_description}_seed_{seed}.json').read_text()) for seed in args.seed_values]
    summary=output_dir/f'summary_{args.experiment_description}.csv'
    frame=pd.DataFrame({f'seed_{seed}':pd.Series(result) for seed,result in zip(args.seed_values,results)})
    numeric=frame.apply(pd.to_numeric,errors='coerce')
    frame['mean']=numeric.mean(axis=1)
    frame['std']=numeric.std(axis=1,ddof=0)
    frame.to_csv(summary)
    print(f'Combined seed results saved to {summary}')


if __name__=='__main__':
    main()
