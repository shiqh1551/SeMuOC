"""Run controlled ablations; seeds within each case execute sequentially."""
import argparse,copy,json,logging,os,subprocess,sys
from pathlib import Path
from collections import deque
from types import SimpleNamespace
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'optuna/scripts'))
import optuna_SemSphere as search
from models.SemSphere_ablation.registry import ABLATION_CASES,apply_ablation_case
from models.SemSphere_ablation.network.model import base_Model

def parse_args(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--selected_dataset',choices=['KPI','IOpsCompetition','ASD','SMD','Exathlon','SWaT','WADI','ESA','UCR'],default='Exathlon')
 p.add_argument('--study_name',default='ablation_rounded_v2')
 p.add_argument('--baseline_config',required=True)
 p.add_argument('--ablation_case',choices=list(ABLATION_CASES),default='06_full')
 p.add_argument('--run_all_ablation',action='store_true',help='Run the original four ablations')
 p.add_argument('--run_combination_ablation',action='store_true',help='Run only Behavior+Recon and Behavior+OC ablations')
 p.add_argument('--seeds',default='1,2,3,4,5')
 p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
 p.add_argument('--njobs',type=int,default=1,help='Parallel cases; seeds within each case are sequential')
 p.add_argument('--num_workers',type=int,default=0)
 p.add_argument('--torch_threads',type=int,default=1)
 p.add_argument('--data_root',default=None)
 p.add_argument('--text_model',default=None)
 p.add_argument('--rerun',action='store_true',help='Recompute matching completed seed results')
 p.add_argument('--internal_case',action='store_true',help=argparse.SUPPRESS)
 for k in ['num_epoch','d_model','tcn_layers','attention_heads','batch_size','window_size','time_step','vus_window','project_dim','oc_hidden_dim']:
  p.add_argument('--'+k,type=int,default=None)
 for k in ['lambda_align','lambda_oc','score_beta']:
  p.add_argument('--'+k,type=float,default=None)
 p.add_argument('--lambda_recon',type=float,choices=[1.0],default=1.)
 p.add_argument('--score_alpha',type=float,choices=[1.0],default=1.)
 a=p.parse_args(argv)
 if a.run_all_ablation and a.run_combination_ablation:p.error('Choose either original or combination ablations')
 a.selected_dataset='IOpsCompetition' if a.selected_dataset=='KPI' else a.selected_dataset
 seeds=[int(x) for x in a.seeds.split(',')]
 if not seeds or len(seeds)!=len(set(seeds)):p.error('Distinct seeds required')
 if a.njobs<1 or a.torch_threads<1 or a.num_workers<0:p.error('Invalid concurrency settings')
 if Path(a.study_name).name!=a.study_name or a.study_name in ['.','..','']:p.error('Invalid study name')
 return a

def effective_context(a):
 base=json.loads(Path(a.baseline_config).read_text())
 if base['dataset']!=a.selected_dataset:raise ValueError('Baseline dataset mismatch')
 args=search.parse_args([])
 for k in vars(args):
  if hasattr(a,k) and getattr(a,k) is not None:setattr(args,k,getattr(a,k))
 for k in ['num_epoch','d_model','tcn_layers','attention_heads','batch_size','window_size','time_step','vus_window','project_dim']:
  if getattr(args,k,None) is None:setattr(args,k,base.get(k))
 original=search.load_configs
 search.load_configs=lambda dataset:SimpleNamespace(**copy.deepcopy(base))
 try:ctx=search.build_context(args)
 finally:search.load_configs=original
 for k in ['lambda_align','lambda_oc','score_beta','oc_hidden_dim']:
  if getattr(a,k) is not None:setattr(ctx.configs,k,getattr(a,k))
 ctx.configs.trial_id=a.ablation_case
 ctx.configs.study_name=f'{a.selected_dataset}_{a.study_name}_{a.ablation_case}'
 apply_ablation_case(ctx.configs,a.ablation_case)
 return ctx

def run_case(a):
 torch.set_num_threads(a.torch_threads)
 ctx=effective_context(a)
 out=ROOT/'results_SemSphere_ablation'/a.selected_dataset/a.study_name/a.ablation_case
 out.mkdir(parents=True,exist_ok=True)
 # Prevent mixing results after any protocol/configuration change.
 metadata={'config':vars(ctx.configs),'data_root':str(Path(a.data_root).resolve()) if a.data_root else None,'seeds':a.seeds}
 config_file=out/'config.json'
 if config_file.exists() and json.loads(config_file.read_text())!=metadata:
  raise ValueError('Ablation settings changed; use a new --study_name')
 config_file.write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
 search.base_Model=base_Model
 data=None;results={}
 for seed in map(int,a.seeds.split(',')):
  target=out/f'metrics_seed_{seed}.json'
  if target.exists() and not a.rerun:
   print(f'Skip completed {a.ablation_case} seed={seed}',flush=True)
   results[seed]=json.loads(target.read_text());continue
  logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
   handlers=[logging.FileHandler(out/f'training_seed_{seed}.log',mode='w'),logging.StreamHandler()],force=True)
  if data is None:data=search.load_dataset(a.selected_dataset,a.data_root)
  logging.info('Starting %s seed=%s',a.ablation_case,seed)
  with search.fixed_vus(ctx.configs):metrics=search.run_one_seed(ctx,seed,data)
  metrics['Hyper-parameter']=str(vars(ctx.configs));results[seed]=metrics
  temp=target.with_suffix('.tmp');temp.write_text(json.dumps(metrics,indent=2));temp.replace(target)
  logging.info('Finished %s seed=%s',a.ablation_case,seed)
 frame=pd.DataFrame({f'seed_{seed}':pd.Series(result) for seed,result in results.items()})
 numeric=frame.apply(pd.to_numeric,errors='coerce');frame['mean']=numeric.mean(axis=1);frame['std']=numeric.std(axis=1,ddof=0)
 frame.to_csv(out/'summary.csv')
 print('Saved',out/'summary.csv',flush=True)

def main():
 a=parse_args();os.chdir(ROOT)
 if a.internal_case:run_case(a);return
 if a.run_combination_ablation:cases=['07_no_behavior_recon','08_no_behavior_oc']
 elif a.run_all_ablation:cases=['02_no_behavior','03_no_recon','04_no_oc','06_full']
 else:cases=[a.ablation_case]
 active=deque();children=[]
 try:
  for case in cases:
   if len(active)>=a.njobs:
    code=active.popleft().wait()
    if code:raise RuntimeError(f'Ablation process failed: {code}')
   child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),*sys.argv[1:],
    '--ablation_case',case,'--internal_case'])
   active.append(child);children.append(child)
  for child in active:
   code=child.wait()
   if code:raise RuntimeError(f'Ablation process failed: {code}')
 finally:
  for child in children:
   if child.poll() is None:child.terminate()
  for child in children:child.wait()
 rows=[]
 for case in cases:
  path=ROOT/'results_SemSphere_ablation'/a.selected_dataset/a.study_name/case/'summary.csv'
  frame=pd.read_csv(path,index_col=0)
  row={'case':case,'model':ABLATION_CASES[case]['label']}
  for metric in ['RPA F1','Test VUS-ROC','Test VUS-PR']:
   row[metric]=frame.loc[metric,'mean'];row[metric+' std']=frame.loc[metric,'std']
  rows.append(row)
 target=ROOT/'results_SemSphere_ablation'/a.selected_dataset/a.study_name/('combination_summary.csv' if a.run_combination_ablation else 'ablation_summary.csv' if a.run_all_ablation else f'summary_{a.ablation_case}.csv')
 pd.DataFrame(rows).to_csv(target,index=False)
 print('Ablation comparison:',target)

if __name__=='__main__':main()
