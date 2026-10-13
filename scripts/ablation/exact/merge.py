"""Aggregate five isolated seed runs; refuse incomplete or mismatched ablations."""
import argparse
import json
import os
from pathlib import Path
import runpy
from types import SimpleNamespace
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
METRICS=['RPA F1','Test VUS-ROC','Test VUS-PR']
IGNORE={'study_name','trial_id','profile_timing','num_workers'}

def atomic_text(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(text);tmp.replace(path)

def merge(root,version,dataset,case_group="original"):
    manifest=json.loads((root/'scripts/ablation/exact/manifest.json').read_text())
    entry=manifest['versions'][version]['datasets'][dataset]
    registry=runpy.run_path(str(root/'models/SemSphere_ablation/registry.py'))
    all_rows=[];prepared=[];roots=set()
    cases=manifest['combination_cases'] if case_group=='combinations' else manifest['cases']
    if case_group=='all':cases=manifest['cases']+manifest['combination_cases']
    for case in cases:
        expected=vars(registry['apply_ablation_case'](SimpleNamespace(**entry['config']),case))
        values={};sources=[]
        for seed in manifest['seeds']:
            folder=root/'results_SemSphere_ablation'/dataset/f'{entry["run_prefix"]}_seed{seed}'/case
            meta=json.loads((folder/'config.json').read_text())
            actual=meta['config']
            for key,val in expected.items():
                if key not in IGNORE and actual.get(key)!=val:
                    raise ValueError(f'Configuration mismatch: {folder}: {key}={actual.get(key)!r}, expected {val!r}')
            roots.add(meta.get('data_root'))
            metric_file=folder/f'metrics_seed_{seed}.json'
            data=json.loads(metric_file.read_text())
            for metric in METRICS:
                if not isinstance(data.get(metric),(int,float)) or not np.isfinite(data[metric]):
                    raise ValueError(f'Missing/nonfinite {metric}: {metric_file}')
            values[f'seed_{seed}']=pd.Series(data)
            sources.append(str(metric_file))
        frame=pd.DataFrame(values)
        numeric=frame.apply(pd.to_numeric,errors='coerce')
        frame['mean']=numeric.mean(axis=1);frame['std']=numeric.std(axis=1,ddof=0)
        row={'case':case,'model':registry['ABLATION_CASES'][case]['label'],'n_seeds':5}
        for metric in METRICS:
            row[metric]=frame.loc[metric,'mean'];row[metric+' std']=frame.loc[metric,'std']
        all_rows.append(row);prepared.append((case,frame,sources))
    if len(roots)!=1:raise ValueError('Dataset roots differ between seeds/cases')
    suffix='' if case_group=='original' else '_'+case_group
    out=root/'results_SemSphere_ablation_groups'/dataset/f'ablation_exact_{version}{suffix}'
    for case,frame,sources in prepared:
        atomic_text(out/case/'summary.csv',frame.to_csv())
        atomic_text(out/case/'sources.json',json.dumps({'seed_metrics':sources,'source_trial':entry['study'],'trial':entry['trial']},indent=2)+'\n')
    atomic_text(out/'ablation_summary.csv',pd.DataFrame(all_rows).to_csv(index=False))
    atomic_text(out/'baseline.json',json.dumps(entry,ensure_ascii=False,indent=2)+'\n')
    print('Five-seed ablation summary:',out/'ablation_summary.csv')
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--version',required=True,choices=['v4','v5','v6','v7','v8','v9'])
    p.add_argument('--dataset',required=True,choices=['KPI','IOpsCompetition','ASD','Exathlon','WADI'])
    p.add_argument('--case_group',choices=['original','combinations','all'],default='original')
    a=p.parse_args();merge(a.root,a.version,'IOpsCompetition' if a.dataset=='KPI' else a.dataset,a.case_group)
