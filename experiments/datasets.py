"""Load entity-level KPI, SMD, ASD, and Exathlon datasets."""
from functools import lru_cache
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from experiments.kpi_data import KPISeries

ROOT = Path(__file__).resolve().parents[1]
ALIASES = {'kpi':'IOpsCompetition','iopscompetition':'IOpsCompetition',
           'smd':'SMD','asd':'ASD','exathlon':'Exathlon'}


def canonical_dataset(name):
    try:
        return ALIASES[name.lower()]
    except KeyError as error:
        raise ValueError('Supported datasets: KPI, SMD, ASD, Exathlon') from error


def dataset_label(name):
    return 'KPI' if name=='IOpsCompetition' else name


def combine_train_test(train, test, labels):
    
    train.columns=[str(c) for c in train.columns]
    test.columns=[str(c) for c in test.columns]
    frame=pd.concat([train,test]).reset_index()
    if 'index' in frame:
        frame.drop(columns=['index'],inplace=True)
    frame.index=pd.to_datetime(frame.index*60,unit='s')
    frame.index.rename('timestamp',inplace=True)
    metadata=pd.DataFrame({'trainval':frame.index < frame.index[train.shape[0]],
                           'anomaly':np.concatenate([np.zeros(train.shape[0],dtype=int),labels])},index=frame.index)
    return frame,metadata


class ProcessedEntities:
    def __init__(self,name,rootdir=None):
        self.dataset=canonical_dataset(name)
        if self.dataset=='IOpsCompetition':
            raise ValueError('Use KPISeries for KPI')
        base=ROOT/'data'
        if rootdir is not None:
            self.rootdir=Path(rootdir)
        else:
            self.rootdir=next((p for p in [base/self.dataset.lower(),base/self.dataset] if p.is_dir()),base/self.dataset.lower())
        if not self.rootdir.is_dir():
            raise FileNotFoundError(f'Missing dataset directory: {self.rootdir}')
        if self.dataset in ('SMD','ASD'):
            self.entity_ids=sorted(p.name.removesuffix('_train.pkl') for p in self.rootdir.glob('*_train.pkl'))
        else:
            self.entity_ids=sorted(f'app{i:02d}' for i in [1,2,3,4,5,6,9,10])
        if not self.entity_ids:
            raise ValueError(f'No entities found in {self.rootdir}')
        for index in range(len(self)):
            for path in self.entity_files(index):
                if not path.is_file():
                    raise FileNotFoundError(path)
        self._cached_load=lru_cache(maxsize=2)(self._load_arrays)

    def __len__(self):
        return len(self.entity_ids)

    def entity_files(self,index):
        entity=self.entity_ids[index]
        if self.dataset in ('SMD','ASD'):
            return [self.rootdir/f'{entity}_{suffix}.pkl' for suffix in ('train','test','test_label')]
        return [self.rootdir/entity/name for name in ('train.npy','test.npy','y_test.npy')]

    def _load_arrays(self,index):
        paths=self.entity_files(index)
        if self.dataset in ('SMD','ASD'):
            arrays=[]
            for path in paths:
                with path.open('rb') as stream:
                    arrays.append(pickle.load(stream))
            train,test,labels=arrays
        else:
            train,test,labels=[np.concatenate(np.load(path,allow_pickle=True),axis=0) for path in paths]
            labels=labels.copy()
            labels[labels>0]=1
        return pd.DataFrame(train),pd.DataFrame(test),np.array(labels).astype(int)

    def __getitem__(self,index):
        train,test,labels=self._cached_load(index)
        return combine_train_test(train.copy(),test.copy(),labels)

    def sizes(self,index):
        train,test,_=self._cached_load(index)
        return len(train),len(test),train.shape[1]

    def files(self):
        return sorted({path for index in range(len(self)) for path in self.entity_files(index)})


def load_dataset(name,rootdir=None):
    name=canonical_dataset(name)
    if name=='IOpsCompetition':
        dataset=KPISeries(rootdir)
        dataset.entity_ids=list(map(str,dataset.kpi_ids))
        return dataset
    return ProcessedEntities(name,rootdir)


def data_files(dataset):
    if isinstance(dataset,KPISeries):
        return [dataset.rootdir/f'phase2_{split}.csv' for split in ('train','test')]
    return dataset.files()


def entity_sizes(dataset,index):
    if isinstance(dataset,KPISeries):
        kpi=dataset.kpi_ids[index]
        train_size,test_size=[len(group.get_group(kpi)) for group in dataset.groups]
        return train_size,test_size,1
    return dataset.sizes(index)
