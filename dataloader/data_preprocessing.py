"""Data loading and preprocessing for KPI, ASD, Exathlon, and WADI."""
from functools import lru_cache
from pathlib import Path
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from merlion.utils import TimeSeries
from merlion.transform.normalize import MeanVarNormalize


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data/iops_competition"


class KPISeries:
    def __init__(self, rootdir=None):
        self.rootdir = Path(rootdir) if rootdir else DEFAULT_DATA_ROOT
        if rootdir is None:
            data_root = DEFAULT_DATA_ROOT.parent
            for candidate in [DEFAULT_DATA_ROOT, data_root/'IOpsCompetition', data_root/'KPI', data_root/'kpi', data_root]:
                if all((candidate/f"phase2_{split}.csv").is_file() for split in ('train','test')):
                    self.rootdir = candidate
                    break
        paths = [self.rootdir / f"phase2_{split}.csv" for split in ("train", "test")]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Missing KPI file: {path}. Extract phase2.zip before running.")
        self.frames = [pd.read_csv(path) for path in paths]
        for frame in self.frames:
            if "KPI ID" not in frame or "timestamp" not in frame or len(frame.columns) < 4:
                raise ValueError("Expected original phase2 CSV columns: timestamp, value, label, KPI ID")
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s")
            columns = frame.columns.values
            columns[2] = "anomaly"
            frame.columns = columns
            frame["anomaly"] = frame["anomaly"].astype(bool)
        self.kpi_ids = sorted(self.frames[0]["KPI ID"].unique())
        if set(self.kpi_ids) != set(self.frames[1]["KPI ID"].unique()):
            raise ValueError("Train/test KPI IDs differ; refusing to silently omit entities")
        if not self.kpi_ids:
            raise ValueError("KPI dataset is empty")
        self.groups = [frame.groupby("KPI ID", sort=False) for frame in self.frames]

    def __len__(self):
        return len(self.kpi_ids)

    def __getitem__(self, index):
        kpi = self.kpi_ids[index]
        parts = []
        for is_train, group in zip((True, False), self.groups):
            frame = group.get_group(kpi).drop(columns="KPI ID").copy()
            frame.insert(3, "trainval", is_train)
            parts.append(frame)
        full = pd.concat(parts).set_index("timestamp")
        metadata = full[["anomaly", "trainval"]]
        return full[[c for c in full.columns if c not in metadata.columns]], metadata

    def costs(self):
        return [sum(len(group.get_group(kpi)) for group in self.groups) for kpi in self.kpi_ids]


ROOT = Path(__file__).resolve().parents[1]
ALIASES = {'kpi':'IOpsCompetition','iopscompetition':'IOpsCompetition',
           'asd':'ASD','exathlon':'Exathlon'}


def canonical_dataset(name):
    try:
        return ALIASES[name.lower()]
    except KeyError as error:
        raise ValueError('Entity loaders support KPI, ASD, Exathlon; use wadi() for WADI') from error


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
        if self.dataset == 'ASD':
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
        if self.dataset == 'ASD':
            return [self.rootdir/f'{entity}_{suffix}.pkl' for suffix in ('train','test','test_label')]
        return [self.rootdir/entity/name for name in ('train.npy','test.npy','y_test.npy')]

    def _load_arrays(self,index):
        paths=self.entity_files(index)
        if self.dataset == 'ASD':
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


def norm(train, test):
    scaler = StandardScaler()
    
    scaler.fit(train)
    train_data = scaler.transform(train)
    test_data = scaler.transform(test)
    return train_data, test_data



def wadi_preprocessing():
    # preprocess for WADI. WADI.A2_19Nov2019
    dataset_folder = os.path.join('../data/', 'wadi')

    train_df = pd.read_csv(os.path.join(dataset_folder, 'WADI_14days_new.csv'), index_col=0, header=0)
    test_df = pd.read_csv(os.path.join(dataset_folder, 'WADI_attackdataLABLE.csv'), index_col=0, header=1)

    train_df = train_df.iloc[:, 2:]
    test_df = test_df.iloc[:, 2:]

    train_df = train_df.fillna(train_df.mean(numeric_only=True))
    test_df = test_df.fillna(test_df.mean(numeric_only=True))
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    # trim column names
    train_df = train_df.rename(columns=lambda x: x.strip())
    test_df = test_df.rename(columns=lambda x: x.strip())

    train_df['label'] = np.zeros(len(train_df))
    test_df['label'] = np.where(test_df['Attack LABLE (1:No Attack, -1:Attack)'] == -1, 1, 0)
    test_df = test_df.drop(columns=['Attack LABLE (1:No Attack, -1:Attack)'])

    output_dir = '../data/wadi/'
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'WADI_train.csv'))
    test_df.to_csv(os.path.join(output_dir, 'WADI_test.csv'))

def fetching_timestamp_for_d3r(dataset):
    if dataset != 'WADI':
        raise ValueError('Timestamp extraction is only supported for WADI')
    dataset_folder = os.path.join('data/', 'wadi')
    train_df = pd.read_csv(os.path.join(dataset_folder, 'WADI_14days_new.csv'), index_col=0, header=0)
    test_df = pd.read_csv(os.path.join(dataset_folder, 'WADI_attackdataLABLE.csv'), index_col=0, header=1)
    train_date = train_df.iloc[:, 0]
    train_time = train_df.iloc[:, 1]
    train_timestamps = train_date + ' ' + '00:' + train_time
    test_date = test_df.iloc[:, 0]
    test_time = test_df.iloc[:, 1]
    test_timestamps = test_date + ' ' + '00:' + test_time
    return (train_timestamps, test_timestamps)



def wadi():
    dataset_folder = os.path.join('data/', 'wadi')
    train_df = pd.read_csv(os.path.join(dataset_folder, 'WADI_train.csv'))
    train_df = np.array(train_df.set_index('Row'))
    train_data = train_df[:, :127]
    test_df = pd.read_csv(os.path.join(dataset_folder, 'WADI_test.csv'))
    
    test_df = np.array(test_df.set_index('Row '))
    test_data = test_df[:, :127]

    train_data, test_data = norm(train_data, test_data)

    train_labels = train_df[:, 127]
    test_labels = test_df[:, 127]

    return train_data, test_data, train_labels, test_labels



def other_datasets(time_series, meta_data):
    train_time_series_ts = TimeSeries.from_pd(time_series[meta_data.trainval])
    test_time_series_ts = TimeSeries.from_pd(time_series[~meta_data.trainval])
    train_labels = TimeSeries.from_pd(meta_data.anomaly[meta_data.trainval])
    test_labels = TimeSeries.from_pd(meta_data.anomaly[~meta_data.trainval])
    mvn = MeanVarNormalize()
    mvn.train(train_time_series_ts)


    # salesforce-merlion==2.0.0
    # 获取所有列名，确保顺序与 to_pd() 后的 numpy 数组一致
    bias, scale = mvn.bias, mvn.scale
    column_names = train_time_series_ts.names
    # 将字典形式的 bias 和 scale 转换为与数据维度对齐的 numpy 数组
    bias = np.array([mvn.bias[name] for name in column_names])
    scale = np.array([mvn.scale[name] for name in column_names])

    # 此时 bias 和 scale 的 shape 都是 (D,)，NumPy 会自动进行广播计算
    train_time_series = train_time_series_ts.to_pd().to_numpy()
    train_data = (train_time_series - bias) / scale

    test_time_series = test_time_series_ts.to_pd().to_numpy()
    test_data = (test_time_series - bias) / scale
    train_labels = train_labels.to_pd().to_numpy()
    test_labels = test_labels.to_pd().to_numpy()

    return train_data, test_data, train_labels, test_labels


    train_time_series = train_time_series_ts.to_pd().to_numpy()
    train_data = (train_time_series - bias) / scale

    test_time_series = test_time_series_ts.to_pd().to_numpy()
    test_data = (test_time_series - bias) / scale
    
    train_labels = train_labels.to_pd().to_numpy()
    test_labels = test_labels.to_pd().to_numpy()

    return train_data, test_data, train_labels, test_labels

def other_datasets_no_Normalize(time_series, meta_data):
    train_time_series_ts = TimeSeries.from_pd(time_series[meta_data.trainval])
    test_time_series_ts = TimeSeries.from_pd(time_series[~meta_data.trainval])
    train_labels = TimeSeries.from_pd(meta_data.anomaly[meta_data.trainval])
    test_labels = TimeSeries.from_pd(meta_data.anomaly[~meta_data.trainval])

    train_data = train_time_series_ts.to_pd().to_numpy()
    test_data = test_time_series_ts.to_pd().to_numpy()

    train_labels = train_labels.to_pd().to_numpy()
    test_labels = test_labels.to_pd().to_numpy()

    return train_data, test_data, train_labels, test_labels

if __name__ == "__main__":
    
    wadi_preprocessing()
