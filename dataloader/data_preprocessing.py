import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from merlion.utils import TimeSeries
from merlion.transform.normalize import MeanVarNormalize


def norm(train, test):
    scaler = StandardScaler()
    # scaler.fit(np.concatenate((train, test), axis=0))
    scaler.fit(train)
    train_data = scaler.transform(train)
    test_data = scaler.transform(test)
    return train_data, test_data

def swat_preprocessing():
    # the following code is adapted from the source code in [Zhihan Li et al. KDD21]
    # preprocess for SWaT. SWaT.A2_Dec2015, version 0
    dataset_folder = os.path.join('../data/', 'swat')

    test_df = pd.read_excel(os.path.join(dataset_folder, 'SWaT_Dataset_Attack_v0.xlsx'), header=1)

    test_df = test_df.set_index(' Timestamp')
    test_df['label'] = np.where(test_df['Normal/Attack'] == 'Attack', 1, 0)
    # test_df.apply(lambda x: 1 if test_df['Normal/Attack'] == 'Attack' else 0)
    test_df = test_df.drop('Normal/Attack', axis=1)
    assert test_df.shape == (449919, 52)

    train_df = pd.read_excel(os.path.join(dataset_folder, 'SWaT_Dataset_Normal_v0.xlsx'), header=1)
    # train_df = train_df.drop(columns=['Unnamed: 0', 'Unnamed: 52'])
    train_df = train_df.set_index(' Timestamp')
    train_df['label'] = np.where(train_df['Normal/Attack'] == 'Attack', 1, 0)
    train_df = train_df.drop('Normal/Attack', axis=1)

    # following [Zhihan Li et al. KDD21] & [Dan Li. ICANN. 2019]
    # fow SWaT data, due to the cold start of the system, starting point is 21600
    train_df = train_df.iloc[21600:]
    assert train_df.shape == (475200, 52)

    output_dir = '../data/swat/'
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'SWaT_train.csv'))
    test_df.to_csv(os.path.join(output_dir, 'SWaT_test.csv'))


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
    if dataset == 'SWaT':
        dataset_folder = os.path.join('data/', 'swat')
        train_df = pd.read_csv(os.path.join(dataset_folder, 'SWaT_train.csv'))
        train_timestamps = train_df[' Timestamp'].values
        test_df = pd.read_csv(os.path.join(dataset_folder, 'SWaT_test.csv'))
        test_timestamps = test_df[' Timestamp'].values
    elif dataset == 'WADI':
        dataset_folder = os.path.join('data/', 'wadi')
        train_df = pd.read_csv(os.path.join(dataset_folder, 'WADI_14days_new.csv'), index_col=0, header=0)
        test_df = pd.read_csv(os.path.join(dataset_folder, 'WADI_attackdataLABLE.csv'), index_col=0, header=1)
        train_date = train_df.iloc[:, 0]
        train_time = train_df.iloc[:, 1]
        train_timestamps = train_date + ' ' + '00:' + train_time
        test_date = test_df.iloc[:, 0]
        test_time = test_df.iloc[:, 1]
        test_timestamps = test_date + ' ' + '00:' + test_time
    else: #ESA
        dataset_folder = os.path.join('data/', 'esa')
        train_df = pd.read_csv(os.path.join(dataset_folder, 'esa_train_labels.csv'))
        train_timestamps = train_df['datetime'].values
        test_df = pd.read_csv(os.path.join(dataset_folder, 'esa_test_labels.csv'))
        test_timestamps = test_df['datetime'].values
    return train_timestamps, test_timestamps

def swat():
    dataset_folder = os.path.join('data/', 'swat')
    train_df = pd.read_csv(os.path.join(dataset_folder, 'SWaT_train.csv'))
    train_df = np.array(train_df.set_index(' Timestamp'))
    train_data = train_df[:, :51]
    test_df = pd.read_csv(os.path.join(dataset_folder, 'SWaT_test.csv'))
    test_df = np.array(test_df.set_index(' Timestamp'))
    test_data = test_df[:, :51]

    # scaler = StandardScaler()
    # scaler.fit(np.concatenate((train_data, test_data), axis=0))
    # train_data = scaler.transform(train_data)
    # test_data = scaler.transform(test_data)
    train_data, test_data = norm(train_data, test_data)

    train_labels = train_df[:, 51]
    test_labels = test_df[:, 51]

    return train_data, test_data, train_labels, test_labels


def wadi():
    dataset_folder = os.path.join('data/', 'wadi')
    train_df = pd.read_csv(os.path.join(dataset_folder, 'WADI_train.csv'))
    train_df = np.array(train_df.set_index('Row'))
    train_data = train_df[:, :127]
    test_df = pd.read_csv(os.path.join(dataset_folder, 'WADI_test.csv'))
    # print(test_df.columns)
    test_df = np.array(test_df.set_index('Row '))
    test_data = test_df[:, :127]

    train_data, test_data = norm(train_data, test_data)

    train_labels = train_df[:, 127]
    test_labels = test_df[:, 127]

    return train_data, test_data, train_labels, test_labels

def esa():
    dataset_folder = os.path.join('data/', 'esa')
    train_data = pd.read_csv(os.path.join(dataset_folder, 'esa_train.csv'))
    train_data = np.array(train_data.set_index('datetime'))
    print(train_data.shape)
    test_data = pd.read_csv(os.path.join(dataset_folder, 'esa_test.csv'))
    test_data = np.array(test_data.set_index('datetime'))
    print(test_data.shape)
    train_labels = pd.read_csv(os.path.join(dataset_folder, 'esa_train_labels.csv'))
    train_labels = np.array(train_labels.set_index('datetime'))

    test_labels = pd.read_csv(os.path.join(dataset_folder, 'esa_test_labels.csv'))
    test_labels = np.array(test_labels.set_index('datetime'))

    print(test_data.shape)
    train_data, test_data = norm(train_data, test_data)

    return train_data, test_data, train_labels, test_labels


def other_datasets(time_series, meta_data):
    train_time_series_ts = TimeSeries.from_pd(time_series[meta_data.trainval])
    test_time_series_ts = TimeSeries.from_pd(time_series[~meta_data.trainval])
    train_labels = TimeSeries.from_pd(meta_data.anomaly[meta_data.trainval])
    test_labels = TimeSeries.from_pd(meta_data.anomaly[~meta_data.trainval])
    mvn = MeanVarNormalize()
    mvn.train(train_time_series_ts)
    # salesforce-merlion==1.1.1
    # bias, scale = mvn.bias, mvn.scale

    # salesforce-merlion==1.3.0
    # bias, scale = mvn.bias[0], mvn.scale[0]
    
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

# def other_datasets(time_series, meta_data):
#     train_time_series_ts = TimeSeries.from_pd(time_series[meta_data.trainval])
#     test_time_series_ts = TimeSeries.from_pd(time_series[~meta_data.trainval])
#     train_labels = TimeSeries.from_pd(meta_data.anomaly[meta_data.trainval])
#     test_labels = TimeSeries.from_pd(meta_data.anomaly[~meta_data.trainval])
    
#     mvn = MeanVarNormalize()
#     mvn.train(train_time_series_ts + test_time_series_ts)

#     # --- 自动识别版本逻辑开始 ---
#     column_names = train_time_series_ts.names
    
#     if isinstance(mvn.bias, dict):
#         # 适用于 2.0.0+ 版本：bias 和 scale 是字典
#         bias = np.array([mvn.bias[name] for name in column_names])
#         scale = np.array([mvn.scale[name] for name in column_names])
#     elif isinstance(mvn.bias, (list, np.ndarray)) and len(mvn.bias) > 0:
#         # 适用于 1.3.0 版本：可能是列表形式
#         bias = mvn.bias[0] if not isinstance(mvn.bias, np.ndarray) else mvn.bias
#         scale = mvn.scale[0] if not isinstance(mvn.scale, np.ndarray) else mvn.scale
#     else:
#         # 适用于 1.1.1 版本：直接是数值或已对齐的数组
#         bias = mvn.bias
#         scale = mvn.scale
    # --- 自动识别版本逻辑结束 ---

    # 此时 bias 和 scale 已经根据版本完成对齐
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
    # swat_preprocessing()
    wadi_preprocessing()
