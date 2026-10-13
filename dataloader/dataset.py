import random
import logging
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
from dataloader.augmentations import DataTransform
from sklearn.model_selection import train_test_split
from utils import subsequences, calculate_fid, calculate_mahalanobis
from merlion.transform.normalize import MeanVarNormalize, MinMaxNormalize
from merlion.utils import TimeSeries
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler


class Load_Dataset(Dataset):
    
    def __init__(self, dataset, config):
        super(Load_Dataset, self).__init__()

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)


        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]
        if hasattr(config, 'augmentation'):
            self.aug1, self.aug2 = DataTransform(self.x_data, config)

    def __getitem__(self, index):
        if hasattr(self, 'aug1'):
            return self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index]
        else:
            return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def validation_windows(train_x, train_y, seed):
    """Keep the original validation sampling; handle insufficient class counts."""
    n_samples = len(train_y)
    if n_samples < 2:
        raise ValueError(f"At least two training windows are required for validation; got {n_samples}")
    classes, counts = np.unique(train_y, return_counts=True)
    n_val = int(np.ceil(n_samples * 0.2))
    can_stratify = counts.min() >= 2 and n_val >= len(classes) and n_samples - n_val >= len(classes)
    stratify = train_y if can_stratify else None
    if not can_stratify:
        logging.warning(
            "Validation stratification unavailable: window class counts=%s; "
            "using unstratified 20%% sampling with seed=%s",
            dict(zip(classes.tolist(), counts.tolist())), seed,
        )
    _, val_x, _, val_y = train_test_split(
        train_x, train_y, test_size=0.2, shuffle=True,
        random_state=seed, stratify=stratify,
    )
    return val_x, val_y

# Gives label values (test_y_window) by time window.
def data_generator1(train_data, test_data, train_labels, test_labels, seed, configs):
    test_anomaly_window_num = int(len(np.where(test_labels[1:] != test_labels[:-1])[0]) / 2)

    train_x = subsequences(train_data, configs.window_size, configs.time_step)
    test_x = subsequences(test_data, configs.window_size, configs.time_step)
    train_y = subsequences(train_labels, configs.window_size, configs.time_step)
    test_y = subsequences(test_labels, configs.window_size, configs.time_step)

    train_y_window = np.zeros(train_x.shape[0])
    test_y_window = np.zeros(test_x.shape[0])
    train_anomaly_window_num = 0
    for i, item in enumerate(train_y[:]):
        if sum(item[:configs.time_step]) >= 1:
            train_anomaly_window_num += 1
            train_y_window[i] = 1
        else:
            train_y_window[i] = 0
    for i, item in enumerate(test_y[:]):
        if sum(item[:configs.time_step]) >= 1:
            test_y_window[i] = 1
        else:
            test_y_window[i] = 0
    train_y = train_y_window
    test_y = test_y_window
    val_x, val_y = validation_windows(train_x, train_y, seed)
    train_x = train_x.transpose((0, 2, 1))
    val_x = val_x.transpose((0, 2, 1))
    test_x = test_x.transpose((0, 2, 1))

    train_dat_dict = dict()
    train_dat_dict["samples"] = train_x
    train_dat_dict["labels"] = train_y

    val_dat_dict = dict()
    val_dat_dict["samples"] = val_x
    val_dat_dict["labels"] = val_y

    test_dat_dict = dict()
    test_dat_dict["samples"] = test_x
    test_dat_dict["labels"] = test_y

    train_dataset = Load_Dataset(train_dat_dict, configs)
    val_dataset = Load_Dataset(val_dat_dict, configs)
    test_dataset = Load_Dataset(test_dat_dict, configs)

    num_workers = getattr(configs, "num_workers", 2)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=False,
                                               num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=num_workers)
    return train_loader, val_loader, test_loader, test_anomaly_window_num
