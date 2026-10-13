"""Delay unrelated dataset imports in the KPI-only experiment process.

The original utils module imports the dataset package with '*', although the
KPI training pipeline only needs numerical helpers. Some dataset classes list
folders at import time. A package namespace with lazy attributes avoids that
side effect without editing any original data/utility/model implementation.
Dataset-specific benchmark helpers in utils.utils are outside this runner.
"""
import importlib
from pathlib import Path
import sys
import types


def enable_kpi_imports():
    name='ts_datasets.ts_datasets.anomaly'
    if name in sys.modules:
        return
    parent=importlib.import_module('ts_datasets.ts_datasets')
    package=types.ModuleType(name)
    package.__package__=name
    package.__path__=[str(Path(__file__).resolve().parents[1]/'ts_datasets/ts_datasets/anomaly')]
    package.__all__=["TSADBaseDataset"]
    mapping={'TSADBaseDataset':'base','IOpsCompetition':'iops_competition',
             'SMD':'smd','ASD':'asd','Exathlon':'exathlon','UCR':'ucr',
             'SMAP':'smap','MSL':'msl','NAB':'nab','Synthetic':'synthetic'}
    def resolve(attribute):
        if attribute not in mapping:
            raise AttributeError(attribute)
        value=getattr(importlib.import_module(f'{name}.{mapping[attribute]}'),attribute)
        setattr(package,attribute,value)
        return value
    package.__getattr__=resolve
    sys.modules[name]=package
    setattr(parent,'anomaly',package)
