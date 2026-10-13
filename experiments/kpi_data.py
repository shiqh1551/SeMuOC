"""KPI-only loader with the original IOps CSV and dataframe semantics.

Avoids importing unrelated datasets whose class definitions list missing folders.
Normalization/windowing are still provided by the original data pipeline.
"""
from pathlib import Path
import pandas as pd

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
