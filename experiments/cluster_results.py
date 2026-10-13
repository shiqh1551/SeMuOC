"""Portable entity records and exact original multi-entity metric aggregation."""
import json
from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from merlion.evaluate.anomaly import TSADScoreAccumulator as ScoreAcc, ScoreType
from models.reasonable_metric import reasonable_accumulator


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, timedelta):
        return {"__timedelta_seconds__": value.total_seconds()}
    raise TypeError(f"Unsupported record value: {type(value)}")


def json_hook(value):
    if set(value) == {"__timedelta_seconds__"}:
        return timedelta(seconds=value["__timedelta_seconds__"])
    return value


def save_entity_record(directory, record):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"seed_{record['seed']}_entity_{record['entity_index']:03d}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, default=json_default, ensure_ascii=False, indent=2))
    temporary.replace(path)
    return path


def read_entity_record(path):
    return json.loads(Path(path).read_text(), object_hook=json_hook)


def aggregate_records(records):
    """Match the entry point: mean continuous metrics, sum score accumulators."""
    from SemSphere_method import safe_weighted_mean, safe_nanmean
    if not records:
        raise ValueError("No records to merge")
    reference = records[0]
    for record in records:
        for key in ("dataset", "seed", "plan_id", "entity_count", "config", "experiment_description"):
            if record[key] != reference[key]:
                raise ValueError(f"Incompatible entity records: {key}")
    rpa, pa, pw = [sum((ScoreAcc(**r['accumulators'][key]) for r in records), ScoreAcc())
                   for key in ("rpa", "pa", "pw")]
    reasonable = sum((reasonable_accumulator(**r['reasonable']) for r in records), reasonable_accumulator())
    weights = [r['anomaly_windows'] for r in records]
    precision = safe_weighted_mean([r['affiliation']['precision'] for r in records], weights)
    recall = safe_weighted_mean([r['affiliation']['recall'] for r in records], weights)
    aff_f1 = 0 if np.isnan(precision) or np.isnan(recall) or precision + recall == 0 else 2*precision*recall/(precision+recall)
    result = {
        "Hyper-parameter": '\n'.join('%s:%s' % item for item in reference['config'].items()),
        "Polluted-rate": reference.get('polluted_rate', 0.0),
        "seed": reference['seed'],
        "Train Time": timedelta(seconds=sum(r['elapsed_seconds'] for r in records)),
        "UCR Accuracy": round(reasonable.get_all_metrics()['accuracy'], 5),
        "Affiliation Precision": round(precision, 5), "Affiliation Recall": round(recall, 5),
        "Affiliation F1": round(aff_f1, 5),
    }
    for prefix, accumulator, kind in [("RPA",rpa,ScoreType.RevisedPointAdjusted),
                                       ("PA",pa,ScoreType.PointAdjusted), ("Point-wise",pw,ScoreType.Pointwise)]:
        for label, method in [("Precision","precision"),("Recall","recall"),("F1","f1")]:
            result[f"{prefix} {label}"] = round(getattr(accumulator, method)(kind), 5)
    for label, options in [("balanced",{}),("high precision",{'fp_weight':.22}),("high recall",{'fn_weight':2.})]:
        result[f"NAB Score ({label})"] = round(rpa.nab_score(**options), 5)
    for label, key in [("ROC-AUC","ROC_AUC"),("PR-AUC","PR_AUC"),("VUS-ROC","VUS_ROC"),("VUS-PR","VUS_PR")]:
        result[f"Test {label}"] = round(safe_nanmean([r['continuous'][key] for r in records]), 5)
    return result


def write_summary(path, results):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    frame = pd.DataFrame({f"SemSphere{i}": pd.Series(result, dtype=object) for i,result in enumerate(results)})
    temporary = path.with_suffix('.csv.tmp')
    frame.to_csv(temporary, index=True)
    temporary.replace(path)
