from typing import Dict, List
import pandas as pd


def append_fold_result(results_csv: str, row: Dict) -> None:
    df = pd.DataFrame([row])
    try:
        from pandas.io.common import file_exists
    except Exception:
        from pandas.io.common import get_handle
        def file_exists(path: str) -> bool:
            try:
                with open(path, 'r'):
                    return True
            except FileNotFoundError:
                return False
    header = not file_exists(results_csv)
    df.to_csv(results_csv, mode='a', header=header, index=False)


def aggregate_results(results_csv: str, summary_csv: str, group_cols: List[str]) -> None:
    df = pd.read_csv(results_csv)
    metrics = [
        'bal_acc', 'f1', 'auc', 'brier', 'kl', 'ecl',
        'train_time_s', 'param_time_s', 'total_train_time_s', 'infer_time_s'
    ]
    agg = df.groupby(group_cols)[metrics].agg(['mean', 'std']).reset_index()
    agg.to_csv(summary_csv, index=False)
