from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold


def create_folds(
    df: pd.DataFrame,
    target_col: str,
    n_splits: int = 5,
    seed: int = 42,
    stratify: bool = True,
) -> List[Tuple[pd.Index, pd.Index]]:
    """Return list of (train_idx, test_idx) index tuples.

    If stratify is True, uses StratifiedKFold on target_col; else KFold.
    """
    if stratify:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        y = df[target_col]
        splits = list(splitter.split(df.index, y))
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(df.index))

    return [(pd.Index(train_idx), pd.Index(test_idx)) for train_idx, test_idx in splits]
