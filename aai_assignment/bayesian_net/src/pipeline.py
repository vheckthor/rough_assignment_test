from typing import Dict, Any, Tuple, List
import time
import os
import numpy as np
import pandas as pd

from bayesian_net.src.structure.structure_algorithm import learn_structure_hc_bic, learn_structure_pc_stable
from bayesian_net.src.cpt import estimate_cpts, BNModel
from bayesian_net.src.evaluator import evaluate_model


class BNPipeline:
    def __init__(self, n_splits: int = 5, seed: int = 42, bins: int = 5):
        self.n_splits = n_splits
        self.seed = seed
        self.bins = bins

    def detect_continuous_columns(self, df: pd.DataFrame, exclude: List[str]) -> List[str]:
        feature_cols = [c for c in df.columns if c not in exclude]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric_cols if df[c].nunique(dropna=True) > 2]

    def bootstrap_qcut_discretize(self, df_train: pd.DataFrame, df_test: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        feature_cols: List[str] = [c for c in df_train.columns if c != target_col]
        continuous_cols = self.detect_continuous_columns(df_train, exclude=[target_col])
        df_train_out = df_train.copy()
        df_test_out = df_test.copy()
        for col in continuous_cols:
            try:
                codes, bins = pd.qcut(df_train_out[col], q=self.bins, labels=False, duplicates='drop', retbins=True)
                df_train_out[col] = codes.astype('Int64').astype(str)
                df_test_out[col] = pd.cut(df_test_out[col], bins=bins, include_lowest=True, labels=False).astype('Int64').astype(str)
            except Exception:
                df_train_out[col] = df_train_out[col].astype(str)
                df_test_out[col] = df_test_out[col].astype(str)
        # cast remaining to string
        for col in feature_cols:
            if col not in continuous_cols:
                df_train_out[col] = df_train_out[col].astype(str)
                df_test_out[col] = df_test_out[col].astype(str)
        df_train_out[target_col] = df_train_out[target_col].astype(str)
        df_test_out[target_col] = df_test_out[target_col].astype(str)
        return df_train_out, df_test_out

    def bnlearn_discretize_with_edges(self, df_train: pd.DataFrame, df_test: pd.DataFrame, target_col: str, edges: List[Tuple[str, str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        import bnlearn as bn
        continuous_cols = self.detect_continuous_columns(df_train, exclude=[target_col])
        # Combine to ensure consistent discretization, then split back
        combined = pd.concat([df_train.drop(columns=[target_col]), df_test.drop(columns=[target_col])], axis=0, ignore_index=True)
        combined_disc = bn.discretize(data=combined, edges=edges, continuous_columns=continuous_cols, max_iterations=8, verbose=0)
        # Reattach targets
        train_len = len(df_train)
        Xtr_disc = combined_disc.iloc[:train_len, :].astype(str)
        Xte_disc = combined_disc.iloc[train_len:, :].astype(str)
        df_train_out = pd.concat([Xtr_disc, df_train[[target_col]].astype(str)], axis=1)
        df_test_out = pd.concat([Xte_disc, df_test[[target_col]].astype(str)], axis=1)
        return df_train_out, df_test_out

    def learn_structure(self, df_train_discrete: pd.DataFrame, method: str = 'hillclimb', save_png: str | None = None) -> Dict[str, Any]:
        if method == 'hillclimb':
            result = learn_structure_hc_bic(df_train_discrete)
            if save_png and result.get('edges'):
                learn_structure_hc_bic(df_train_discrete, save_png=save_png)
        elif method == 'pc_stable':
            result = learn_structure_pc_stable(df_train_discrete)
            if save_png and result.get('edges'):
                learn_structure_pc_stable(df_train_discrete, save_png=save_png)
        else:
            raise ValueError(f"Unknown structure learning method: {method}")
        return result

    def learn_parameters(self, df_train_discrete: pd.DataFrame, edges: List[tuple]) -> BNModel:
        start = time.perf_counter()
        model = estimate_cpts(df_train_discrete, edges, laplace=1.0)
        elapsed = time.perf_counter() - start
        return model, elapsed

    def evaluate(self, model: BNModel, df_test_discrete: pd.DataFrame, target_col: str) -> Tuple[Dict[str, float], float]:
        test_rows = df_test_discrete.astype(str).to_dict(orient='records')
        metrics, infer_time = evaluate_model(model, test_rows, target_col)
        return metrics, infer_time
