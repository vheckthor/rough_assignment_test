from typing import Optional
import os
import time
import pandas as pd

from bayesian_net.src.kfold import create_folds
from bayesian_net.src.pipeline import BNPipeline
from bayesian_net.src.reporting import append_fold_result, aggregate_results


def run_experiment(csv_path: str, target_col: Optional[str] = None, n_splits: int = 5, seed: int = 42, bins: int = 5):
    df = pd.read_csv(csv_path)
    if target_col is None:
        target_col = df.columns[-1]

    results_dir = os.path.join(os.path.dirname(csv_path), 'reports')
    os.makedirs(results_dir, exist_ok=True)
    per_fold_csv = os.path.join(results_dir, f"bn_results_{os.path.basename(csv_path)}.csv")
    summary_csv = os.path.join(results_dir, f"bn_summary_{os.path.basename(csv_path)}.csv")

    folds = create_folds(df, target_col=target_col, n_splits=n_splits, seed=seed, stratify=True)
    pipeline = BNPipeline(n_splits=n_splits, seed=seed, bins=bins)

    # Run both structure learning methods
    methods = ['hillclimb', 'pc_stable']
    
    for method in methods:
        print(f"Running {method} structure learning...")
        
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            df_train = df.loc[train_idx].copy()
            df_test = df.loc[test_idx].copy()

            # Clean: drop missing rows
            df_train = df_train.dropna()
            df_test = df_test.dropna()

            # Initial discretization (bootstrap qcut)
            df_train_disc, df_test_disc = pipeline.bootstrap_qcut_discretize(df_train, df_test, target_col)

            # Structure learning on discretized data
            png_path = os.path.join(results_dir, f"dag_{method}_fold{fold_idx+1}.png")
            start = time.perf_counter()
            struct = pipeline.learn_structure(df_train_disc, method=method, save_png=png_path)
            struct_time = time.perf_counter() - start
            edges = struct.get('edges', [])

            if edges:
                try:
                    df_train_disc, df_test_disc = pipeline.bnlearn_discretize_with_edges(df_train, df_test, target_col, edges)
                except Exception as e:
                    print(f"Warning: bnlearn re-discretization failed for {method} fold {fold_idx+1}: {e}")
                    # Keep bootstrap discretization

            # Parameter learning (CPTs)
            model, param_time = pipeline.learn_parameters(df_train_disc, edges)
            total_train_time = struct_time + param_time

            # Evaluate
            metrics, infer_time = pipeline.evaluate(model, df_test_disc, target_col)

            # Determine method-specific parameters
            if method == 'hillclimb':
                score = 'bic'
                max_iter = 2_000_000
            else:  # pc_stable
                score = 'chi_square'
                max_iter = 0  # PC doesn't use iterations

            row = {
                'dataset': os.path.basename(csv_path),
                'fold': fold_idx + 1,
                'n_train': len(df_train_disc),
                'n_test': len(df_test_disc),
                'discretizer': 'bootstrap+qcut+bnlearn' if edges else 'bootstrap+qcut',
                'bins': bins,
                'method': method,
                'score': score,
                'max_iter': max_iter,
                'num_edges': len(edges),
                'train_time_s': round(struct_time, 6),
                'param_time_s': round(param_time, 6),
                'total_train_time_s': round(total_train_time, 6),
                'infer_time_s': round(infer_time, 6),
                **metrics,
            }
            append_fold_result(per_fold_csv, row)

    # Aggregate results
    aggregate_results(per_fold_csv, summary_csv, group_cols=['dataset', 'method'])
    print(f"Wrote per-fold results to {per_fold_csv}")
    print(f"Wrote summary to {summary_csv}")


if __name__ == "__main__":
    pass
