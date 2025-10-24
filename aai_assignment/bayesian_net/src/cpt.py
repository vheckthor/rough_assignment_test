"""
CPT estimation utilities for Bayesian Networks.

Reference: Inspired by ideas from aai-workshop-w3 (CPT_Generator.py) but reimplemented here.
"""
from typing import Dict, List, Tuple
import itertools
import pandas as pd


class BNModel:
    def __init__(self, random_variables: List[str], edges: List[Tuple[str, str]]):
        self.random_variables = list(random_variables)
        self.edges = list(edges)  # list of (parent, child)
        self.structure = self._build_structure_strings()
        self.rv_values: Dict[str, List[str]] = {}
        self.cpts: Dict[str, Dict[str, float]] = {}

    def _build_structure_strings(self) -> List[str]:
        parents_of: Dict[str, List[str]] = {v: [] for v in self.random_variables}
        for parent, child in self.edges:
            parents_of[child].append(parent)
        structure = []
        for v in self.random_variables:
            parents = parents_of[v]
            if len(parents) == 0:
                structure.append(f"P({v})")
            else:
                parents_sorted = ",".join(sorted(parents))
                structure.append(f"P({v}|{parents_sorted})")
        return structure

    def write_config(self, path: str) -> None:
        with open(path, 'w') as f:
            f.write(f"name:bn-model\n\n")
            f.write("random_variables:" + ";".join(self.random_variables) + "\n\n")
            f.write("structure:" + ";".join(self.structure) + "\n\n")
            for key, table in self.cpts.items():
                f.write(f"CPT{key[1:]}:\n")  # key already like P(X|...)
                items = list(table.items())
                for i, (domain_vals, prob) in enumerate(items):
                    sep = ";" if i < len(items) - 1 else ""
                    f.write(f"{domain_vals}={prob}{sep}\n")
                f.write("\n")


def _unique_values(df: pd.DataFrame, columns: List[str]) -> Dict[str, List[str]]:
    values: Dict[str, List[str]] = {}
    for col in columns:
        vals = df[col].astype(str).unique().tolist()
        values[col] = vals
    return values


def _parents_map(edges: List[Tuple[str, str]], variables: List[str]) -> Dict[str, List[str]]:
    parents_of: Dict[str, List[str]] = {v: [] for v in variables}
    for p, c in edges:
        parents_of[c].append(p)
    return parents_of


def estimate_cpts(
    df_train: pd.DataFrame,
    edges: List[Tuple[str, str]],
    laplace: float = 1.0,
) -> BNModel:
    """Estimate CPTs with Laplace smoothing from training data and edges.

    Returns BNModel with structure, rv_values, and CPTs populated.
    """
    variables = list(df_train.columns)
    model = BNModel(random_variables=variables, edges=edges)
    model.rv_values = _unique_values(df_train, variables)
    parents_of = _parents_map(edges, variables)

    # Prior probabilities (no parents)
    for v in variables:
        parents = parents_of[v]
        if len(parents) == 0:
            counts: Dict[str, int] = {val: 0 for val in model.rv_values[v]}
            for val in df_train[v].astype(str):
                counts[val] += 1
            total = sum(counts.values())
            J = len(counts)
            key = f"P({v})"
            cpt: Dict[str, float] = {}
            for val, cnt in counts.items():
                cpt[val] = (cnt + laplace) / (total + J * laplace)
            model.cpts[key] = cpt

    # Conditional probabilities (with parents)
    for v in variables:
        parents = sorted(parents_of[v])
        if len(parents) == 0:
            continue
        parent_values_lists = [model.rv_values[p] for p in parents]
        parent_assignments = list(itertools.product(*parent_values_lists))
        key = f"P({v}|{','.join(parents)})"
        cpt: Dict[str, float] = {}
        J = len(model.rv_values[v])
        for assign in parent_assignments:
            # filter rows matching this parent assignment
            mask = None
            for p, val in zip(parents, assign):
                m = df_train[p].astype(str) == str(val)
                mask = m if mask is None else (mask & m)
            subset = df_train[mask]
            denom = len(subset)
            # compute counts of child values
            for child_val in model.rv_values[v]:
                num = int((subset[v].astype(str) == str(child_val)).sum())
                # Laplace smoothing
                prob = (num + laplace) / (denom + J * laplace) if denom is not None else 1.0 / J
                cpt[f"{child_val}|{','.join(map(str, assign))}"] = prob
        model.cpts[key] = cpt

    return model
