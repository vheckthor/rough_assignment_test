"""
Exact inference by enumeration for Bayesian Networks.

Reference: Informed by aai-workshop-w3 (BayesNetInference.py) enumeration approach; reimplemented here.
"""
from typing import Dict, List

from bayesian_net.src.cpt import BNModel


def _get_parents(var: str, model: BNModel) -> List[str]:
    return sorted([p for p, c in model.edges if c == var])


def _get_domain_values(var: str, model: BNModel) -> List[str]:
    return model.rv_values[var]


def _prob_given_parents(var: str, value: str, evidence: Dict[str, str], model: BNModel) -> float:
    parents = _get_parents(var, model)
    if len(parents) == 0:
        key = f"P({var})"
        return float(model.cpts[key][value])
    else:
        parent_vals = ",".join([str(evidence[p]) for p in parents])
        key = f"P({var}|{','.join(parents)})"
        return float(model.cpts[key][f"{value}|{parent_vals}"])


def _enumerate_all(variables: List[str], evidence: Dict[str, str], model: BNModel) -> float:
    if len(variables) == 0:
        return 1.0
    v = variables[0]
    rest = variables[1:]
    if v in evidence:
        p = _prob_given_parents(v, str(evidence[v]), evidence, model)
        return p * _enumerate_all(rest, evidence, model)
    else:
        total = 0.0
        for val in _get_domain_values(v, model):
            evidence[v] = val
            p = _prob_given_parents(v, val, evidence, model)
            total += p * _enumerate_all(rest, evidence, model)
            evidence.pop(v, None)
        return total


def predict_distribution(model: BNModel, evidence: Dict[str, str], query_var: str) -> Dict[str, float]:
    Q: Dict[str, float] = {}
    for val in _get_domain_values(query_var, model):
        ev = dict(evidence)
        ev[query_var] = val
        Q[val] = _enumerate_all(list(model.random_variables), ev, model)
    # normalize
    s = sum(Q.values())
    if s > 0:
        for k in Q:
            Q[k] = Q[k] / s
    return Q
