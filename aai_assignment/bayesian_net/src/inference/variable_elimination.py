from typing import Dict

from bayesian_net.src.cpt import BNModel


def query_variable_elimination(model: BNModel, evidence: Dict[str, str], query_var: str) -> Dict[str, float]:
    """Placeholder for exact inference by Variable Elimination.
    To be implemented with factors, elimination ordering (min-fill), and factor ops.
    """
    raise NotImplementedError("Variable Elimination not yet implemented.")

