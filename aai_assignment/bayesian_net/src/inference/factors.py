from typing import Dict, List
from itertools import product


class Factor:
    def __init__(self, variables: List[str], table: Dict[tuple, float]):
        self.variables = list(variables)
        self.table = dict(table)  # keys are tuples aligned with variables

    def normalize(self) -> 'Factor':
        s = sum(self.table.values())
        if s == 0:
            return self
        self.table = {k: v / s for k, v in self.table.items()}
        return self

    def multiply(self, other: 'Factor') -> 'Factor':
        vars_union = list(dict.fromkeys(self.variables + other.variables))
        idx_self = [vars_union.index(v) for v in self.variables]
        idx_other = [vars_union.index(v) for v in other.variables]
        new_table: Dict[tuple, float] = {}
        domains = []  # domain sizes to be provided by caller in future
        # Placeholder: assume consistent assignments already provided externally
        for assignment_s, val_s in self.table.items():
            for assignment_o, val_o in other.table.items():
                # naive merge if consistent overlap
                consistent = True
                for v in set(self.variables) & set(other.variables):
                    if assignment_s[self.variables.index(v)] != assignment_o[other.variables.index(v)]:
                        consistent = False
                        break
                if not consistent:
                    continue
                # build union assignment
                union = [None] * len(vars_union)
                for i, v in enumerate(self.variables):
                    union[vars_union.index(v)] = assignment_s[i]
                for i, v in enumerate(other.variables):
                    union[vars_union.index(v)] = assignment_o[i]
                new_table[tuple(union)] = new_table.get(tuple(union), 0.0) + val_s * val_o
        return Factor(vars_union, new_table)

    def sum_out(self, var: str) -> 'Factor':
        if var not in self.variables:
            return self
        idx = self.variables.index(var)
        new_vars = self.variables[:idx] + self.variables[idx+1:]
        new_table: Dict[tuple, float] = {}
        for assignment, value in self.table.items():
            reduced = assignment[:idx] + assignment[idx+1:]
            new_table[reduced] = new_table.get(reduced, 0.0) + value
        return Factor(new_vars, new_table)

