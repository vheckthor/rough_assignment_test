import bnlearn as bn
import matplotlib
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional


def _save_structure_png(edges: List[Tuple[str, str]], title: str, file_path: str) -> bool:
    try:
        matplotlib.use('Agg')
        G = nx.DiGraph()
        G.add_edges_from(edges)
        pos = nx.spring_layout(G)
        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, node_size=1000, node_color='lightblue', font_size=10, arrows=True)
        plt.title(title)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        return True
    except Exception:
        return False


def learn_structure_hc_bic(df, max_iter: int = 1_000_000, save_png: Optional[str] = None) -> Dict[str, Any]:
    """Learn structure using bnlearn hillclimb + BIC. Returns dict with edges and optional model.
    If save_png is provided, attempts to save a DAG image to that path using networkx/matplotlib if available.
    """
    try:
        model = bn.structure_learning.fit(df, methodtype='hillclimbsearch', scoretype='bic', max_iter=max_iter)
        edges: List[Tuple[str, str]] = list(model.get('model_edges', []))
        print(f"Learnt Structure HC+BIC: {edges}")
        if save_png is not None:
            _save_structure_png(edges, "Learnt Structure HC+BIC", save_png)
        return {"edges": edges, "model": model}
    except Exception as e:
        # bnlearn not available or failed; return empty result with error
        return {"edges": [], "model": None, "error": str(e)}


def learn_structure_pc_stable(df, alpha: float = 0.05, ci_test: str = 'chi_square', save_png: Optional[str] = None, max_iter: int = 1_000_000) -> Dict[str, Any]:
    """Learn structure using bnlearn PC-Stable (constraint-based). Returns dict with edges and optional model.
    If save_png is provided, attempts to save a DAG image to that path using networkx/matplotlib if available.
    """
    try:
        model = bn.structure_learning.fit(df, methodtype='pc', params_pc={'alpha': alpha, 'ci_test': ci_test}, max_iter=max_iter)
        edges: List[Tuple[str, str]] = list(model.get('dag_edges', []))
        print(f"Learnt Structure PC-Stable: {edges}")
        if save_png is not None:
            _save_structure_png(edges, "Learnt Structure PC-Stable", save_png)
        return {"edges": edges, "model": model}
    except Exception as e:
        # bnlearn not available or failed; return empty result with error
        return {"edges": [], "model": None, "error": str(e)}
