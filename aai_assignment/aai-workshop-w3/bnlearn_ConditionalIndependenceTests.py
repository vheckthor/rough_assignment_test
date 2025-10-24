import bnlearn as bn
import pandas as pd
from pgmpy.estimators import CITests

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 


def save_structure(edges, title, file_name):
    G = nx.DiGraph()
    G.add_edges_from(edges)
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color='lightblue', font_size=10, arrows=True)
    plt.title(title)
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    print("Look at %s ... to understand the %s!!!\n" % (file_name, title))


if __name__ == "__main__":
    # definition of directed acyclic graphs (predefined structures)
    edges_lungcancer1 = [('Lung_cancer', 'Smoking'), ('Lung_cancer', 'Yellow_Fingers'), ('Lung_cancer', 'Anxiety'), ('Lung_cancer', 'Peer_Pressure'), ('Lung_cancer', 'Genetics'), ('Lung_cancer', 'Attention_Disorder'), ('Lung_cancer', 'Born_an_Even_Day'), ('Lung_cancer', 'Car_Accident'), ('Lung_cancer', 'Fatigue'), ('Lung_cancer', 'Allergy'), ('Lung_cancer', 'Coughing')]
    edges_lungcancer2 = [] # up to you

    # examples of training data include 'data\lung_cancer-train.csv' or  'data\lang_detect_train.csv', etc.
    # examples of net structure (as below): edges_langdet1, edges_langdet2, edges_lungcancer1, edges_lungcancer2
    # choices of CI test: chi_square, g_sq, log_likelihood, freeman_tuckey, modified_log_likelihood, neyman, cressie_read
    TRAINING_DATA = 'data/lung_cancer-train.csv'
    NETWORK_STRUCTURE = edges_lungcancer1
    CONDITIONAL_INDEPENDENCE_TEST = 'chi_square'
    VISUALISE_STRUCTURE=True
    ALPHA=0.05

    # data loading using pandas
    data = pd.read_csv(TRAINING_DATA, encoding='latin')
    print("DATA:\n", data)

    # creation of the directed acyclic graph (DAG)
    DAG = bn.make_DAG(NETWORK_STRUCTURE)
    print("DAG:\n", DAG)

    if VISUALISE_STRUCTURE: # saved DAG to a file
        title = "Predefined Structure %s" % (NETWORK_STRUCTURE)
        save_structure(NETWORK_STRUCTURE, title, "structures/lung_cancer-DAG-structure1.png")

    # parameter learning using Maximum Likelihood Estimation
    model = bn.parameter_learning.fit(DAG, data, methodtype="maximumlikelihood") 
    print("model=",model)

    # statististical test of independence using bnlearn: constraint-based learning approach
    model = bn.independence_test(model, data, test=CONDITIONAL_INDEPENDENCE_TEST, alpha=ALPHA)
    ci_results = list(model['independence_test']['stat_test'])
    num_edges2remove = ci_results.count(False)
    print(model['independence_test'])
    print("num_edges2remove=%s\n" % (num_edges2remove))

    # call to pc-stable, which generates a skeleton (undirected graph) and a DAG as well
    print("Executing PC-Stable:")
    model_pc_stable = bn.structure_learning.fit(data, methodtype='pc', params_pc={'alpha': ALPHA, 'ci_test': CONDITIONAL_INDEPENDENCE_TEST})
    for key, values in model_pc_stable.items():
        print("key=%s values=%s" % (key, values))
    save_structure(model_pc_stable["dag_edges"], "PC-Stable Structure", "structures/lung_cancer-DAG-pc-stable.png")