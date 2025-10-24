import bnlearn as bn
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from bnlearn_ConditionalIndependenceTests import save_structure

import matplotlib
matplotlib.use('Agg') 

# examples of training data include 'data\lung_cancer-train.csv' or  'data\lang_detect_train.csv', etc.
# choices of scoring functions: bic, k2, bdeu, bds, aic
TRAINING_DATA = 'data/data_banknote_authentication-train.csv'
METHOD_TYPE='hillclimbsearch'
SCORING_FUNCTION = 'aic'
MAX_ITERATIONS=20000000
VISUALISE_STRUCTURE=True

# data loading using pandas
data = pd.read_csv(TRAINING_DATA, encoding='latin')
print("DATA:\n", data)

# definition of directed acyclic graph (predefined Naive Bayes structure -- only for discretising data)
edges = [('target', 'x1'),('target', 'x2'),('target', 'x3'),('target', 'x4')]

# performs discretisation of continuous data for the columns specified and structure provided
# the output of this steps is later used for training a Bayesian network -- no longer the original dataset
continuous_columns = ["x1", "x2", "x3", "x4"]
discrete_data = bn.discretize(data, edges, continuous_columns, max_iterations=1, verbose=3)
for randvar in discrete_data:
    print("VARIABLE:",randvar)
    print(discrete_data[randvar])

# structure learning using a chosen scoring function (such as 'bic' or 'aic')
model = bn.structure_learning.fit(discrete_data, methodtype=METHOD_TYPE, scoretype=SCORING_FUNCTION, max_iter=MAX_ITERATIONS)
num_model_edges = len(model['model_edges'])
print("model=",model)
print("num_model_edges="+str(num_model_edges))

# visualise the learnt structure
if VISUALISE_STRUCTURE:
    title = "Learnt Structure %s" % (METHOD_TYPE)
    save_structure(model['model_edges'], title, "structures/bankdata-DAG-hc-%s.png" % (SCORING_FUNCTION))
