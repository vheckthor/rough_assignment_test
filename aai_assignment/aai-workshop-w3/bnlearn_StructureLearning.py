import bnlearn as bn
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from bnlearn_ConditionalIndependenceTests import save_structure

import matplotlib
matplotlib.use('Agg') 

# examples of training data include 'data\lung_cancer-train.csv' or  'data\lang_detect_train.csv', etc.
# choices of scoring functions: bic, k2, bdeu, bds, aic
TRAINING_DATA = 'data/lung_cancer-train.csv'
METHOD_TYPE='hillclimbsearch'
SCORING_FUNCTION = 'bic' 
MAX_ITERATIONS=20000000
VISUALISE_STRUCTURE=True

# data loading using pandas
data = pd.read_csv(TRAINING_DATA, encoding='latin')
print("DATA:\n", data)

# structure learning using a chosen scoring function (such as 'bic' or 'aic' or 'bdeu')
model = bn.structure_learning.fit(data, methodtype=METHOD_TYPE, scoretype=SCORING_FUNCTION, max_iter=MAX_ITERATIONS)
print("model [%s]=%s" % (METHOD_TYPE, model))
print("num_model_edges [%s]=%s" % (METHOD_TYPE, len(model['model_edges'])))

# visualise the learnt structure
if VISUALISE_STRUCTURE:
    title = "Learnt Structure %s" % (METHOD_TYPE)
    save_structure(model['model_edges'], title, "structures/lung_cancer-DAG-hc-%s.png" % (SCORING_FUNCTION))


# The above is for discrete data. For continuous data, (a) discretise your data, 
# (b) learn a structure, then (c) use it with continuous random variables. 
# More on the later next week.
# For data discretisation, see https://erdogant.github.io/bnlearn/pages/html/Discretizing.html