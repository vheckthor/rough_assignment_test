import BayesNetUtil as bnu
from CPT_Generator import CPT_Generator
from BayesNetInference import BayesNetInference
from ModelEvaluator import ModelEvaluator

print("\n##############################")
print("# Step 1. Structure learning #")
print("##############################")
print("Nothing here yet...up to you")

print("\n##############################")
print("# Step 2. Parameter learning #")
print("##############################")
configfile_name = "config/config-lungcancer-structure1.txt"
datafile_train = "data/lung_cancer-train.csv"
cpt_gen = CPT_Generator(configfile_name, datafile_train)

print("\n###################################")
print("# Step 3. Probabilistic inference #")
print("###################################")
algorithm_name = "InferenceByEnumeration"
prob_queries = ["P(Lung_cancer|Smoking=1,Coughing=1)", "P(Smoking|Lung_cancer=1,Coughing=1)"]
prob_query = None # we run a set of queries below instead of a single one
num_samples = None # this is not needed for exact inference
bni = BayesNetInference(algorithm_name, configfile_name, prob_query, num_samples)
for prob_query in prob_queries:
    bni.query = bnu.tokenise_query(prob_query, True)
    prob_dist = bni.enumeration_ask()
    normalised_dist = bnu.normalise(prob_dist)
    print("unnormalised P(%s)=%s" % (bni.query["query_var"], prob_dist))
    print("normalised P(%s)=%s" % (bni.query["query_var"], normalised_dist))

print("\n##############################")
print("# Step 4. MOdel evaluation   #")
print("##############################")
datafile_test = "data/lung_cancer-test.csv"
ModelEvaluator(configfile_name, datafile_test)