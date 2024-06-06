from architecture import *
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn import svm
import numpy as np

algo_dico = {}
algo_dico["model"] = svm.SVC()
algo_dico["param"] = {}
"""algo_dico["param"] = {"n_neighbors": 10}"""
"""algo_dico["param"] = {"n_estimators": 50,
                        "criterion": "gini",
                        "max_depth": 10}"""

data = load_transform_label_train_data("Data/", "HC")
test = load_transform_test_data("test_data/", "HC")

print("done")

train_data = {}
test_data = {}

train_data["X"] = np.asarray([data[filename][0] for filename in data.keys()])
train_data["y"] = np.asarray([data[filename][1] for filename in data.keys()])

model = learn_model_from_data(train_data, algo_dico)

write_predictions("result/", "the_data_wizards.txt", test, model)

scores = estimate_model_score(train_data, model, 5)
print(scores)