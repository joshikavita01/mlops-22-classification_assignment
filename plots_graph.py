# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics, tree
import pdb
import pandas as pd

from utils import (
    preprocess_digits,
    train_dev_test_split,
    data_viz,
    get_all_h_param_comb,
    tune_and_save,
    macro_f1
)
from joblib import dump, load

train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
assert train_frac + dev_frac + test_frac == 1.0

# 1. set the ranges of hyper parameters
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

svm_params = {}
svm_params["gamma"] = gamma_list
svm_params["C"] = c_list
svm_h_param_comb = get_all_h_param_comb(svm_params)

max_depth_list = [2, 10, 20, 50, 100]

dec_params = {}
dec_params["max_depth"] = max_depth_list
dec_h_param_comb = get_all_h_param_comb(dec_params)

h_param_comb = {"svm": svm_h_param_comb, "decision_tree": dec_h_param_comb}

# PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)
# housekeeping
del digits

# define the evaluation metric
#metric_list = [metrics.accuracy_score, macro_f1]
#h_metric = metrics.accuracy_score
metric=metrics.accuracy_score
number_cv = 5
final_result = {}
for n in range(number_cv):
    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
        data, label, train_frac, dev_frac
    )
    # PART: Define the model
    # Create a classifier: a support vector classifier
    models_of_choice = {
        "svm": svm.SVC(),
        "decision_tree": tree.DecisionTreeClassifier(),
    }
    for classifier_name in models_of_choice:
        clf = models_of_choice[classifier_name]
        print("[{}] Running hyper param tuning for {}".format(n,classifier_name))
        actual_model_path = tune_and_save(
            clf, x_train, y_train, x_dev, y_dev, metric, h_param_comb[classifier_name], model_path=None
        )

        # 2. load the best_model
        best_model = load(actual_model_path)

        # PART: Get test set predictions
        # Predict the value of the digit on the test subset
        predicted = best_model.predict(x_test)
        if not classifier_name in final_result:
            final_result[classifier_name]=[]    

        final_result[classifier_name].append(metric(y_pred=predicted, y_true=y_test) )
        # 4. report the test set accurancy with that best model.
        # PART: Compute evaluation metrics
        print(
            f"Classification report for classifier {clf}:\n"
            f"{metrics.classification_report(y_test, predicted)}\n"
        )

print(final_result)
df=pd.DataFrame.from_dict(final_result)
df.index = df.index + 1
df.loc['mean'] = df.mean()
df.loc['std'] = df.std()
df.index.name='run'
print(df)