"""
================================
Recognizing hand-written digits
===============================

"""
#Part1: library dependencies: sklearn, torch, tensorflow, numpy, transformers
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# 1. set the ranges of hyper parameters 
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10] 
h_param_comb = [{'gamma':g, 'C':c} for g in gamma_list for c in c_list]

assert len(h_param_comb) == len(gamma_list)*len(c_list)
train_frac=0.8
dev_frac=0.1
test_frac=0.1
#Part 2 : load dataset 
digits = datasets.load_digits()
#Part 3 : Sanity check visualization of the data

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)
#Part 4: data preprocessing-to remove some noise, to normalize data, format the data to be consumed by mode 
# flatten the images
n_samples = len(digits.images)
#print(digits.images.shape)
resize_images=[]
for i in range (n_samples):
    resize_images.append(resize(digits.images[i], (32,32), anti_aliasing=True))
digits.images = np.array(resize_images)
print(digits.images.shape)

data = digits.images.reshape((n_samples, -1))
# Split data into 80% train and 10% test subsets and 10% validation
#PART 5: define train/test/dev splits of experimental protocol
dev_test_frac = 1-train_frac
X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size=dev_test_frac, shuffle=True
)

X_test, X_dev, y_test, y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=(dev_frac/dev_test_frac), shuffle=True
)

#train to train model
#dev to set hyperparameter to the model
# test to evaluate the performance of the model
# split: https://stackoverflow.com/questions/68060275/how-do-i-best-make-80-train-10-validation-and-10-percent-test-splits-using
#if testing on the same as training set : the performance metrics may overestimate the goodnesss of the model

# Part 6: Define the model 
search_hyperparams = []
accuracy = []
# Create a classifier: a support vector classifier
for cur_h_params in h_param_comb:
    clf = svm.SVC()
# Part 7: Setting up hyperparameter
    hyper_params=cur_h_params
    clf.set_params(**hyper_params)


# Learn the digits on the train subset
#Part 8: Train Model
    clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
#Part 9: Get test prediction
    predicted_dev = clf.predict(X_test)
    result = {
                    'accuracy': metrics.classification_report(y_test, predicted_dev, output_dict=True)['accuracy']
                    
                }
    print(result)
    search_hyperparams.append(
                {
                    "params": cur_h_params,
                    "training_accuracy": metrics.classification_report(y_train, clf.predict(X_train), output_dict=True)['accuracy'],
                    "testing_accuracy": metrics.classification_report(y_test, clf.predict(X_test), output_dict=True)['accuracy'],
                    "dev_accuracy": metrics.classification_report(y_dev, clf.predict(X_dev), output_dict=True)['accuracy']
                }
    )
    print(search_hyperparams)
    accuracy.append(
                result
            )

    print(accuracy)
###############################################################################
print(pd.DataFrame(search_hyperparams))
print("Best hyperparameters were:")
print(cur_h_params)
clf = svm.SVC()

hyper_params = cur_h_params
clf.set_params(**hyper_params)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)
