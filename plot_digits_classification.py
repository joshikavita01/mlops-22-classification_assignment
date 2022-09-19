"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""
#Part1: library dependencies: sklearn, torch, tensorflow, numpy, transformers
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
GAMMA=0.001
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
data = digits.images.reshape((n_samples, -1))
# Split data into 80% train and 10% test subsets and 10% validation
#PART 5: define train/test/dev splits of experimental protocol
X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size=1-train_frac, shuffle=False
)

X_test, X_dev, y_test, y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=((dev_frac)/(test_frac+dev_frac)), shuffle=False
)

#train to train model
#dev to set hyperparameter to the model
# test to evaluate the performance of the model
# split: https://stackoverflow.com/questions/68060275/how-do-i-best-make-80-train-10-validation-and-10-percent-test-splits-using
#if testing on the same as training set : the performance metrics may overestimate the goodnesss of the model

# Part 6: Define the model 

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)
# Part 7: Setting up hyperparameter
hyper_params={'gamma':GAMMA}
clf.set_params(**hyper_params)


# Learn the digits on the train subset
#Part 8: Train Model
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
#Part 9: Get test prediction
predicted = clf.predict(X_test)

###############################################################################
# Below we visualize the first 4 test samples and show their predicted
# digit value in the title.
#Part 10: Sanity check of predictions
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")
#Part 11: Compute evaluation merics
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()
