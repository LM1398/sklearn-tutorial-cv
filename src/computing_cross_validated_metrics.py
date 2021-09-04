# 3.1 Importing modules
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm


# Simple train_test split and fitting on svm

X, y = datasets.load_iris(return_X_y=True)
X.shape, y.shape
((150, 4), (150,))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

X_train.shape, y_train.shape
((90, 4), (90,))
X_test.shape, y_test.shape
((60, 4), (60,))

clf = svm.SVC(kernel="linear", C=1).fit(X_train, y_train)
clf.score(X_test, y_test)

# 3.1.1. Computing cross-validated metrics¶

# Cross validation using cross_val_score; the train datasets are divided into 5 different groups (cv=5) and cross validted
# to check the scores of each set of data

from sklearn.model_selection import cross_val_score

clf = svm.SVC(kernel="linear", C=1, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)
scores

# To get the score as well as the std the following code can be used

print(
    "%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std())
)

# It's possible to change the scoring (e.g. accuracy, precision, f1_macro) with the codes below

from sklearn import metrics

scores = cross_val_score(clf, X, y, cv=5, scoring="f1_macro")
scores

# It's also possible to add a cv iterator instead of giving an integer and specify the cv method

from sklearn.model_selection import ShuffleSplit

n_samples = X.shape[0]
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
cross_val_score(clf, X, y, cv=cv)

# Can also use an iterable yielding splits as an array of indices


def custom_cv_2folds(X):
    n = X.shape[0]
    i = 1
    while i <= 2:
        idx = np.arange(n * (i - 1) / 2, n * i / 2, dtype=int)
        yield idx, idx
        i += 1


custom_cv = custom_cv_2folds(X)
cross_val_score(clf, X, y, cv=custom_cv)

# Preprocessing should always be done on the train set to have better predictions and these processes should also
# be done on the test set aswell

from sklearn import preprocessing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
clf = svm.SVC(C=1).fit(X_train_transformed, y_train)
X_test_transformed = scaler.transform(X_test)
clf.score(X_test_transformed, y_test)

# Using a pipeline will allow the data to be preprocessed and fit into an estimator through one function

from sklearn.pipeline import make_pipeline

clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
cross_val_score(clf, X, y, cv=cv)

# 3.1.1.1. The cross_validate function and multiple metric evaluation¶

# cross_validate allows user to test multiple scoring metrics at once for cross validation (e.g. precision, accuracy)
# It can also return the fit-time, score-time, the training scores as well as the fitted estimator
# These data are all returned as a dictinonary (e.g. ['estimator', 'fit_time', 'score_time'])

from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score

scoring = ["precision_macro", "recall_macro"]
clf = svm.SVC(kernel="linear", C=1, random_state=0)
scores = cross_validate(clf, X, y, scoring=scoring)
sorted(scores.keys())
scores["test_recall_macro"]

# Can also make a dictionary specifying the scoring with more detail

from sklearn.metrics import make_scorer

scoring = {
    "prec_macro": "precision_macro",
    "rec_macro": make_scorer(recall_score, average="macro"),
}
scores = cross_validate(clf, X, y, scoring=scoring, cv=5, return_train_score=True)
sorted(scores.keys())
scores["train_rec_macro"]

# Or it can also be used with a single scoring metric

scores = cross_validate(
    clf, X, y, scoring="precision_macro", cv=5, return_estimator=True
)
sorted(scores.keys())
