# 3.1.2. Cross validation iterators¶

# 3.1.2.1. Cross-validation iterators for i.i.d. (independent and identically distributed) data

# 3.1.2.1.1. K-fold

# Kfold divides the dataset into k groups of samples, called folds
# Example of a 2-fold cross validation split

import numpy as np
from sklearn.model_selection import KFold

X = ["a", "b", "c", "d"]
kf = KFold(n_splits=2)
for train, test in kf.split(X):
    print("%s %s" % (train, test))

# Train test splits can also be done manually

X = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0], [2.0, 2.0]])
y = np.array([0, 1, 0, 1])
X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

# 3.1.2.1.2. Repeated K-Fold¶

# A repeated K-fold repeats the k-fold multiple times

import numpy as np
from sklearn.model_selection import RepeatedKFold

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
random_state = 12883823
rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=random_state)
for train, test in rkf.split(X):
    print("%s %s" % (train, test))

# 3.1.2.1.3. Leave One Out (LOO)¶

# Cross validation method where it makes learning sets where each time one sample is left out (as the test)
# and the other values as the train set

from sklearn.model_selection import LeaveOneOut

X = [1, 2, 3, 4]
loo = LeaveOneOut()
for train, test in loo.split(X):
    print("%s %s" % (train, test))

# 3.1.2.1.4. Leave P Out (LPO)¶

# LPO is similar to LeaveOneOut but instead removes p amount of data and creates all possible train-test splits
# that are possible without the p datasets

from sklearn.model_selection import LeavePOut

X = np.ones(4)
lpo = LeavePOut(p=2)
for train, test in lpo.split(X):
    print("%s %s" % (train, test))

# 3.1.2.1.5. Random permutations cross-validation a.k.a. Shuffle & Split¶

# Shuffle split will shuffle the whole data set and split it into train and test datasets
# The number of train-test splits can be chosen using n_splits

from sklearn.model_selection import ShuffleSplit

X = np.arange(10)
ss = ShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
for train_index, test_index in ss.split(X):
    print("%s %s" % (train_index, test_index))

# 3.1.2.2.1. Stratified k-fold

# StratifiedKFold is a variation of k-fold which returns stratified folds: each set contains
# approximately the same percentage of samples of each target class as the complete set.
# e.g. [1,1,2,2,3,3,4,4,5,5] -> [1,2,3,4,5,] [1,2,3,4,5]

from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np

X, y = np.ones((50, 1)), np.hstack(([0] * 45, [1] * 5))
skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
    print(
        "train -  {}   |   test -  {}".format(
            np.bincount(y[train]), np.bincount(y[test])
        )
    )
kf = KFold(n_splits=3)
for train, test in kf.split(X, y):
    print(
        "train -  {}   |   test -  {}".format(
            np.bincount(y[train]), np.bincount(y[test])
        )
    )

# 3.1.2.2.2. Stratified Shuffle Split¶

# Returns stratified splits, i.e which creates splits by preserving the same percentage for each
# target class as in the complete set

# 3.1.2.3.1. Group k-fold

# GroupKFold is a variation of k-fold which ensures that the same group is not represented in both testing
# and training sets

from sklearn.model_selection import GroupKFold

X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10]
y = ["a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]
groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
gkf = GroupKFold(n_splits=3)
for train, test in gkf.split(X, y, groups=groups):
    print("%s %s" % (train, test))

# 3.1.2.3.2. Leave One Group Out¶

# LeaveOneGroupOut is a cross-validation scheme which holds out the samples according to a third-party
# provided array of integer groups.

from sklearn.model_selection import LeaveOneGroupOut

X = [1, 5, 10, 50, 60, 70, 80]
y = [0, 1, 1, 2, 2, 2, 2]
groups = [1, 1, 2, 2, 3, 3, 3]
logo = LeaveOneGroupOut()
for train, test in logo.split(X, y, groups=groups):
    print("%s %s" % (train, test))

# 3.1.2.3.3. Leave P Groups Out
# LeavePGroupsOut is similar as LeaveOneGroupOut, but removes samples related to  groups for each training/test set

from sklearn.model_selection import LeavePGroupsOut

X = np.arange(6)
y = [1, 1, 1, 2, 2, 2]
groups = [1, 1, 2, 2, 3, 3]
lpgo = LeavePGroupsOut(n_groups=2)
for train, test in lpgo.split(X, y, groups=groups):
    print("%s %s" % (train, test))

# 3.1.2.3.4. Group Shuffle Split

# The GroupShuffleSplit iterator behaves as a combination of ShuffleSplit and LeavePGroupsOut, and generates
# a sequence of randomized partitions in which a subset of groups are held out for each split

from sklearn.model_selection import GroupShuffleSplit

X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 0.001]
y = ["a", "b", "b", "b", "c", "c", "c", "a"]
groups = [1, 1, 2, 2, 3, 3, 4, 4]
gss = GroupShuffleSplit(n_splits=4, test_size=0.5, random_state=0)
for train, test in gss.split(X, y, groups=groups):
    print("%s %s" % (train, test))

# 3.1.2.5. Using cross-validation iterators to split train and test

# Perform train test split using GroupShuffleSplit

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

X = np.array([0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 0.001])
y = np.array(["a", "b", "b", "b", "c", "c", "c", "a"])
groups = np.array([1, 1, 2, 2, 3, 3, 4, 4])
train_indx, test_indx = next(GroupShuffleSplit(random_state=7).split(X, y, groups))
X_train, X_test, y_train, y_test = (
    X[train_indx],
    X[test_indx],
    y[train_indx],
    y[test_indx],
)
X_train.shape, X_test.shape
np.unique(groups[train_indx]), np.unique(groups[test_indx])

# 3.1.2.6.1. Time Series Split

# TimeSeriesSplit is a variation of k-fold which returns first  folds as train set and the (k +1)th fold as test set

from sklearn.model_selection import TimeSeriesSplit

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4, 5, 6])
tscv = TimeSeriesSplit(n_splits=3)
print(tscv)
TimeSeriesSplit(gap=0, max_train_size=None, n_splits=3, test_size=None)
for train, test in tscv.split(X):
    print("%s %s" % (train, test))

