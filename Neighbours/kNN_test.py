import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import kNN
from sklearn.cross_validation import StratifiedKFold, train_test_split
import time


def trans_class(x):
    x = x.strip()
    if x == "Iris-setosa":
        return 0
    elif x == "Iris-versicolor":
        return 1
    else:
        return 2

if __name__ == '__main__':
    data = pd.read_csv("../datasets/iris/iris.data")
    feats = data.columns[:-1]
    target_feat = 'class'
    data['class'] = data['class'].apply(trans_class)
    # shuffle data, since it is sequential in nature
    data = data.iloc[np.random.permutation(len(data))]

    X_train, X_test, y_train, y_test = train_test_split(data[feats].values, data['class'].values,
                                                        test_size=0.1, random_state=42)
    skf = StratifiedKFold(data[target_feat].values, n_folds=10, random_state=42)
    index = 0
    avg_acc = 0.0
    t0 = time.time()
    for train_index, test_index in skf:
        print "Fold : ", index+1
        visible_train = data.iloc[train_index]
        blind_train = data.iloc[test_index]
        preds = kNN.knn_classifier(visible_train[feats].values, visible_train['class'].values, blind_train[feats].values,
                               distance_metric='minkowski', power=5)
        acc = accuracy_score(blind_train['class'].values, preds)
        print "\tAccuracy :", acc
        index += 1
        avg_acc += acc
    print "Avg accuracy :", avg_acc/index
    print "Time elapsed : %0.2f secs" % (time.time() - t0)