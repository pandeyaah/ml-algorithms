import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


def predict_prob(feature_matrix, coefficients):
    product = feature_matrix.dot(coefficients)

    predictions = 1.0 / (1 + np.exp(-product))
    return predictions


def feature_deriv(errors, feature, coefficients, l2_penal, is_constant):
    derivative = errors.dot(feature)
    if not is_constant:
        derivative = derivative - 2 * l2_penal * coefficients
    return derivative


def log_res(X_train, y_train, initial_coefficients, step_size, l2_penalty, max_iter):
    coefficients = np.array(initial_coefficients)
    #train_data['intercept'] = 1
    #feats = train_data.columns
    #feats = feats.difference([target_feat])
    for itr in xrange(max_iter):
        predictions = predict_prob(X_train, coefficients)
        indicator = (y_train==+1)

        errors = indicator - predictions
        for j in xrange(len(coefficients)):
            is_intercept = (j == 0)
            derivative = feature_deriv(errors, X_train[:, j],
                                       coefficients[j], l2_penalty, is_intercept)
            coefficients[j] += step_size * derivative
    return coefficients


def get_classification_accuracy(feature_matrix, y, coefficients):
    scores = np.dot(feature_matrix, coefficients)
    apply_threshold = np.vectorize(lambda x: 1. if x > 0  else -1.)
    predictions = apply_threshold(scores)

    num_correct = (predictions == y).sum()
    accuracy = num_correct / len(feature_matrix)
    return accuracy


def class_predict(feature_matrix, coefficients):
    scores = predict_prob(feature_matrix, coefficients)
    classes = np.zeros(feature_matrix.shape[0], dtype=np.int8)
    for i in xrange(len(scores)):
        if scores[i] > 0.5:
            classes[i] = 1
        else:
            classes[i] = 0
    return classes


def trans_class(x):
    x = x.strip()
    if x == "Iris-setosa":
        return 0
    elif x == "Iris-versicolor":
        return 1
    else:
        return 2


if __name__ == '__main__':
    data = pd.read_csv("./datasets/iris/iris.data")
    data['intercept'] = 1
    feats = data.columns[:-1]
    target_feat = 'class'
    data['class'] = data['class'].apply(trans_class)
    # shuffle data, since it is sequential in nature
    data = data.iloc[np.random.permutation(len(data))]
    X_train, X_test, y_train, y_test = train_test_split(data[feats].values, data['class'].values,
                                                        test_size=0.1, random_state=42)

    coefficient_pen_0 = log_res(X_train, y_train, np.zeros(data[feats].shape[1]), step_size=5e-6,l2_penalty=0, max_iter=501)
    coefficient_pen_4 = log_res(X_train, y_train, np.zeros(data[feats].shape[1]), step_size=5e-6,l2_penalty=4, max_iter=501)
    coefficient_pen_10 = log_res(X_train, y_train, np.zeros(data[feats].shape[1]), step_size=5e-6,l2_penalty=10, max_iter=501)
    coefficient_pen_1e2 = log_res(X_train, y_train, np.zeros(data[feats].shape[1]), step_size=5e-6,l2_penalty=1e2, max_iter=501)

    preds = class_predict(X_test, coefficient_pen_0)
    print preds
