import operator
import numpy as np


def min_max_scaler(X_train, X_test, feature_range=(0, 1)):
    """
    Min Max Scaler. Scales both X_train and X_test
    as per the min and max of X_train

    Parameters
    ----------

    X_train: array-like, shape(n_train_samples, n_features)
        training data

    X_test:  array-like, shape(n_test_samples, n_features)
        test data to predict class labels for

    feature_range: tuple. optional (default=(0,1))
        min and max ranges for scaling resp

    Returns
    -------
    X_train_scaled : scaled X_train

    X_test_scaled : scaled X_test
    """
    min_range, max_range = feature_range
    train_min = X_train.min(axis=0)
    train_max = X_train.max(axis=0)
    X_std = (X_train - train_min) / (train_max - train_min)
    X_train_scaled = X_std * (max_range - min_range) + min_range
    X_std = (X_test - train_min) / (train_max - train_min)
    X_test_scaled = X_std * (max_range - min_range) + min_range
    return X_train_scaled, X_test_scaled


def autoNorm(X_train, X_test):
    """
        Min Max Scaler. Scales both X_train and X_test
         as per the min and max of X_train

        Parameters
        ----------

        X_train: array-like, shape(n_train_samples, n_features)
            training data

        X_test:  array-like, shape(n_test_samples, n_features)
            test data to predict class labels for

        Returns
        -------
        X_train_scaled : scaled X_train

        X_test_scaled : scaled X_test
        """
    minVals = X_train.min(0)
    maxVals = X_train.max(0)
    ranges = maxVals - minVals
    norm_train_dataset = np.zeros(np.shape(X_train))
    norm_test_dataset = np.zeros(np.shape(X_test))
    m_train = X_train.shape[0]
    m_test = X_test.shape[0]
    norm_train_dataset = X_train - np.tile(minVals, (m_train, 1))
    norm_train_dataset = norm_train_dataset / np.tile(ranges, (m_train, 1))
    norm_test_dataset = X_test - np.tile(minVals, (m_test, 1))
    norm_test_dataset = norm_test_dataset / np.tile(ranges, (m_test, 1))
    return norm_train_dataset, norm_test_dataset


def classify0(test_instance, X, y, k=3, d_metric='euclidean', p=1):
    """The real deal. This implements the kNN algorithm

    Parameters
    ----------
    test_instance: array-like, shape(1, n_features)
        instance to make class predictions for

    X: array-like, shape(n_train_samples, n_features)
        training data

    y: vector, shape(n_train_samples, )
        training data labels

    k: integer. optional (default= 3)
        no of neighbours to use for computations

    d_metric: string. optional. {'minkowski', 'euclidean', 'manhattan'}
        default = 'euclidean'

    p: integer, optional (default=1)
        Power parameter for the Minkowski metric.

    Return
    ------
    predicted class label

    """
    if test_instance is None or X is None or y is None:
        ValueError("Problem with the arrays passed. Check your params")
    if k == 0:
        ValueError("k (no of neighbours) cannot be zero.")

    if d_metric is not "minkowski" and p != 1:
        print("Attention! No valid match found for d_metric and power (p) combination")

    if d_metric == 'euclidean':
        p = 2
    if d_metric == 'manhattan':
        p = 1

    data_set_size = X.shape[0]
    diff = np.absolute(np.tile(test_instance, (data_set_size, 1)) - X)
    pow_distances = (diff ** p).sum(axis=1)
    distances = pow_distances ** (1.0 / p)
    sorted_dist_indices = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_i_label = y[sorted_dist_indices[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
    sorted_class_count = sorted(
        class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def knn_classifier(X, y, X_test, k=3, distance_metric='euclidean', power=1):
    """
    Classifier that incorporates K-Nearest Neighbours Classification Technique
    Parameters
    ----------
    X : array-like, shape(n_train_samples, n_features)
        training data

    y : vector of shape (n_train_samples,)
        training labels

    X_test : array-like, shape(n_test_samples, n_features)
        test data to predict class labels for

    k : integer, optional (default = 3)
        no of neighbours to use for computations

    distance_metric : string. {'minkowski', 'euclidean', 'manhattan'}
        default = 'euclidean'
        if distance_metric is 'minkowski':
            p=2 is equivalent to standard euclidean distance
            p=1 is equivalent to manhattan distance

    power : integer, optional (default = 3)
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    Returns
    -------
    predictions : array of shape [n_test_samples]
        Class labels for each data sample.
    """
    # normalizing data
    X, X_test = min_max_scaler(X, X_test)
    # allocating space for results
    predictions = np.zeros((X_test.shape[0], 3))
    for x in range(X_test.shape[0]):
        predictions[x] = classify0(
            X_test[x], X, y, k, d_metric=distance_metric, p=power)
    return predictions
