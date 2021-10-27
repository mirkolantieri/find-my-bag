import warnings
import numpy as np
import pandas as pd

print("Files imported successfully")
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 30*30 + 1)


class KNearestNeighbor(object):
    """ KNN classifier with L2 distance a.k.a Euclidean """

    def __init__(self) -> None:
        super().__init__()
        self.y_train = None
        self.y_pred = None
        self.X_train = None

    def predict_label(self, dist: np.ndarray, k: int = 1):
        """ `predict_label`: returns the predicted true labels """
        num_test = dist.shape[0]
        self.y_pred = np.zeros(num_test)

        closest = []

        for i in range(num_test):
            closest = self.y_train[np.argsort(dist[i])][0:k]
            self.y_pred = np.bincount(closest).argmax()

        return self.y_pred

    def train(self, x: np.array, y: np.array):
        """ Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i]. """

        self.X_train = x
        self.y_train = y

    def predict(self, X: np.array, k: int = 1):
        """ Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i]. """

        dist = self.compute_distance(X)

        return self.predict_labels(dist, k=k)

    def compute_distance(self, X: np.array):
        """ Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops. """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dist = np.zeros((num_test, num_train))

        dist = np.sqrt((X ** 2).sum(axis=1, keepdims=1) +
                       (self.X_train ** 2).sum(axis=1) - 2 * X.dot(self.X_train.T))
        return dist

    def predict_labels(self, dist: np.array, k: int = 1):
        """ Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """

        num_test = dist.shape[0]
        self.y_pred = np.zeros(num_test)
        closest = []

        for i in range(num_test):
            closest = self.y_train[np.argsort(dist[i])][0:k]
            closest = closest.astype(int)
            self.y_pred[i] = np.bincount(closest).argmax()
            
        return self.y_pred
