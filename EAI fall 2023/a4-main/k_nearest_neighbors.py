# k_nearest_neighbors.py: Machine learning implementation of a K-Nearest Neighbors classifier from scratch.
#
# Submitted by: [Aniket Sharma - Shreya Parab - Vaishnavi Rai] -- [anikshar-shparab-vairai]
#
# Based on skeleton code by CSCI-B 551 Fall 2023 Course Staff

import numpy as np
from utils import euclidean_distance, manhattan_distance
from collections import Counter


class KNearestNeighbors:
    """
    A class representing the machine learning implementation of a K-Nearest Neighbors classifier from scratch.

    Attributes:
        n_neighbors
            An integer representing the number of neighbors a sample is compared with when predicting target class
            values.

        weights
            A string representing the weight function used when predicting target class values. The possible options are
            {'uniform', 'distance'}.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model and
            predicting target class values.

        _y
            A numpy array of shape (n_samples,) representing the true class values for each sample in the input data
            used when fitting the model and predicting target class values.

        _distance
            An attribute representing which distance metric is used to calculate distances between samples. This is set
            when creating the object to either the euclidean_distance or manhattan_distance functions defined in
            utils.py based on what argument is passed into the metric parameter of the class.

    Methods:
        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_neighbors = 5, weights = 'uniform', metric = 'l2'):
        # Check if the provided arguments are valid
        if weights not in ['uniform', 'distance'] or metric not in ['l1', 'l2'] or not isinstance(n_neighbors, int):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the KNearestNeighbors model object
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._X = None
        self._y = None
        self._distance = euclidean_distance if metric == 'l2' else manhattan_distance

    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """
        self._X = X
        self._y = y

        return

        # raise NotImplementedError('This function must be implemented by the student.')

    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """

        output = []
        for i in range(len(X)):
            dist = []
            for j in range(len(self._X)):
                d1 = self._distance(self._X[j], X[i])
                dist.append((d1, j))

                #d2 = euclidean_distance(self._X[j], X[i])
                #Euclidean Distance = sqrt(sum i to N (x1_i — x2_i)²)

                #d1.append([d2, j])
            dist.sort()
            neighbors = dist[:self.n_neighbors]
            #Manhatten Distance = sum for i to N sum || X1i – X2i ||

            if self.weights == 'dist':
                votes = Counter()
                for d1, index in neighbors:
                    weight = 1 / max(d1, 1e-6)
                    votes[self._y[index]] += weight
                ans = max(votes, key = votes.get)
                
            else:
                neighbor_label = [self._y[index] for d1, index in neighbors]
                ans = Counter(neighbor_label).most_common(1)[0][0]
            output.append(ans)
        return output

        # https://towardsdatascience.com/how-to-build-knn-from-scratch-in-python-5e22b8920bd2
        # raise NotImplementedError('This function must be implemented by the student.')
