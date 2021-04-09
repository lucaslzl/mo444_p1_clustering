import numpy as np
from scipy.spatial import distance

from model import Model


class KMeans(Model):
    def __init__(self, k=3, max_iter=100, init='forgy', random_seed=None):
        super().__init__()
        self.k = k
        self.max_iter = max_iter
        self.init = init.lower()
        self.random_seed = random_seed
        self.centroids = {}
        self.clusters = {}
        self.history = {}

    def fit(self, data):
        # data initialization
        self._data_init(data)

        for i in range(self.max_iter):
            # assign the previous centroids
            previous_centroids = dict(self.centroids)

            # initialize the clusters dictionary
            for label in range(self.k):
                self.clusters[label] = []

            # assign each data point to a cluster
            for data_point in data:
                distances = self._centroid_distance(data_point)
                # the data point is assigned to the nearest cluster
                label = np.argmin(distances)
                self.clusters[label].append(data_point)

            # the position of each centroid is moved to the mean of the points in the cluster
            for label, cluster in self.clusters.items():
                self.centroids[label] = np.mean(cluster, axis=0)

            # save current centroids and clusters to history
            self.history[i] = {'centroids': dict(self.centroids), 'clusters': dict(self.clusters)}

            # check if the algorithm has converged
            converged = True
            for label, centroid in self.centroids.items():
                previous_centroid = previous_centroids[label]
                centroid_distance = distance.euclidean(centroid, previous_centroid)
                # if the distance between the current and previous centroid is greater than zero
                # then the algorithm hasn't converged yet
                if centroid_distance > 0:
                    converged = False

            # if the algorithm converged then the learning process is finished
            if converged:
                break

    def predict(self, data):
        predict_data = []
        for data_point in data:
            # the data point is assigned to the nearest cluster
            distances = self._centroid_distance(data_point)
            label = np.argmin(distances)
            predict_data.append(label)

        return predict_data

    def _data_init(self, data):
        if self.init == 'kmeans++':
            self._kmeans_plus_plus_init(data)
        else:
            self._forgy_init(data)

    def _forgy_init(self, data):
        # chooses k random points from the data as initial centroids
        np.random.seed(self.random_seed)
        random_centroids = np.random.choice(len(data), self.k)

        for i in range(self.k):
            self.centroids[i] = data[random_centroids[i]]

    def _kmeans_plus_plus_init(self, data):
        pass

    def _centroid_distance(self, data_point):
        # compute the euclidean distance from the data point to each centroid
        distances = []
        for _, centroid in self.centroids.items():
            centroid_distance = distance.euclidean(data_point, centroid)
            distances.append(centroid_distance)

        return distances