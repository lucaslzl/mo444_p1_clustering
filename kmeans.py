import numpy as np
from scipy.spatial import distance


class KMeans:
    def __init__(self, k=3, max_iter=100, init='kmeans++', random_seed=None):
        self.k = k
        self.max_iter = max_iter
        self.init = init.lower()
        self.random_seed = random_seed
        self.inertia = None
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
                # compute the mean only if the cluster is not empty
                if cluster:
                    self.centroids[label] = np.mean(cluster, axis=0)

            # compute inertia
            self._compute_inertia()

            # save current centroids and clusters to history
            self.history[i] = {'centroids': dict(self.centroids), 'clusters': dict(self.clusters), 'inertia': self.inertia}

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
        random_centroids = np.random.choice(len(data), self.k, replace=False)

        for i in range(self.k):
            self.centroids[i] = data[random_centroids[i]]

    def _kmeans_plus_plus_init(self, data):
        np.random.seed(self.random_seed)
        # the first centroid is chosen at random
        centroids = [data[np.random.choice(len(data))]]

        for i in range(self.k - 1):
            dx_array = []
            for data_point in data:
                centroid_distances = []
                # compute the distance from the data point to each centroid
                for centroid in centroids:
                    centroid_distance = distance.euclidean(data_point, centroid)
                    centroid_distances.append(centroid_distance)

                # dx denotes the square of the shortest distance from the data point to a centroid
                dx = np.min(centroid_distances) ** 2
                dx_array.append(dx)

            # compute the probabilities
            square_sum = np.sum(dx_array)
            probabilities = np.divide(dx_array, square_sum)
            # the point with the highest probability is the new centroid
            highest_probability = np.max(probabilities)
            new_centroid = data[np.where(probabilities == highest_probability)][0]
            centroids.append(new_centroid)

        for i in range(self.k):
            self.centroids[i] = centroids[i]

    def _centroid_distance(self, data_point):
        # compute the euclidean distance from the data point to each centroid
        distances = []
        for _, centroid in self.centroids.items():
            centroid_distance = distance.euclidean(data_point, centroid)
            distances.append(centroid_distance)

        return distances

    def _compute_inertia(self):
        # compute the inertia (sum of squared distances of samples to their closest centroid)
        inertia = 0
        for label, centroid in self.centroids.items():
            for point in self.clusters[label]:
                error = distance.euclidean(centroid, point)
                inertia += error ** 2
        self.inertia = inertia