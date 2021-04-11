import numpy as np
from scipy.spatial import distance

from model import Model


class DBScan(Model):


    def __init__(self, distance=1.0, min_neighbors=2):

        self.distance = distance
        self.min_neighbors = min_neighbors


    def _verify_neighbor(self, point_a, point_b):

        calc_dist = distance.euclidean(point_a, point_b)

        if calc_dist < self.distance:
            return True, self.distance

        return False, self.distance


    def _get_neighbors(self, x, i):

        neighbors = []

        for j in range(len(x)):

            if i == j:
                continue

            verif, _ = self._verify_neighbor(x[i], x[j])

            if verif:
                neighbors.append(j)

        return neighbors


    def _get_neighbors_predict(self, x, i, y):

        neighbors = []

        for j in range(len(x)):

            verif, dist = self._verify_neighbor(y[i], x[j])

            if verif:
                neighbors.append((j, dist))

        return neighbors

    
    def _get_by_closest(self, neighbors):

        neighbors = np.array(neighbors)
        return neighbors[neighbors[:, 1].argsort()]


    def fit(self, x):

        # Description
        # -1 - no cluster
        # 0 - edge point
        # 1-n clusters
        clusters = [-1] * len(x)

        cluster_id = 1

        for i in range(len(x)):

            if clusters[i] == -1:
                continue

            neighbors = self._get_neighbors(x, i)

            if len(neighbors) > 0 and len(neighbors) < self.min_neighbors:
                clusters[i] = 0

            indx = 0
            while (indx > len(neighbors)):

                j = neighbors[indx]

                if clusters[j] == 0:
                    clusters[j] = cluster_id

                if clusters[j] != -1:
                    continue

                post_neighbors = self._get_neighbors(x, j)

                if len(neighbors) > 0 and len(post_neighbors) > self.min_neighbors:
                    neighbors.extend(post_neighbors)

                indx += 1

            cluster_id += 1

        return clusters


    def predict(self, x, clusters, y):

        clusters_pred = [-1] * len(y)

        new_cluster_id = max(clusters) + 1

        for i in range(len(y)):

            neighbors = self._get_neighbors_predict(x, i, y)

            if len(neighbors) > 0:

                neighbors = self._get_by_closest(neighbors)

                for j in range(len(neighbors)):

                    print(f'J: {neighbors[j]}')

                    if clusters[j] > 0:
                        clusters_pred[i] = clusters[j]
                        break

            if clusters_pred[i] == -1:
                clusters_pred[i] = new_cluster_id
                new_cluster_id += 1

        return clusters_pred


x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0.5, 0.5], [0.7, 0.7]]

db = DBScan()
cl = db.fit(x)
clp = db.predict(x, cl, y)

print(cl)
print(clp)