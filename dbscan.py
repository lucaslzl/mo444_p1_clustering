from scipy.spatial import distance

from model import Model


class DBScan(Model):


    def __init__(self, distance=0.5, min_neighbors=3):

        self.distance = distance
        self.min_neighbors = min_neighbors


    def _verify_neighbor(self, point_a, point_b):

        calc_dist = distance.euclidean(point_a, point_b)

        if calc_dist < self.distance:
            return True

        return False


    def _get_neighbors(self, x, i):

        neighbors = []

        for j in range(len(x)):

            if i == j:
                continue

            if _verify_neighbor(x[i], x[j]):
                neighbors.append(j)

        return neighbors


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

                if clusters[j] != -1
                    continue

                post_neighbors = self._get_neighbors(x, j)

                if len(neighbors) > 0 and len(post_neighbors) > self.min_neighbors:
                    neighbors.extend(post_neighbors)

                indx += 1

            cluster_id += 1


    def predict(self, y):
        pass


db = DBScan()
db.fit([[0, 0], [0, 1], [1, 0], [1, 1]])
