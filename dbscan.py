import numpy as np
from scipy import stats
from scipy.spatial import distance
from tqdm import tqdm

from model import Model


class DBScan(Model):


    def __init__(self, distance=0.2, min_neighbors=3):

        self.distance = distance
        self.min_neighbors = min_neighbors
        self.summed_dist = []


    def _verify_neighbor(self, point_a, point_b):

        calc_dist = distance.euclidean(point_a, point_b)

        self.summed_dist.append(calc_dist)

        if calc_dist <= self.distance:
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
        # [nc, ci]
        # (nc) node classification
        # -- 0  : none
        # -- -1 : noise
        # -- 1  : border
        # -- 2  : core
        # (ci) cluster id

        nc = [0] * len(x)
        ci = [0] * len(x)

        cluster_id = 1

        for i in tqdm(range(len(x))):

            if nc[i] != 0:
                continue

            neighbors = self._get_neighbors(x, i)

            if len(neighbors) < self.min_neighbors:
                nc[i] = -1
                continue

            nc[i] = 2
            ci[i] = cluster_id

            indx = 0
            while True:

                if indx == len(neighbors):
                    break

                j = neighbors[indx]

                if nc[j] == -1:
                    nc[j] = 1
                    ci[j] = cluster_id

                if nc[j] != 0:
                    indx += 1
                    continue

                post_neighbors = self._get_neighbors(x, j)

                if len(post_neighbors) >= self.min_neighbors:

                    nc[j] = 2
                    ci[j] = cluster_id

                    neighbors.extend(post_neighbors)
                    neighbors = list(set(neighbors))
                
                else:

                    nc[j] = 1
                    ci[j] = cluster_id

                indx += 1

            cluster_id += 1

        return (nc, ci)


    def predict(self, x, res, y):

        # Description
        # [nc, ci]
        # (nc) node classification
        # -- 0  : none
        # -- -1 : noise
        # -- 1  : border
        # -- 2  : core
        # (ci) cluster id

        (nc, ci) = res

        ci_pred = [0] * len(y)

        new_cluster_id = max(ci) + 1

        for i in tqdm(range(len(y))):

            neighbors = self._get_neighbors_predict(x, i, y)

            if len(neighbors) > 0:

                neighbors = self._get_by_closest(neighbors)

                for j in range(len(neighbors)):

                    indx_j = int(neighbors[j][0])

                    if nc[indx_j] == 2:
                        ci_pred[i] = ci[indx_j]
                        break

        return ci_pred

    
    def get_description_dist(self):
        return f'Mean: {np.mean(self.summed_dist)}\nMedian: {np.median(self.summed_dist)}\nMode: {stats.mode(self.summed_dist)[0][0]}'


# x = [[0, 0], [0, 1], [1, 0], [1, 1]]
# y = [[0.5, 0.5], [0.7, 0.7]]

# db = DBScan()
# cl = db.fit(x)
#clp = db.predict(x, cl, y)

# print('\n Result')
# print(cl)
# print(clp)