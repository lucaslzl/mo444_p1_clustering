import numpy as np
from scipy import stats
from scipy.spatial import distance
from tqdm import tqdm

from model import Model
from inout import *


class DBScan(Model):


    def __init__(self, distance=0.12, min_neighbors=2):

        self.distance = distance
        self.min_neighbors = min_neighbors
        self.summed_dist = []


    def _verify_neighbor(self, point_a, point_b):

        # Calculate euclidean distance
        calc_dist = distance.euclidean(point_a, point_b)

        # Append distance to verify description
        self.summed_dist.append(calc_dist)

        # Verify if it is a neighbor
        if calc_dist <= self.distance:
            return True, self.distance

        return False, self.distance


    def _get_neighbors(self, x, i):

        # Neighbor list
        neighbors = []

        for j in range(len(x)):

            if i == j:
                continue
            
            # Verify if it is a neighbor
            verif, _ = self._verify_neighbor(x[i], x[j])

            if verif:
                # Append to the list of neighbors
                neighbors.append(j)

        return neighbors


    def _get_neighbors_predict(self, x, i, y):

        # Neighbor list
        neighbors = []

        for j in range(len(x)):

            # Verify if it is a neighbor
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

        # Initialize with 0's
        nc = [0] * len(x)
        ci = [0] * len(x)

        cluster_id = 0

        # Iterate through all records
        for i in tqdm(range(len(x))):

            # If already classified, skip
            if nc[i] != 0 or ci[i] != 0:
                continue

            # Get neighbors
            neighbors = self._get_neighbors(x, i)

            # Verify if it is an outlier
            if len(neighbors) < self.min_neighbors:
                nc[i] = -1
                continue

            cluster_id += 1

            # Core record
            nc[i] = 2
            ci[i] = cluster_id

            # Iterate through each neighbor
            indx = 0
            while True:

                # If list of neighbors ended
                if indx == len(neighbors):
                    break
                
                j = neighbors[indx]

                if i == j:
                    indx += 1
                    continue

                # At least it is a border point
                nc[j] = 1
                ci[j] = cluster_id

                post_neighbors = self._get_neighbors(x, j)

                # Verify if neighbor is core point
                if len(post_neighbors) >= self.min_neighbors:
                    # Classify as core point
                    nc[j] = 2

                    # Continue exploring neighbourhood
                    neighbors.extend(post_neighbors)
                    neighbors = list(set(neighbors))

                indx += 1

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

        # Initialize with 0's
        ci_pred = [0] * len(y)

        for i in tqdm(range(len(y))):

            # Get neighbors
            neighbors = self._get_neighbors_predict(x, i, y)

            if len(neighbors) > 0:

                # Get closest neighbors
                neighbors = self._get_by_closest(neighbors)

                for j in range(len(neighbors)):

                    # Get closest neighbors
                    indx_j = int(neighbors[j][0])

                    # Verify if core point
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