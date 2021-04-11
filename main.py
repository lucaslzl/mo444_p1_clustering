from io import read_csv
from dbscan import DBScan
from kmeans import KMeans


class Main:

    dataset = read_csv()

    for model in [DBScan(), KMeans()]:

        c = model.fit(dataset)
        model.predict(dataset, c, test)

        pass
