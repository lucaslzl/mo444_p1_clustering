from inout import read_datasets, scale_datasets, split_data
from dbscan import DBScan
from kmeans import KMeans

#### Tasks
## Clustering
# - Select another Dataset
# - Train with new dataset
# - Evaluate different parameters
# - Use metrics to evaluate

## PCA
# - Implement PCA
# - Run kmeans with distinct clusters
# - Compare results


class Main:

    datasets = read_datasets()

    datasets = scale_datasets(datasets)
    datasets = split_data(datasets)
    
    for model in [DBScan()]:

        for dataset in datasets:

            clusters = model.fit(dataset['train'].to_numpy())
            pred = model.predict(dataset['train'].to_numpy(), clusters, dataset['test'].to_numpy())

            print(clusters)
            print(pred)
