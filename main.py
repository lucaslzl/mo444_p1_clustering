from inout import *
from dbscan import DBScan
from kmeans import KMeans
from pca import OurPCA

#### Tasks
## Clustering
# - Evaluate different parameters
# - Use metrics to evaluate

## PCA
# - Compare results


class Main:

    """
    # Experiment 1
    - Read datasets
    - Split datasets
    - Call models

    # Experiment 2
    - Read datasets
    - Scale datasets
    - Split datasets
    - Call models

    # Experiment 3
    - Read datasets
    - Scale datasets
    - Split datasets
    - PCA
    - Call models
    """

    datasets = read_datasets()

    datasets = scale_datasets(datasets)
    datasets = split_data(datasets)
    
    model = DBScan()

    for dataset in datasets:

        res = model.fit(dataset['train'].to_numpy())
        pred = model.predict(dataset['train'].to_numpy(), res, dataset['test'].to_numpy())

        plot(dataset['train'], res)


    # datasets = read_datasets()

    # for n_components in [range(2, 3, 4)]:

    #     for i, dataset in enumerate(datasets):

    #         dataset = OurPCA().fit_transform(dataset.to_numpy(), n_components)

    #         dataset = scale_datasets([dataset])
    #         dataset = split_data([dataset])
            
    #         model = KMeans()
    #         res = model.fit(dataset['train'].to_numpy())
    #         pred = model.predict(dataset['train'].to_numpy(), res, dataset['test'].to_numpy())

    #         plot(dataset['train'], res)
