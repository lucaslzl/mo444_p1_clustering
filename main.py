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
    
    # model = DBScan()
    #
    # for dataset in datasets:
    #
    #     res = model.fit(dataset['train'].to_numpy())
    #     pred = model.predict(dataset['train'].to_numpy(), res, dataset['test'].to_numpy())
    #
    #     plot(dataset['train'], res)

    # Experiments - KMeans
    model = KMeans()

    datasets = read_datasets()
    scaled_datasets = scale_datasets(datasets)

    datasets = split_data(datasets)
    scaled_datasets = split_data(scaled_datasets)

    for dataset in scaled_datasets:
        # Experiment 1
        interval = range(2, 12)
        random_seed = 84
        data_train = dataset['train'].to_numpy()
        data_test = dataset['test'].to_numpy()
        # plot_elbow(interval, data_train, random_seed=random_seed)
        # plot_silhouette(data_train, interval, show_clusters=True, random_seed=random_seed)

        # Experiment 3 (forgy initialization)
        clf = KMeans(k=3, init='forgy', random_seed=random_seed)
        clf.fit(data_train)
        # plot_quality_measures(clf)
        # plot_cluster_by_iteration(clf)
        plot_cluster(np.array(list(clf.centroids.values())), data_test, clf.predict(data_test), 'Clusters (Predict)')

        # Experiment 3 (kmeans++ initialization)
        clf = KMeans(k=3, init='kmeans++', random_seed=random_seed)
        clf.fit(data_train)
        # plot_cluster_by_iteration(clf)
        # plot_quality_measures(clf)
        plot_cluster(np.array(list(clf.centroids.values())), data_test, clf.predict(data_test), 'Clusters (Predict)')
    # for dataset in datasets:
    #     random_seed = 88
    #     data_train = dataset['train'].to_numpy()
    #     data_test = dataset['test'].to_numpy()

        # Experiment 2 (forgy initialization)
        # clf = KMeans(k=3, init='forgy', random_seed=random_seed)
        # clf.fit(data_train)
        # plot_quality_measures(clf)
        # plot_cluster_by_iteration(clf)
        # plot_cluster(np.array(list(clf.centroids.values())), data_test, clf.predict(data_test), 'Clusters (Predict)')

        # Experiment 2 (kmeans++ initialization)
        # clf = KMeans(k=3, init='kmeans++', random_seed=random_seed)
        # clf.fit(data_train)
        # plot_cluster_by_iteration(clf)
        # plot_quality_measures(clf)
        # plot_cluster(np.array(list(clf.centroids.values())), data_test, clf.predict(data_test), 'Clusters (Predict)')

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
