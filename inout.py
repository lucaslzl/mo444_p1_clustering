import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from kmeans import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import pickle


################################################
###              READ DATA                   ###
################################################

def read_csv(file_name, sep=' '):
    return pd.read_csv(file_name, sep=sep)


def read_datasets():
    datasets = []

    for file, sep in zip(['cluster.dat'], [' ', None]):
        dataset = read_csv(f'datasets/{file}', sep)
        dataset = dataset.fillna(0)

        datasets.append(dataset)

    return datasets


def read_results(file_name):

    with open (f'./results/{file_name}.p', 'rb') as fp:
        results = pickle.load(fp)

    return results

################################################
###              WRITE DATA                  ###
################################################

def write_results(result, file_name):
    
    with open(f'./results/{file_name}.p', 'wb') as fp:
        pickle.dump(result, fp)


################################################
###           TRANSFORM DATA                 ###
################################################

def scale(data):
    for c in data.columns:
        data[c] = data[c] / max(data[c])

    return data


def scale_datasets(datasets):
    for d in range(len(datasets)):
        datasets[d] = scale(datasets[d])

    return datasets


def split_data(datasets):
    data = []

    for d in range(len(datasets)):
        train, test = train_test_split(datasets[d], test_size=0.1, random_state=42)
        data.append({'train': train,
                     'test': test})

    return data


def merge_result(data, res, dist_var):

    (nc, ci) = res

    data['Type'] = nc
    data['Cluster'] = ci
    data['Experiment'] = dist_var

    return data


################################################
###              PLOT DATA                   ###
################################################

def plot(data, res, file_name):
    
    (nc, ci) = res

    data['Type'] = nc
    data['Cluster'] = ci

    cols = list(data.columns)

    sns.set_theme()
    palette = sns.color_palette("vlag", as_cmap=True)

    sns.relplot(
        data=data,
        x=f"{cols[0]}", y=f"{cols[1]}", hue="Cluster", style='Type',
        palette=palette 
    )

    plt.savefig(f'./plots/{file_name}')


def plot_all(data, file_name):

    data = data[data['Experiment'] != 'remove']

    cols = list(data.columns)

    sns.set_theme()
    palette = sns.color_palette("vlag", as_cmap=True)

    sns.relplot(
        data=data,
        x=f"{cols[0]}", y=f"{cols[1]}", hue="Cluster", style='Type',
        palette=palette
    )

    plt.savefig(f'./plots/{file_name}')


def plot_silhouette(data, interval, show_clusters=False, random_seed=None):
    for n_clusters in interval:
        if show_clusters:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)
        else:
            ax1 = plt.subplot()

        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])

        clf = KMeans(k=n_clusters, init='kmeans++', random_seed=random_seed)
        clf.fit(data)
        cluster_labels = np.array(clf.predict(data))

        silhouette_avg = silhouette_score(data, cluster_labels)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(data, cluster_labels)

        y_lower = 10

        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        title = "The silhouette plot for the various cluster\nAverage silhouette_score: {}".format(silhouette_avg)
        if not show_clusters:
            title = "The silhouette plot for {} clusters\nAverage silhouette_score: {}".format(n_clusters,
                                                                                               silhouette_avg)
        ax1.set_title(title)
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        if show_clusters:
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(data[:, 0], data[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                        c=colors, edgecolor='k')

            # Labeling the clusters
            centers = np.array([v for k, v in clf.centroids.items()])
            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

        if show_clusters:
            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                          "with n_clusters = %d" % n_clusters),
                         fontsize=14, fontweight='bold')

        plt.show()


def plot_metrics(k, metric, title='', xlabel='', ylabel=''):
    lower_bound_k = np.min(k)
    upper_bound_k = np.max(k)
    step_k = max(1, round((upper_bound_k - lower_bound_k) / len(k)))
    lower_bound_inertia = np.min(metric)
    upper_bound_inertia = np.max(metric)
    step_inertia = max(1, (upper_bound_inertia - lower_bound_inertia) / len(metric))
    plt.xticks(np.arange(lower_bound_k, upper_bound_k + step_k, step=step_k))
    plt.yticks(np.arange(lower_bound_inertia, upper_bound_inertia + step_inertia, step=step_inertia))
    plt.plot(k, metric, color='purple')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()


def plot_elbow(interval, data, random_seed=None):
    inertia = []
    for n_clusters in interval:
        clf = KMeans(k=n_clusters, init='kmeans++', random_seed=random_seed)
        clf.fit(data)
        inertia.append(clf.inertia)
    plot_metrics(interval, inertia, 'Elbow method', 'Number of clusters (K)', 'Sum of Squared Error')


def plot_cluster(centroids, clusters, labels, title='', xlabel='', ylabel='', scaler=None):
    for label, cluster in zip(labels, clusters):
        if scaler is not None:
            cluster = scaler.inverse_transform(cluster)
        color = plt.get_cmap('hsv')(float(label) / len(centroids))
        plt.scatter(cluster[0], cluster[1], marker='+', color=color)

    plt.scatter(centroids[:, 0], centroids[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centroids):
        plt.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()


def plot_cluster_by_iteration(clf):
    for i, history in clf.history.items():
        centroids = np.array(list(history['centroids'].values()))
        clusters = []
        labels = []
        for label, cluster in history['clusters'].items():
            for data in cluster:
                clusters.append(data)
                labels.append(label)

        plot_cluster(centroids, clusters, labels, 'Clusters as iteration {}'.format(i + 1), 'X Axis', 'Y Axis')


def plot_quality_measures(clf):
        inter_cluster_distances = []
        intra_cluster_distances = []
        k = []
        for i, history in clf.history.items():
            k.append(i + 1)
            dists = euclidean_distances([v for _, v in history['centroids'].items()])
            tri_dists = dists[np.triu_indices(clf.k, 1)]
            inter_cluster_distances.append(tri_dists.sum() ** 2)
            intra_cluster_distances.append(history['inertia'])

        plot_metrics(k, inter_cluster_distances, 'Inter cluster distance', 'Iteration', 'Sum of Squared Distance')
        plot_metrics(k, intra_cluster_distances, 'Intra cluster distance', 'Iteration', 'Inertia')
