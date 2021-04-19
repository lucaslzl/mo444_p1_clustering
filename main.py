from inout import *
from dbscan import DBScan
from kmeans import KMeans
from pca import OurPCA


class Main:

    exp_execute = 2

    ############################################
    # Experiment DBScan 1
    # Verify distances with and without scale
    ############################################

    if exp_execute == 1:

        for d, scale in zip([0, 0, 1, 1], [False, True, False, True]):

            datasets = read_datasets()

            if scale:
                datasets = scale_datasets(datasets)
            
            datasets = split_data(datasets)

            dist = 1.0
            min_neighb = 3

            model = DBScan(distance=dist, min_neighbors=min_neighb)
            dataset = datasets[d]

            res = model.fit(dataset['train'].to_numpy())
            res = model.get_description_dist()
            print(res)

    ############################################
    # Experiment DBScan 2
    # Execute model for Datset 1
    ############################################

    if exp_execute == 2:

        datasets = read_datasets()

        datasets = scale_datasets(datasets)
        datasets = split_data(datasets)

        for dist in [0.05, 0.1, 0.15]:

            df_merge = datasets[0]['train'].copy()
            df_merge['Type'] = -2
            df_merge['Cluster'] = -2
            df_merge['Experiment'] = 'remove'

            for min_neighb in [3, 5, 8]:

                model = DBScan(distance=dist, min_neighbors=min_neighb)
                dataset = datasets[0]

                res = model.fit(dataset['train'].to_numpy())
                pred = model.predict(dataset['train'].to_numpy(), res, dataset['test'].to_numpy())

                # plot(dataset['train'].copy(), res, file_name=f'dbscan_vardist_{dist}_{min_neighb}.png')

                write_results(datasets, f'dbscan_vardist_{dist}_{min_neighb}_datasets')
                write_results(res, f'dbscan_vardist_{dist}_{min_neighb}_res')
                write_results(pred, f'dbscan_vardist_{dist}_{min_neighb}_pred')

                df_merge = pd.concat([df_merge, merge_result(dataset['train'].copy(), res, f'Distance: {dist} Neighbor: {min_neighb}')])

            plot_all(df_merge, file_name=f'all_dbscan_{dist}.png')

    ############################################
    # Experiment DBScan 3
    # Execute model for Datset 2
    ############################################

    if exp_execute == 2:

        datasets = read_datasets()

        datasets = scale_datasets(datasets)
        datasets = split_data(datasets)

        for dist in [0.05, 0.1, 0.15]:

            df_merge = datasets[1]['train'].copy()
            df_merge['Type'] = -2
            df_merge['Cluster'] = -2
            df_merge['Experiment'] = 'remove'

            for min_neighb in [3, 5, 8]:

                model = DBScan(distance=dist, min_neighbors=min_neighb)
                dataset = datasets[1]

                res = model.fit(dataset['train'].to_numpy())
                pred = model.predict(dataset['train'].to_numpy(), res, dataset['test'].to_numpy())

                # plot(dataset['train'].copy(), res, file_name=f'dbscan_vardist_{dist}_{min_neighb}.png')

                write_results(datasets, f'dbscan_vardist_{dist}_{min_neighb}_datasets')
                write_results(res, f'dbscan_vardist_{dist}_{min_neighb}_res')
                write_results(pred, f'dbscan_vardist_{dist}_{min_neighb}_pred')

                df_merge = pd.concat([df_merge, merge_result(dataset['train'].copy(), res, f'Distance: {dist} Neighbor: {min_neighb}')])

            plot_all(df_merge, file_name=f'all_dbscan_{dist}.png')
