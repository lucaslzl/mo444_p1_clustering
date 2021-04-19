import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


################################################
###              READ DATA                   ###
################################################

def read_csv(file_name, sep=' '):

    return pd.read_csv(file_name, sep=sep)


def read_datasets():

    datasets = []

    for file, sep in zip(['cluster.dat', 'credit.csv'], [' ', None]):

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
        x=f"{cols[0]}", y=f"{cols[1]}", hue="Cluster", style='Type', col='Experiment',
        palette=palette
    )

    plt.savefig(f'./plots/{file_name}')