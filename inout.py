import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


################################################
###              READ DATA                   ###
################################################

def read_csv(file_name, sep=' '):

    return pd.read_csv(file_name, sep=sep)


def read_datasets(datasets=['cluster.dat', 'credit.csv']):

    return [read_csv(f'datasets/{file}') for file in datasets]


################################################
###              WRITE DATA                  ###
################################################

def write_results():
    pass

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


################################################
###              PLOT DATA                   ###
################################################

def plot(data, res):
    
    (nc, ci) = res

    data['Type'] = nc
    data['Cluster'] = ci

    cols = list(data.columns)

    sns.set_theme()

    sns.relplot(
        data=data,
        x=f"{cols[0]}", y=f"{cols[1]}", hue="Cluster", style="Type", 
    )

    plt.show()