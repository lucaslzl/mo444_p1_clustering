import pandas as pd
from sklearn.model_selection import train_test_split


################################################
###              READ DATA                   ###
################################################

def read_csv(file_name):

    return pd.read_csv(file_name, sep=' ')


def read_datasets(datasets=['cluster.dat']):

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

def plot():
    pass