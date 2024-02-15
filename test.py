"""
This script can be used as skelton code to read the challenge train and test
geojsons, to train a trivial model, and write data to the submission file.
"""
import geopandas as gpd
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from data import get_train_data, get_test_data


from sklearn.neighbors import KNeighborsClassifier

from result import create_result_file

change_type_map = {'Demolition': 0, 'Road': 1, 'Residential': 2, 'Commercial': 3, 'Industrial': 4, 'Mega Projects': 5}

## Read csvs

# train_df = gpd.read_file('data/train.geojson', index_col=0)
# test_df = gpd.read_file('data/test.geojson', index_col=0)

## Filtering column "mail_type"
# train_x = np.asarray(train_df[['geometry']].area)
# train_x = train_x.reshape(-1, 1)
# train_y = train_df['change_type'].apply(lambda x: change_type_map[x])

train_x, train_y, _, _, normalized_coeffs = get_train_data(['area'], n_data=-1, val_size=0.0, file_name="data/train.geojson")
# train_x = np.asarray(train_x[['area']]).reshape(-1, 1)

# test_x = np.asarray(test_df[['geometry']].area)
# test_x = test_x.reshape(-1, 1)

test_x = get_test_data(['area'], normalized_coeffs)
# test_x = np.asarray(test_x).reshape(-1, 1)

print (train_x.shape, train_y.shape, test_x.shape)


## Train a simple OnveVsRestClassifier using featurized data
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_x, train_y)
create_result_file(neigh.predict(test_x), 'knn_submission_verif.csv')