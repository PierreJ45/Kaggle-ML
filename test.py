from data import get_train_data, get_test_data
from features import *
import pandas as pd
from shapely.geometry import Polygon
from tqdm import tqdm


features = base_features

train_x, train_y, _, _, normalized_coeffs = get_train_data(features, n_data=100, val_size=0.0, file_name="data/train.geojson")
test_x = get_test_data(features, normalized_coeffs)

# common_data = pd.merge(train_x, test_x, how='inner')

# print(common_data.shape, test_x.shape, train_x.shape)

tolerance = 1e-2

def approximately_equal(val1, val2):
    try:
        return abs(float(val1) - float(val2)) < tolerance
    except ValueError:
        return val1 == val2

def geometry_equal(geom1, geom2):
    return geom1.equals_exact(geom2, tolerance)

common_data_indices = []
for i, train_point in tqdm(train_x.iterrows()):
    for j, test_point in test_x.iterrows():
        numerical_equal = all(approximately_equal(train_point[feature], test_point[feature]) for feature in train_point.index if not isinstance(train_point[feature], Polygon))
        geometric_equal = all(geometry_equal(train_point[feature], test_point[feature]) for feature in train_point.index if isinstance(train_point[feature], Polygon))
        if numerical_equal and geometric_equal:
            common_data_indices.append((i, j))
            print((i, j), "is in the train and test sets")

print("Number of approximately equal data points between train_x and test_x:", len(common_data_indices))