from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from data import get_train_data, get_test_data
from features import *
from result import create_result_file
import numpy as np
import pandas as pd
from tqdm import tqdm


features = ["duration", "area", "perimeter", "elongation"] + start_color_features + end_color_features + URBAN_FEATURES + GEOGRAPHY_FEATURES

train_x, train_y, val_x, val_y, normalized_coeffs = get_train_data(features, n_data=-1, val_size=0.2, file_name="data/train.geojson")
# test_x = get_test_data(features, normalized_coeffs)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_x, train_y)

rf = make_pipeline(
    PCA(n_components=11),
    RandomForestClassifier(
        n_estimators=300,
        max_depth=59,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight='balanced'
    )
)
rf.fit(train_x, train_y)

def combined_predict(X):
    y = np.zeros(len(X))
    knn_preds = knn.predict(X)
    for i in tqdm(range(len(X))):
        neighbors = knn.kneighbors(X.iloc[[i]], return_distance=False)[0]
        if len(np.unique(train_y.values[neighbors])) <= 1:
            y[i] = knn_preds[i]
        else:
            y[i] = rf.predict(X.iloc[[i]])[0]
    return y


# combined_preds = combined_predict(test_x)

# create_result_file(combined_preds, 'combined.csv')
print(f1_score(val_y, combined_predict(val_x), average="macro"))