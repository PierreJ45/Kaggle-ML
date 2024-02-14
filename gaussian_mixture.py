from data import get_train_data
from features import *
from sklearn.mixture import GaussianMixture
from sklearn.metrics import f1_score
import pandas as pd

train_x, train_y, test_x, test_y = get_train_data(
    ["duration", "area", "perimeter", "elongation", "nb_points"] + start_color_features + end_color_features,
    n_data = -1
)

gmm = GaussianMixture(n_components=100, random_state=42, verbose=3)
gmm.fit(train_x)

train_predictions = gmm.predict(train_x)
test_predictions = gmm.predict(test_x)

train_score = f1_score(train_y, train_predictions, average="macro")
test_score = f1_score(test_y, test_predictions, average="macro")

nb_errors = (test_predictions != test_y).sum()
nb_errors_train = (train_predictions != train_y).sum()

print('F1 score:', test_score, 'Train F1 score:', train_score, 'Nb errors:', nb_errors / len(test_y), 'Nb errors train:', nb_errors_train / len(train_y))

component_means = gmm.means_
component_covariances = gmm.covariances_
