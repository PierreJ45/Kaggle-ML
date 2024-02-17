from data import get_train_data, get_test_data
from result import create_result_file
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from features import *
import matplotlib.pyplot as plt

features = ["duration", "area", "perimeter", "elongation"] + start_color_features + end_color_features

train_x, train_y, val_x, val_y, normalized_coeffs = get_train_data(features, n_data=-1, val_size=0.2, file_name="data/train.geojson")
test_x = get_test_data(features, normalized_coeffs)

scores = []
for n_neighbors in range(1, 20):
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
    neigh.fit(train_x, train_y)
    scores.append(f1_score(val_y, neigh.predict(val_x), average="macro"))

plt.plot(range(1, 20), scores)
plt.show()

best_n_neighbors = scores.index(max(scores)) + 1

neigh = KNeighborsClassifier(n_neighbors=best_n_neighbors)
neigh.fit(train_x, train_y)

create_result_file(neigh.predict(test_x), 'knn.csv')