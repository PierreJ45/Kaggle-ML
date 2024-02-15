from data import get_train_data, get_test_data
from result import create_result_file
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

train_x, train_y, val_x, val_y, normalized_coeffs = get_train_data(['area'], n_data=-1, val_size=0.0, file_name="data/train.geojson")
test_x = get_test_data(['area'], normalized_coeffs)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_x, train_y)

score = f1_score(val_y, neigh.predict(val_x), average="macro")
train_score = f1_score(train_y, neigh.predict(train_x), average="macro")

print()

create_result_file(neigh.predict(test_x), 'knn_submission_verif.csv')