from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

from data import get_train_data


train_x, train_y, test_x, test_y = get_train_data(["duration", "area", "perimeter"], n_data=-1)

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(train_x, train_y)

pred_y = neigh.predict(test_x)
score = f1_score(test_y, pred_y, average="weighted")

print(score)