from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

from data import get_train_data
from features import DATE_FEATURES, COLOR_FEATURES, CHANGE_STATUS_DATE_FEATURES


train_x, train_y, val_x, val_y = get_train_data(["duration", "area", "perimeter"], n_data=-1)

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(train_x, train_y)

pred_y = neigh.predict(val_x)
score = f1_score(val_y, pred_y, average="weighted")

print('score = 'score)