from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import time

from data import get_train_data
from features import DATE_FEATURES, COLOR_FEATURES, CHANGE_STATUS_DATE_FEATURES

t0 = time.time()

train_x, train_y, val_x, val_y = get_train_data(["duration", "area", "perimeter"], n_data=-1)

pipeline = make_pipeline(PCA(n_components=2), KNeighborsClassifier(n_neighbors=5))

pipeline.fit(train_x, train_y)

pca_components = pipeline.named_steps['pca'].components_

pred_y = pipeline.predict(val_x)

score = f1_score(val_y, pred_y, average="weighted")

print('score = ', score)
print('pca_components = ', pca_components)
print('time = ', time.time() - t0)