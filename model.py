from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import time

from data import get_train_data
from features import DATE_FEATURES, COLOR_FEATURES, CHANGE_STATUS_DATE_FEATURES, GEOGRAPHY_FEATURES, URBAN_FEATURES

t0 = time.time()

train_x, train_y, val_x, val_y, _ = get_train_data(["duration", "area", "perimeter", "elongation"] + COLOR_FEATURES, n_data=-1)

pipeline = make_pipeline(PCA(n_components=2), KNeighborsClassifier(n_neighbors=5))

pipeline.fit(train_x, train_y)

#pca_components = pipeline.named_steps['pca'].components_

pred_y = pipeline.predict(val_x)
train_pred_y = pipeline.predict(train_x)

scoreTrain = f1_score(train_y, train_pred_y, average="macro")
scoreTest = f1_score(val_y, pred_y, average="macro")

print('scoreTrain = ', scoreTrain)
print('scoreTest = ', scoreTest)
#print('pca_components = ', pca_components)
print('time = ', time.time() - t0)