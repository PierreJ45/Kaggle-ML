from data import get_train_data, get_test_data
from features import *
from sklearn.tree import DecisionTreeClassifier
#random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import time
from result import create_result_file


features = ["duration", "area", "perimeter", "elongation"] + start_color_features + end_color_features

train_x, train_y, val_x, val_y = get_train_data(
    ["duration", "area", "perimeter", "elongation"] + start_color_features + end_color_features,
    n_data = -1,
    val_size = 0.2,
    file_name = "data/train.geojson"
)

test_x = get_test_data(["duration", "area", "perimeter", "elongation"] + start_color_features + end_color_features)

# param_dist = {
#     'n_estimators': randint(50, 100),
#     'max_depth': randint(1, 50),
#     'min_samples_split': randint(5, 100),
#     'min_samples_leaf': randint(1, 100),
# }

# rand_search = RandomizedSearchCV(
#     RandomForestClassifier(),
#     param_distributions = param_dist, 
#     n_iter = 20,
#     cv = 5,
#     verbose = 3
# )

# rand_search.fit(train_x, train_y)

# best_rf = rand_search.best_estimator_
# print('Best hyperparameters:',  rand_search.best_params_)

t0 = time.time()

pipeline = make_pipeline(PCA(n_components=10), RandomForestClassifier(n_estimators=300, max_depth=50, min_samples_split=5, min_samples_leaf=1, random_state=42))

pipeline.fit(train_x, train_y)

#print(pipeline.predict(val_x)[:5])

score = f1_score(val_y, pipeline.predict(val_x), labels=range(NB_CLASSES), average="macro")
train_score = f1_score(train_y, pipeline.predict(train_x), labels=range(NB_CLASSES), average="macro")
nb_errors = (pipeline.predict(val_x) != val_y).sum()
nb_errors_train = (pipeline.predict(train_x) != train_y).sum()

print('F1 score:', score, 'Train F1 score:', train_score, 'Nb errors:', nb_errors / len(val_y), 'Nb errors train:', nb_errors_train / len(train_y))
print('DataLessTime:', time.time() - t0)

#feature_importances = pd.Series(pipeline.feature_importances_, index=train_x.columns).sort_values(ascending=False)
#print(feature_importances)

create_result_file(pipeline.predict(test_x), 'result.csv')