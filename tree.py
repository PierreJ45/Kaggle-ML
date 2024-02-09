from data import get_train_data
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


train_x, train_y, test_x, test_y = get_train_data(
    ["duration", "area", "perimeter"] + URBAN_FEATURES + GEOGRAPHY_FEATURES + COLOR_FEATURES,
    n_data = None
)

param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(1, 50),
    'min_samples_split': randint(2, 10),
}

rand_search = RandomizedSearchCV(
    RandomForestClassifier(),
    param_distributions = param_dist, 
    n_iter = 5,
    cv = 2,
    verbose = 3
)

rand_search.fit(train_x, train_y)

best_rf = rand_search.best_estimator_
print('Best hyperparameters:',  rand_search.best_params_)

score = f1_score(test_y, best_rf.predict(test_x), average="weighted")
print('F1 score:', score)

feature_importances = pd.Series(best_rf.feature_importances_, index=train_x.columns).sort_values(ascending=False)
print(feature_importances)