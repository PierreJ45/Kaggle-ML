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
    ["duration", "area", "perimeter", "elongation", "nb_points"] + start_color_features + end_color_features,
    n_data = -1,
    val_size = 0.1
)

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

best_rf = RandomForestClassifier(n_estimators=100, max_depth=50, min_samples_split=5, min_samples_leaf=1, random_state=42)
best_rf.fit(train_x, train_y)

print(best_rf.predict(test_x)[:5])

score = f1_score(test_y, best_rf.predict(test_x), labels=range(NB_CLASSES), average="weighted")
train_score = f1_score(train_y, best_rf.predict(train_x), labels=range(NB_CLASSES), average="weighted")
nb_errors = (best_rf.predict(test_x) != test_y).sum()
nb_errors_train = (best_rf.predict(train_x) != train_y).sum()

print('F1 score:', score, 'Train F1 score:', train_score, 'Nb errors:', nb_errors / len(test_y), 'Nb errors train:', nb_errors_train / len(train_y))

feature_importances = pd.Series(best_rf.feature_importances_, index=train_x.columns).sort_values(ascending=False)
print(feature_importances)

