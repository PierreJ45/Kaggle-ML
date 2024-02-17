from data import get_train_data, get_test_data
from features import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from result import *


# features = ["duration", "area", "perimeter", "elongation"] + start_color_features + end_color_features
features = ["duration", "area", "perimeter", "elongation", "nb_points"] + delta_color_features + start_color_features + end_color_features + URBAN_FEATURES + GEOGRAPHY_FEATURES + COLOR_FEATURES

train_x, train_y, val_x, val_y, normalized_coeffs = get_train_data(
    features,
    n_data = -1,
    val_size = 0.0,
    file_name = "data/train.geojson",
    same_coeffs = True
)

# test_x = get_test_data(features, normalized_coeffs)

# param_dist = {
#     'pca__n_components': randint(8, 12),    
#     'randomforestclassifier__n_estimators': randint(250, 300),
#     'randomforestclassifier__max_depth': randint(40, 60),
#     'randomforestclassifier__min_samples_split': randint(3, 7),
#     'randomforestclassifier__min_samples_leaf': randint(1, 3),
# }

# rand_search = RandomizedSearchCV(
#     make_pipeline(PCA(), RandomForestClassifier(class_weight='balanced')),
#     param_distributions = param_dist, 
#     n_iter = 20,
#     cv = 5,
#     verbose = 3,
#     scoring = make_scorer(f1_score, average='macro'),
# )

# rand_search.fit(train_x, train_y)

# rf = rand_search.best_estimator_
# print('Best hyperparameters:',  rand_search.best_params_)

# rf = make_pipeline(PCA(n_components=11), RandomForestClassifier(n_estimators=300, max_depth=59, min_samples_split=3, min_samples_leaf=1, random_state=42, class_weight='balanced'))
rf = RandomForestClassifier(n_estimators=300, max_depth=59, min_samples_split=3, min_samples_leaf=1, random_state=42, class_weight='balanced')
rf.fit(train_x, train_y)

# print_score(rf, train_x, train_y, val_x, val_y)

feature_importances = pd.Series(rf.feature_importances_, index=train_x.columns).sort_values(ascending=False)
# print(feature_importances)
for feature in feature_importances.index:
    print(feature, feature_importances[feature])

# create_result_file(rf.predict(test_x), 'rf.csv')