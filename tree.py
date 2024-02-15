from data import get_train_data, get_test_data
from features import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from result import *


features = ["duration", "area", "perimeter", "elongation"] + start_color_features + end_color_features

train_x, train_y, val_x, val_y, normalized_coeffs = get_train_data(
    ["duration", "area", "perimeter", "elongation"] + start_color_features + end_color_features,
    n_data = -1,
    val_size = 0.2,
    file_name = "data/train.geojson"
)

test_x = get_test_data(features, normalized_coeffs)

param_dist = {
    'pca__n_components': randint(1, 20),    
    'randomforestclassifier__n_estimators': randint(50, 300),
    'randomforestclassifier__max_depth': randint(1, 50),
    'randomforestclassifier__min_samples_split': randint(5, 100),
    'randomforestclassifier__min_samples_leaf': randint(1, 100),
}

rand_search = RandomizedSearchCV(
    make_pipeline(PCA(), RandomForestClassifier()),
    param_distributions = param_dist, 
    n_iter = 20,
    cv = 5,
    verbose = 3,
    scoring = make_scorer(f1_score, average='macro'),
)

rand_search.fit(train_x, train_y)

best_rf = rand_search.best_estimator_
print('Best hyperparameters:',  rand_search.best_params_)

# pipeline = make_pipeline(PCA(n_components=10), RandomForestClassifier(n_estimators=300, max_depth=50, min_samples_split=5, min_samples_leaf=1, random_state=42))
# pipeline.fit(train_x, train_y)

print_score(best_rf, train_x, train_y, val_x, val_y)

#feature_importances = pd.Series(pipeline.feature_importances_, index=train_x.columns).sort_values(ascending=False)
#print(feature_importances)

create_result_file(best_rf.predict(test_x), 'rf.csv')