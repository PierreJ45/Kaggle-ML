from data import get_train_data
from features import GEOGRAPHY_FEATURES

train_x, train_y, test_x, test_y = get_train_data(features = GEOGRAPHY_FEATURES[1:4])

print(train_x.head())