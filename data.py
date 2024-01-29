import geopandas as gpd
import numpy as np
from sklearn.model_selection import train_test_split
from features import all_features, base_features, other_features, base_features_func, other_features_func
from typing import List


change_type_id = {"Demolition": 0, "Road": 1, "Residential": 2, "Commercial": 3, "Industrial": 4, "Mega Projects": 5}

def normalize(x):
    for feature in x.columns:
        if x[feature].dtype == np.float64:
            x[feature] = (x[feature] - x[feature].mean()) / x[feature].std()


def get_train_data(features: List[str] = all_features, n_data=-1, test_size=0.2):
    print("Reading train csvs...")
    train_df: gpd.GeoDataFrame = gpd.read_file("data/train.geojson", index_col=0, rows=n_data)
    
    for feature in features:
        if feature not in all_features:
            raise ValueError(f"Feature {feature} is not a valid feature (you may want to implement it in features.py)")
    
    used_base_features = [feature for feature in features if feature in base_features]
    used_other_features = [feature for feature in features if feature in other_features]
    
    train_x = train_df[used_base_features]
    
    for feature in base_features:
        if base_features_func[feature] is not None:
            train_df[feature] = train_df.apply(base_features_func[feature], axis=1)
        
        if feature in used_base_features:
            train_x[feature] = train_df[feature]
    
    for feature in used_other_features:
        train_x[feature] = train_df.apply(other_features_func[feature], axis=1)
    
    train_y = train_df["change_type"].apply(change_type_id.get)
    
    # train_x = (train_x - train_x.mean()) / train_x.std()
    
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=test_size, random_state=42)
    
    return train_x, train_y, test_x, test_y


def get_test_data(features=all_features):
    print("Reading test csvs...")
    test_df = gpd.read_file("data/test.geojson", index_col=0)
    
    test_x = test_df[features]
    
    test_x = (test_x - test_x.mean()) / test_x.std()
    
    return test_x


if __name__ == "__main__":
    train_x, train_y, test_x, test_y = get_train_data(["area", "duration", "date0"], 100)
    print(train_x)