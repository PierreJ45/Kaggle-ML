import geopandas as gpd
import numpy as np
from sklearn.model_selection import train_test_split
from features import all_features, base_features, other_features, base_features_func, other_features_func
from typing import List


change_type_id = {"Demolition": 0, "Road": 1, "Residential": 2, "Commercial": 3, "Industrial": 4, "Mega Projects": 5}

def normalize(x):
    for feature in x.columns:
        if x[feature].dtype == np.float64 or x[feature].dtype == np.int64:
            std = x[feature].std()
            
            if std != 0:
                x[feature] = (x[feature] - x[feature].mean()) / std
            else:
                x[feature] = 1.0


def get_train_data(features: List[str] = all_features, n_data=-1, val_size=0.2):
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
    
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=val_size, random_state=42)

    normalize(train_x)
    normalize(val_x)
    
    return train_x, train_y, val_x, val_y


def get_test_data(features=all_features):
    print("Reading test csvs...")
    test_df = gpd.read_file("data/test.geojson", index_col=0)

    for feature in features:
        if feature not in all_features:
            raise ValueError(f"Feature {feature} is not a valid feature (you may want to implement it in features.py)")
    
    used_base_features = [feature for feature in features if feature in base_features]
    used_other_features = [feature for feature in features if feature in other_features]
    
    test_x = test_df[used_base_features]
    
    for feature in base_features:
        if base_features_func[feature] is not None:
            test_df[feature] = test_df.apply(base_features_func[feature], axis=1)
        
        if feature in used_base_features: #in case it was changed previous if ?
            test_x[feature] = test_df[feature] 
    
    for feature in used_other_features:
        test_x[feature] = test_df.apply(other_features_func[feature], axis=1)

    normalize(test_x)
    
    return test_x


if __name__ == "__main__":
    train_x, train_y, val_x, val_y = get_train_data(["area", "duration", "date0"], 100)
    print(train_x.head())
    print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',train_x['area'].mean())
    print('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',train_x['area'].std())
    print('cccccccccccccccccccccccccccccccccccccc',train_x['duration'].mean())
    print('dddddddddddddddddddddddddddddddddddddd',train_x['duration'].std())
    print('eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee',train_x['date0'].mean())
    print('ffffffffffffffffffffffffffffffffffffff',train_x['date0'].std())
    print('11111111111111111111111111111111111111',train_x['area'].dtype)
    print('22222222222222222222222222222222222222',train_x['duration'].dtype)
    print('33333333333333333333333333333333333333',train_x['date0'].dtype)
    print(val_x.head())
