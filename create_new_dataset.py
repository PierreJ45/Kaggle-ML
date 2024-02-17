import geopandas as gpd
from random import random
from features import *
from tqdm import tqdm
from shapely.geometry import Polygon
import pandas as pd

COLOR_MEAN_DELTA = 5.0
COLOR_STD_DELTA = 5.0
COORDINATE_DELTA = 1e-5
MAX_DATA_TO_ADD_PER_CLASS = -1


def get_random_float(max) -> float:
    return (2*random() - 1)*max

print("Reading train csvs...")
train_df: gpd.GeoDataFrame = gpd.read_file("data/train.geojson", engine='pyogrio')

max_data = train_df["change_type"].value_counts().max()

for change_type in CLASSES:
    change_type_data = train_df[train_df["change_type"] == change_type]
    
    if len(change_type_data) == 0:
        print(f"Class {change_type} has no data, skipping...")
        continue
    
    n_data = max_data - len(change_type_data)
    if MAX_DATA_TO_ADD_PER_CLASS > 0:
        n_data = min(n_data, MAX_DATA_TO_ADD_PER_CLASS)
    
    print(f"Adding {n_data} data for class {change_type}...")
    
    new_rows = pd.DataFrame()
    
    for j in tqdm(range(n_data)):
        new_row = change_type_data.sample(1).copy()
        
        polygon = new_row["geometry"].iloc[0]
        new_row["geometry"] = Polygon([(x + get_random_float(COORDINATE_DELTA), y + get_random_float(COORDINATE_DELTA)) for x, y in polygon.exterior.coords])

        for date in range(NB_DATES):
            for color in COLORS:
                new_row[f"img_{color}_mean_date{date + 1}"] += get_random_float(COLOR_MEAN_DELTA)
                new_row[f"img_{color}_std_date{date + 1}"] += get_random_float(COLOR_STD_DELTA)
        
        new_row["index"] = len(train_df)
        
        new_rows = pd.concat([new_rows, new_row])
        if j % 1000 == 0:
            train_df = pd.concat([train_df, new_rows])
            new_rows = pd.DataFrame()
    
    train_df = pd.concat([train_df, new_rows])


print("Writing balanced_train.geojson...")
train_df.to_file("data/balanced_train.geojson", driver='GeoJSON')