from functools import partial
from datetime import datetime

CLASSES = ['Demolition', 'Road', 'Residential', 'Commercial', 'Industrial', 'Mega Projects']
NB_CLASSES = len(CLASSES)
NB_DATES = 5
COLORS = ["red", "green", "blue"]
COLOR_FEATURES = []
DATE_FEATURES = []
CHANGE_STATUS_DATE_FEATURES = []
URBAN_TYPES = ['Dense Urban', 'Sparse Urban', 'Industrial', 'N,A', 'Rural', 'Urban Slum']
URBAN_FEATURES = ["urban_type_" + urban_type for urban_type in URBAN_TYPES]
GEOGRAPHY_TYPES = ['N,A', 'Barren Land', 'Hills', 'Dense Forest', 'Desert', 'Sparse Forest', 'River', 'Grass Land', 'Farms', 'Coastal', 'Snow', 'Lakes']
GEOGRAPHY_FEATURES = ["geography_type_" + geography_type for geography_type in GEOGRAPHY_TYPES]

def get_date(row, i) -> float:
    year_one = datetime.strptime("01-01-2000", "%d-%m-%Y")
    
    if row[f"date{i}"] is None:
        dates = ["01-08-2018", "09-12-2013", "10-09-2016", "22-07-2019", "24-07-2017"]
        date = datetime.strptime(dates[i], "%d-%m-%Y")
    else:
        date = datetime.strptime(row[f"date{i}"], "%d-%m-%Y")
    
    return (date - year_one).days


def is_type(row, base_type, type_name) -> float:
    if type_name in row[base_type]:
        return 1.0
    return 0.0



def null_color(row, color, value, date) -> float:
    number = row[f"img_{color}_{value}_date{date}"]
    
    if number != number: # is NaN
        return 128.0
    return number


base_features_func = {
    "geometry": None,
    "urban_type": None,
    "geography_type": None,
    "index": None,
}

for i in range(NB_DATES):
    base_features_func[f"date{i}"] = partial(get_date, i=i)
    base_features_func[f"change_status_date{i}"] = None
    DATE_FEATURES.append(f"date{i}")
    CHANGE_STATUS_DATE_FEATURES.append(f"change_status_date{i}")
    
    for color in COLORS:
        for value in ["mean", "std"]:
            base_features_func[f"img_{color}_{value}_date{i + 1}"] = partial(null_color, color=color, value=value, date=i + 1)
            COLOR_FEATURES.append(f"img_{color}_{value}_date{i + 1}")


def get_area(row) -> float:
    return row["geometry"].area


def get_perimeter(row) -> float:
    return row["geometry"].length


def get_construction_dates_idx(row) -> tuple:
    dates = [row[f"date{i}"] for i in range(NB_DATES)]
    change_status_dates = [row[f"change_status_date{i}"] for i in range(NB_DATES)]

    start_dates = [date for i, date in enumerate(dates) if change_status_dates[i] == "Greenland"]
    start = min(dates) if len(start_dates) == 0 else max(start_dates)
    
    end_dates = [date for i, date in enumerate(dates) if change_status_dates[i] == "Construction Done"]
    end = max(dates) if len(end_dates) == 0 else min(end_dates)
    
    return dates.index(start), dates.index(end)


def get_duration(row) -> float:
    start, end = get_construction_dates_idx(row)
    return row[f"date{end}"] - row[f"date{start}"]


def get_color_start(row, color, value) -> float:
    start, _ = get_construction_dates_idx(row)
    return row[f"img_{color}_{value}_date{start + 1}"]


def get_color_end(row, color, value) -> float:
    _, end = get_construction_dates_idx(row)
    return row[f"img_{color}_{value}_date{end + 1}"]


def get_elongation(row) -> float:
    area = get_area(row)
    if area == 0:
        return 0.0
    return get_perimeter(row)**2 / get_area(row)


def get_nb_points(row) -> float:
    return len(row["geometry"].exterior.coords)


base_features = list(base_features_func.keys())

other_features_func = {
    "area": get_area,
    "perimeter": get_perimeter,
    "duration": get_duration,
    "elongation": get_elongation,
    "nb_points": get_nb_points,
}

for urban_feature, urban_type in zip(URBAN_FEATURES, URBAN_TYPES):
    other_features_func[urban_feature] = partial(is_type, base_type="urban_type", type_name=urban_type)
for geography_feature, geography_type in zip(GEOGRAPHY_FEATURES, GEOGRAPHY_TYPES):
    other_features_func[geography_feature] = partial(is_type, base_type="geography_type", type_name=geography_type)

for color in COLORS:
    for value in ["mean", "std"]:
        other_features_func[f"{color}_{value}_start"] = partial(get_color_start, color=color, value=value)
        other_features_func[f"{color}_{value}_end"] = partial(get_color_end, color=color, value=value)

start_color_features = [f"{color}_{value}_start" for color in COLORS for value in ["mean", "std"]]
end_color_features = [f"{color}_{value}_end" for color in COLORS for value in ["mean", "std"]]

other_features = list(other_features_func.keys())

all_features = base_features + other_features