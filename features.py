from functools import partial
from datetime import datetime

NB_DATES = 5
COLORS = ["red", "green", "blue"]
COLOR_FEATURES = []
DATE_FEATURES = []
CHANGE_STATUS_DATE_FEATURES = []


def get_date(row, i) -> float:
    year_one = datetime.strptime("01-01-2000", "%d-%m-%Y")
    
    if row[f"date{i}"] is None:
        dates = ["01-08-2018", "09-12-2013", "10-09-2016", "22-07-2019", "24-07-2017"]
        date = datetime.strptime(dates[i], "%d-%m-%Y")
    else:
        date = datetime.strptime(row[f"date{i}"], "%d-%m-%Y")
    
    return (date - year_one).days


base_features_func = {
    "geometry": None,
    "urban_type": None,
    "geography_type": None,
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
            base_features_func[f"img_{color}_{value}_date{i + 1}"] = None
            COLOR_FEATURES.append(f"img_{color}_{value}_date{i + 1}")


def get_area(row) -> float:
    return row["geometry"].area


def get_perimeter(row) -> float:
    return row["geometry"].length


def get_duration(row) -> float:
    dates = [row[f"date{i}"] for i in range(NB_DATES)]
    change_status_dates = [row[f"change_status_date{i}"] for i in range(NB_DATES)]
    
    start_dates = [date for i, date in enumerate(dates) if change_status_dates[i] == "Greenland"]
    start = min(dates) if len(start_dates) == 0 else max(start_dates)
    
    end_dates = [date for i, date in enumerate(dates) if change_status_dates[i] == "Construction Done"]
    end = max(dates) if len(end_dates) == 0 else min(end_dates)
    
    return end - start


base_features = list(base_features_func.keys())

other_features_func = {
    "area": get_area,
    "perimeter": get_perimeter,
    "duration": get_duration,
}
other_features = list(other_features_func.keys())

all_features = base_features + other_features