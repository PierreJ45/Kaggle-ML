from data import get_train_data
from pandas import concat

train_x, train_y, test_x, test_y = get_train_data(["geography_type"], n_data=-1)
x = concat([train_x, test_x])
y = concat([train_y, test_y])

x["geography_type"] = x["geography_type"].apply(lambda x: x.split(","))
geography_types = set()
for i in x["geography_type"]:
    geography_types.update(i)

print(geography_types)