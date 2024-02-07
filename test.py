from data import get_train_data

train_x, train_y, test_x, test_y = get_train_data(["urban_type_Rural"], n_data=10)

print(train_x)