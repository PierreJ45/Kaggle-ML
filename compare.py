from data import get_train_data, get_test_data

train_x, train_y, val_x, val_y = get_train_data(
    n_data = -1,
    val_size = 0.2,
    file_name = "data/train.geojson"
)

test_x = get_test_data()

print(type(train_x), type(train_y), type(val_x), type(val_y), type(test_x))
print(train_x.shape, train_y.shape, val_x.shape, val_y.shape, test_x.shape)