from data import get_train_data
from features import URBAN_FEATURES, GEOGRAPHY_FEATURES
from sklearn.tree import DecisionTreeClassifier
#random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

train_x, train_y, test_x, test_y = get_train_data(["duration", "area", "perimeter"] + URBAN_FEATURES + GEOGRAPHY_FEATURES, n_data=-1)

accuracies = []
for i in tqdm(range(100)):
    #max_depth=11, min_samples_leaf=1
    #140: 0.5
    # model = DecisionTreeClassifier(min_samples_leaf=80 , max_depth=14)
    model = RandomForestClassifier(n_estimators=i+1)
    model.fit(train_x, train_y)

    predictions = model.predict(test_x)
    accuracy = f1_score(test_y, predictions, average="weighted")
    accuracies.append(accuracy)

plt.plot(accuracies)
plt.show()

# from sklearn.tree import export_text
# tree_rules = export_text(model, feature_names=["duration", "area", "perimeter"])
# print(tree_rules)