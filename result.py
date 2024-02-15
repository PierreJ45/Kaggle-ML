import pandas as pd
from sklearn.metrics import f1_score


def create_result_file(predictions, file_name='result.csv'):
    pred_df = pd.DataFrame(predictions, columns=['change_type'])
    pred_df.to_csv(file_name, index=True, index_label='Id')


def print_score(model, train_x, train_y, val_x, val_y):
    if val_x is not None:
        print('F1 score:', f1_score(val_y, model.predict(val_x), average="macro"))
    
    print('Train F1 score:', f1_score(train_y, model.predict(train_x), average="macro"))
    
    if val_x is None:
        print('Nb errors:', (model.predict(val_x) != val_y).sum() / len(val_y))
    
    print('Nb errors train:', (model.predict(train_x) != train_y).sum() / len(train_y))