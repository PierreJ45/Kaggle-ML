import pandas as pd


def create_result_file(predictions, file_name='result.csv'):
    pred_df = pd.DataFrame(predictions, columns=['change_type'])
    pred_df.to_csv(file_name, index=True, index_label='Id')