from tqdm import tqdm

def create_result_file(predictions, indexes: list, file_name='result.csv'):
    with open(file_name, 'w') as f:
        f.write('Id,change_type\n')
        
        for i in tqdm(range(len(indexes))):
            index = indexes.index(i)
            f.write(f"{i},{predictions[index]}\n")