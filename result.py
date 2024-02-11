def create_result_file(predictions, file_name='result.csv'):
    with open(file_name, 'w') as f:
        f.write('Id,change_type\n')
        for i in range(len(predictions)):
            f.write(f'{i},{predictions[i]}\n')