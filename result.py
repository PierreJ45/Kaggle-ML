def create_result_file(predictions):
    with open('result.csv', 'w') as f:
        f.write('Id,change_type\n')
        for i in range(len(predictions)):
            f.write(f'{i},{predictions[i]}\n')