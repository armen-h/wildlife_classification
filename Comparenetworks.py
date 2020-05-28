import pandas as pd

project_root = '/home/homberge/Projet/'


def compare_networks():
    folder_list = []
    user_input = 'input'
    while user_input != 'done':
        print('Enter a folder name for network comparison (enter \'done\' to terminate): ')
        user_input = str(input())
        if user_input != 'done':
            folder_list.append(user_input)

    result = pd.DataFrame()

    for folder in folder_list:
        csv_train = project_root + 'results/' + folder + '/train/' + 'raw_data.csv'
        csv_validation = project_root + 'results/' + folder + '/validation/' + 'raw_data.csv'
        csv_test = project_root + 'results/' + folder + '/test/' + 'raw_data.csv'

        df_train = pd.read_csv(csv_train)
        df_validation = pd.read_csv(csv_validation)
        df_test = pd.read_csv(csv_test)

        labels = pd.DataFrame([['Train'], ['Validation'], ['Test']])
        folder_results = pd.concat([df_train.tail(1), df_validation.tail(1), df_test.tail(1)], ignore_index=True)
        folder_results = pd.concat([labels, folder_results], axis=1, ignore_index=True)

        result = pd.concat([result, folder_results])

    result_headers = ['Phase', 'Network', 'Epoch', 'Examples seen', 'CrossEntropy loss',
                          'Average CrossEntropy loss over epoch',
                          'Precision', 'Recall', 'F-measure', 'Seed']

    result.to_csv('/home/homberge/Projet/results/Network_comparison.csv', mode='w', header=result_headers,
                      index=None)


compare_networks()
