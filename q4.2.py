import random
import numpy as np
from copy import deepcopy

INITIAL_DATA = None

def set_initial_data(data_path):
    global INITIAL_DATA
    with open(data_path,'r') as f:
        INITIAL_DATA = f.readlines()

def generate_80_20_splits(num_splits,saving_name='CandC'):
    if not INITIAL_DATA:
        print("Please set initial data first!")
        return None

    split_20_num = int(len(INITIAL_DATA) * 0.2)

    for i in range(1,num_splits+1):
        temp_data = deepcopy(INITIAL_DATA)
        test_split = ""
        for j in range(0,split_20_num):
            test_selection = random.randint(0,len(temp_data)-1)
            test_split+= temp_data[test_selection]
            temp_data.pop(test_selection)

        train_split = "".join(temp_data)
        with open("{}-train{}.csv".format(saving_name,i),'w') as f:
            f.write(train_split)
        with open("{}-test{}.csv".format(saving_name,i),'w') as f:
            f.write(test_split)

def split_x_y(data):
    '''
    Takes in a list of rows where all columns except last one are xi's
    and splits them into separate vectors
    '''

    split_data = []

    for line in data:
        temp_line = line.split(',')
        y_vector = [float(temp_line[-1])]
        temp_line.pop(-1)
        x_vector = [float(i) for i in temp_line]

        split_data.append((x_vector,y_vector))

    return split_data

def extract_x_from_split_data(data):
    x_matrix = []
    for row in data:
        x_matrix.append(row[0])
    return x_matrix

def extract_y_from_split_data(data):
    y_matrix = []
    for row in data:
        y_matrix.append(row[1])
    return y_matrix


def get_best_fit(x_matrix,y_matrix):
    '''
    Using the initialized data set this method finds the best fit for the data.
    It uses the least squares regression method and returns an array containing
    all the coefficients found with this method.
    '''

    mpps_x = np.linalg.pinv(np.array(x_matrix))

    return np.matmul(mpps_x,np.array(y_matrix))

def get_predictions(coefficients,x_matrix):
    predictions = []
    for i in range(0,len(x_matrix)):
        prediction = 0
        for j in range(0,len(x_matrix[i])):
            prediction += coefficients[j]*x_matrix[i][j]
        predictions.append(prediction)

    return predictions

def get_mean_squared_error(target_values, predictions):
    output = 0
    num_predictions = len(predictions)

    for i in range(0, num_predictions):
        output += ((target_values[i] - predictions[i]) ** 2)

    return (output / num_predictions)[0]


if __name__ == "__main__":
    init_data_path = 'Datasets/CrimeData/crime_data_updated_custom.csv'
    split_data_name = 'Datasets/CrimeData/CandC'
    num_splits = 5

    print("Initializing data from {}".format(init_data_path))
    set_initial_data(init_data_path)

    print("Generating {} 80-20 splits with the initial data".format(num_splits))
    generate_80_20_splits(num_splits,saving_name=split_data_name)

    total_mse = 0
    for i in range(1,num_splits+1):
        training_data = []
        testing_data = []

        with open("{}-train{}.csv".format(split_data_name,i),'r') as f:
            training_data = f.readlines()

        with open("{}-test{}.csv".format(split_data_name,i),'r') as f:
            testing_data = f.readlines()

        split_training_data = split_x_y(training_data)
        split_testing_data = split_x_y(testing_data)

        training_data_x = extract_x_from_split_data(split_training_data)
        training_data_y = extract_y_from_split_data(split_training_data)
        testing_data_x = extract_x_from_split_data(split_testing_data)
        testing_data_y = extract_y_from_split_data(split_testing_data)

        best_fit_coefficients = get_best_fit(training_data_x,training_data_y)
        predictions = get_predictions(best_fit_coefficients,testing_data_x)
        mse = get_mean_squared_error(testing_data_y,predictions)

        total_mse += mse

        print("The MSE for Dataset {} is {}".format(i,mse))

    print("The average MSE over these {} Datasets is {}".format(num_splits,total_mse/num_splits))



