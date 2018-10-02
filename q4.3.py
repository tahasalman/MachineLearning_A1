import random
import numpy as np
import matplotlib.pyplot as plt
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


def get_best_fit_l2(x_matrix,y_matrix,lambda_val):
    x_matrix = np.array(x_matrix)
    y_matrix = np.array(y_matrix)

    output = np.matmul(x_matrix.transpose(),x_matrix)
    output = output + (lambda_val*np.identity(len(x_matrix[0])))
    output = np.linalg.inv(output)
    output = np.matmul(np.matmul(output,x_matrix.transpose()),y_matrix)
    return output


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


def divide_interval(interval,num_divisions,end_point_inclusive=True):
    '''
    This method takes in a tuple that represents an interval and returns a list with the given number
    of divisions. If the variable end_point_inclusive is set to true then the returned list has
    the end_point_included
    '''
    output = [interval[0]]
    if interval[0] < interval[1]:
        delta = (interval[1] - interval[0])/num_divisions
        point = interval[0] + delta
        while point < interval[1]:
            output.append(point)
            point+=delta

    if end_point_inclusive:
        output.append(interval[1])

    return output


def find_min(dict_in):
    '''
    Takes a dictionary as input and returns the key and val with the smallest value
    '''
    if len(dict_in) < 1:
        return None

    min_key = list(dict_in.keys())[0]
    min_value = dict_in[min_key]
    output = [min_key,min_value]

    for key,val in dict_in.items():
        if val < output[1]:
            output[0] = key
            output[1] = val

    return output



def remove_missing_data_cols(data_path,removal_threshold=0.5):
    '''
    This method removes all variables that have instances greater that the removal threshold for missing data
    and updates the rest of the missing data with median.
    '''

    with open(data_path,'r') as f:
        lines=f.readlines()

    missing_data_cols = {}
    for i in range(0,len(lines[0].split(','))):
        missing_data_cols[i] = 0

    for line in lines:
        temp_list = line.split(',')
        for i in range(0,len(temp_list)):
            if temp_list[i] == "":
                missing_data_cols[i]+=1

    total_features = len(lines)
    removal_threshold = removal_threshold*total_features
    output = ""

    for line in lines:
        temp_list=line.split(',')
        formatted_list = []
        for i in range(0,len(temp_list)):
            if missing_data_cols[i] < removal_threshold:
                formatted_list.append(temp_list[i])
        output+= (',').join(formatted_list)

    return output

def remove_missing_data_rows(data_path):
    '''
    This method removes all rows that have any missing data
    '''
    output = ""
    with open(data_path,'r') as f:
        lines=f.readlines()

    for line in lines:
        add_to_output = True
        temp_list=line.split(',')
        for i in range(0,len(temp_list)):
            if temp_list[i] == "":
                add_to_output=False
                break
        if add_to_output:
            output+=line

    return output


def run_a():
    init_data_path = 'Datasets/CrimeData/crime_data_updated_custom.csv'
    split_data_name = 'Datasets/CrimeData/CandC'
    num_splits = 5
    lambda_divisions = 10

    print("Initializing data from {}".format(init_data_path))
    set_initial_data(init_data_path)

    print("Generating {} 80-20 splits with the initial data".format(num_splits))
    generate_80_20_splits(num_splits, saving_name=split_data_name)

    lambda_values = divide_interval([0, 1], lambda_divisions, True)
    mse_dict = {}
    for l_val in lambda_values:
        total_mse = 0
        for i in range(1, num_splits + 1):
            training_data = []
            testing_data = []

            with open("{}-train{}.csv".format(split_data_name, i), 'r') as f:
                training_data = f.readlines()

            with open("{}-test{}.csv".format(split_data_name, i), 'r') as f:
                testing_data = f.readlines()

            split_training_data = split_x_y(training_data)
            split_testing_data = split_x_y(testing_data)

            training_data_x = extract_x_from_split_data(split_training_data)
            training_data_y = extract_y_from_split_data(split_training_data)
            testing_data_x = extract_x_from_split_data(split_testing_data)
            testing_data_y = extract_y_from_split_data(split_testing_data)

            best_fit_coefficients = get_best_fit_l2(training_data_x, training_data_y, l_val)
            predictions = get_predictions(best_fit_coefficients, testing_data_x)
            mse = get_mean_squared_error(testing_data_y, predictions)

            total_mse += mse

        avg_mse = total_mse / num_splits
        mse_dict[l_val] = avg_mse

    min_mse = find_min(mse_dict)
    print("The smallest MSE {} was found for {}".format(min_mse[1], min_mse[0]))
    plt.plot(mse_dict.keys(), mse_dict.values(), 'b-')
    plt.xlabel("Lambda Values")
    plt.ylabel("MSE Values")
    plt.axis([0, 1, 0, 0.1])
    plt.suptitle("M.S.E. Values for Different Lambdas")
    plt.show()

def run_b():
    removal_threshold = 0.2
    init_data_path = 'Datasets/CrimeData/crime_data_updated_custom.csv'
    reduced_data_path = 'Datasets/CrimeData/crime_data_feature_reduced.csv'
    split_data_name = 'Datasets/CrimeData/CandC'
    num_splits = 5

    print("Removing data columns with missing data over {} of the total data".format(removal_threshold))
    refined_data = remove_missing_data_cols(data_path=init_data_path,removal_threshold=removal_threshold)
    with open(reduced_data_path,'w') as f:
        f.write(refined_data)
    print("Removing all rows with missing data")
    refined_data = remove_missing_data_rows(reduced_data_path)
    with open(reduced_data_path,'w') as f:
        f.write(refined_data)
    print("Updated data saved to {}".format(reduced_data_path))

    print("Initializing data from {}".format(reduced_data_path))
    set_initial_data(reduced_data_path)
    print("Generating {} 80-20 splits with the initial data".format(num_splits))
    generate_80_20_splits(num_splits, saving_name=split_data_name)

    lambda_values = divide_interval([0, 1], lambda_divisions, True)
    mse_dict = {}
    for l_val in lambda_values:
        total_mse = 0
        for i in range(1, num_splits + 1):
            training_data = []
            testing_data = []

            with open("{}-train{}.csv".format(split_data_name, i), 'r') as f:
                training_data = f.readlines()

            with open("{}-test{}.csv".format(split_data_name, i), 'r') as f:
                testing_data = f.readlines()

            split_training_data = split_x_y(training_data)
            split_testing_data = split_x_y(testing_data)

            training_data_x = extract_x_from_split_data(split_training_data)
            training_data_y = extract_y_from_split_data(split_training_data)
            testing_data_x = extract_x_from_split_data(split_testing_data)
            testing_data_y = extract_y_from_split_data(split_testing_data)

            best_fit_coefficients = get_best_fit_l2(training_data_x, training_data_y, l_val)
            predictions = get_predictions(best_fit_coefficients, testing_data_x)
            mse = get_mean_squared_error(testing_data_y, predictions)

            total_mse += mse

        avg_mse = total_mse / num_splits
        mse_dict[l_val] = avg_mse

    min_mse = find_min(mse_dict)
    print("The smallest MSE {} was found for {}".format(min_mse[1], min_mse[0]))
    plt.plot(mse_dict.keys(), mse_dict.values(), 'b-')
    plt.xlabel("Lambda Values")
    plt.ylabel("MSE Values")
    plt.axis([0, 1, 0, 0.1])
    plt.suptitle("M.S.E. Values for Different Lambdas")
    plt.show()


if __name__ == "__main__":
    run_a()