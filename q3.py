import matplotlib.pyplot as plt
import numpy as np
from random import random

TRAINING_DATA = None
VALIDATION_DATA = None
TEST_DATA = None

import matplotlib.pyplot as plt
import numpy as np

TRAINING_DATA = None
VALIDATION_DATA = None
TEST_DATA = None

class DataSet():
    def __init__(self,inp,out):
        self.x = inp
        self.y = out


def get_mean_squared_error(target_values, predictions):
    output = 0
    num_predictions = len(predictions)
    for i in range(0, num_predictions):
        output += ((target_values[i] - predictions[i]) ** 2)

    return (output / num_predictions)


def get_polynomial_output(x,coefficients):
    '''
    Returns the output of the polynomial for a given x and its coefficients.
    The coefficients must be in ascending order
    '''
    degree = len(coefficients)

    output = 0
    for power in range(0, degree):
        output += coefficients[power] * (x ** power)

    return output

def get_predictions(inputs,coefficients):
    predictions = []
    for input in inputs:
        predictions.append(get_polynomial_output(input,coefficients))

    return predictions


def stochastic_gradient_descent(dataset,parameters,step_size,num_epochs):
    param_data = []

    num_points = len(dataset.x)
    data_array = np.zeros((num_points,2))       #numpy 2D array to hold given dataset

    #This is a N*2 matrix where N is the number of data points. Each row contains the x and y values in that order
    for point in range(0,num_points):
        data_array[point][0] = dataset.x[point]
        data_array[point][1] = dataset.y[point]

    for i in range(0,num_epochs):
        temp_data_array = np.copy(data_array)
        np.random.shuffle(temp_data_array)
        for j in range(0,num_points):
            x = temp_data_array[j][0]
            y = temp_data_array[j][1]
            prediction = get_predictions([x],parameters)[0]

            parameters[0] = parameters[0] - step_size*(prediction - y)
            parameters[1] = parameters[1] - step_size*(prediction - y)*x

        param_data.append([parameters[0], parameters[1]])
    return param_data


def initialize_data():
    '''
    This method calls the read_data method and initializes the data into the empty variables
    created at the beginning of the program
    '''
    global TRAINING_DATA
    TRAINING_DATA = read_data("Datasets/Dataset_2_train.csv")
    global VALIDATION_DATA
    VALIDATION_DATA = read_data("Datasets/Dataset_2_valid.csv")
    global TEST_DATA
    TEST_DATA = read_data("Datasets/Dataset_2_test.csv")


def read_data(filename):
    inp = []
    out = []
    with open(filename, 'r') as f:
        for line in f:
            temp_line = line.split(',')
            inp.append(float(temp_line[0]))
            out.append(float(temp_line[1]))
    return DataSet(inp, out)

def run_q1_a():
    initialize_data()
    initial_params = [random(),random()]
    step_size = 10**(-6)
    num_epochs = 10000

    print("The data has been initialized: ")
    print("The initial parameters are set to ({},{}) ".format(initial_params[0],initial_params[1]))
    print("The step size is set to {}".format((step_size)))
    print("The number of epochs is set to {} ".format(num_epochs))

    parameters = stochastic_gradient_descent(dataset=TRAINING_DATA,
                                parameters=initial_params,
                                step_size=step_size,
                                num_epochs=num_epochs)

    epochs = []
    mse_array = []
    for i in range(0,num_epochs):
        predictions = get_predictions(VALIDATION_DATA.x,parameters[i])
        mse = get_mean_squared_error(VALIDATION_DATA.y,predictions)
        mse_array.append(mse)
        epochs.append(i+1)

    print("The MSE for the last epoch of {} is {}".format(num_epochs,mse_array[-1]))

    plt.plot(epochs,mse_array,'b-')
    plt.suptitle("Mean Squared Error For Parameters Generated at Each Epoch")
    plt.xlabel("Epoch Number")
    plt.ylabel("Mean Squared Error")
    plt.show()

def run_q1_b():
    initialize_data()
    initial_params = [random(), random()]
    step_size = 10 ** (-6)
    num_epochs = 10000

    print("The data has been initialized: ")
    print("The initial parameters are set to ({},{}) ".format(initial_params[0], initial_params[1]))
    print("The step size is set to {}".format((step_size)))
    print("The number of epochs is set to {} ".format(num_epochs))

    parameters = stochastic_gradient_descent(dataset=TRAINING_DATA,
                                             parameters=initial_params,
                                             step_size=step_size,
                                             num_epochs=num_epochs)

    epochs = []
    training_mse_array = []
    validation_mse_array = []
    for i in range(0, num_epochs):
        training_predictions = get_predictions(TRAINING_DATA.x,parameters[i])
        validation_predictions = get_predictions(VALIDATION_DATA.x, parameters[i])

        training_mse = get_mean_squared_error(TRAINING_DATA.y,training_predictions)
        validation_mse = get_mean_squared_error(VALIDATION_DATA.y, validation_predictions)

        training_mse_array.append(training_mse)
        validation_mse_array.append(validation_mse)

        epochs.append(i + 1)

    plt1, = plt.plot(epochs, training_mse_array, 'b-',label="Training MSE")
    plt2, = plt.plot(epochs,validation_mse_array,'r-',label="Validation MSE")
    plt.suptitle("Mean Squared Error For Parameters Generated at Each Epoch")
    plt.xlabel("Epoch Number")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.show()

def run_q2_a():
    initialize_data()
    initial_params = [random(), random()]
    num_epochs = 10000

    print("The data has been initialized: ")
    print("The initial parameters are set to ({},{}) ".format(initial_params[0], initial_params[1]))
    print("The number of epochs is set to {} ".format(num_epochs))

    for power in range(0,10):
        step_size = 10**(-power)
        print("Setting step size to {}".format(step_size))

        parameters = stochastic_gradient_descent(dataset=TRAINING_DATA,
                                             parameters=initial_params,
                                             step_size=step_size,
                                             num_epochs=num_epochs)

        final_parameters = parameters[-1]
        predictions = get_predictions(VALIDATION_DATA.x,final_parameters)
        mse = get_mean_squared_error(VALIDATION_DATA.y,predictions)

        print("For step size {}, after {} epochs, the MSE with Validation Data is {}".format(step_size, num_epochs, mse))

def run_q2_b():
    initialize_data()
    initial_params = [random(), random()]
    num_epochs = 10000
    step_size = 0.01

    print("The data has been initialized: ")
    print("The initial parameters are set to ({},{}) ".format(initial_params[0], initial_params[1]))
    print("The number of epochs is set to {} ".format(num_epochs))
    print("The step size is set to {}".format(step_size))

    parameters = stochastic_gradient_descent(dataset=TRAINING_DATA,
                                         parameters=initial_params,
                                         step_size=step_size,
                                         num_epochs=num_epochs)

    final_parameters = parameters[-1]
    predictions = get_predictions(TEST_DATA.x,final_parameters)
    mse = get_mean_squared_error(TEST_DATA.y,predictions)

    print("For step size {}, after {} epochs, the MSE with Test Data is {}".format(step_size, num_epochs, mse))

def run_q3():
    initialize_data()
    initial_params = [random(), random()]
    step_size = 0.01
    num_epochs = 10000
    range = [0,1.5]
    num_sample_points = 150
    random_5_epochs = (10,500,2000,5000,9000)

    print("The data has been initialized: ")
    print("The initial parameters are set to ({},{}) ".format(initial_params[0], initial_params[1]))
    print("The step size is set to {}".format((step_size)))
    print("The number of epochs is set to {} ".format(num_epochs))

    parameters = stochastic_gradient_descent(dataset=TRAINING_DATA,
                                             parameters=initial_params,
                                             step_size=step_size,
                                             num_epochs=num_epochs)

    plot_index = 321
    for epoch in random_5_epochs:
        plt.subplot(plot_index)
        plt1, = plt.plot(TEST_DATA.x,TEST_DATA.y,'r.',label="Test Data")

        best_fit_x = np.linspace(range[0],range[1],num_sample_points)
        best_fit_y = get_polynomial_output(best_fit_x,parameters[epoch+1])

        plt2, = plt.plot(best_fit_x,best_fit_y,'b-',label="Regression Fit")

        plt.legend()
        plt.title("Fit for Epoch {}".format(epoch))

        plot_index+=1

    plt.subplots_adjust(wspace=0.4,hspace=0.5)
    plt.show()


if __name__ == "__main__":
    run_q1_a()