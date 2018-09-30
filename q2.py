'''
This program has been specifically tailored to answer question 2 in assignment 1. Its methods
and classes are specifically tailored for this question, however, I have tried to make them
as abstract as possible for other similar programming needs.
'''
import matplotlib.pyplot as plt
import numpy as np

TRAINING_DATA = None
VALIDATION_DATA = None
TEST_DATA = None

class DataSet():
    def __init__(self,inp,out):
        self.x = inp
        self.y = out

class PolynomialMaster():
    '''
    This class can be used to generate the best polynomial fits for a line
    '''

    def __init__(self,dataset):
        self.dataset = dataset

    def get_best_fit(self,degree):
        '''
        Using the initialized data set this method finds the best fit polynomial
        of the specified n-th degree. It uses the least squares regression method
        and returns an array containing all the coefficients found with this method.
        '''
        rows = len(self.dataset.x)
        cols = degree + 1
        basis_array = np.zeros((rows, cols))

        for i in range(0, rows):
            for j in range(0, cols):
                basis_array[i][j] = PolynomialMaster.polynomial_basis_function(j, self.dataset.x[i])

        mpps_of_basis = np.linalg.pinv(basis_array)

        return np.matmul(mpps_of_basis,np.array(self.dataset.y))

    def get_best_fit_l2(self,degree,lambda_val):
        rows = len(self.dataset.x)
        cols = degree + 1
        basis_array = np.zeros((rows, cols))

        for i in range(0, rows):
            for j in range(0, cols):
                basis_array[i][j] = PolynomialMaster.polynomial_basis_function(j, self.dataset.x[i])

        # output = np.matmul(basis_array.transpose(),basis_array)
        # output = output + (lambda_val * np.identity(cols))
        # output = np.linalg.inv(output)

        output = np.linalg.inv((np.matmul(basis_array.transpose(),basis_array)+(lambda_val * np.identity(cols))))
        output = np.matmul(np.matmul(output,basis_array.transpose()),self.dataset.y)
        return output

    @staticmethod
    def get_mean_squared_error(target_values, predictions):
        output = 0
        num_predictions = len(predictions)
        for i in range(0, num_predictions):
            output += ((target_values[i] - predictions[i]) ** 2)

        return (output / num_predictions)

    @staticmethod
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

    @staticmethod
    def polynomial_basis_function(power, x):
        if power == 0:
            return 1
        else:
            return x**power

    @staticmethod
    def get_predictions(inputs,coefficients):
        predictions = []
        for input in inputs:
            predictions.append(PolynomialMaster.get_polynomial_output(input,coefficients))

        return predictions

class Plotter():
    def add_plot(self,x,y,plot_type='r.',label=None):
        if label:
            return plt.plot(x,y,plot_type,label)
        else:
            return plt.plot(x,y,plot_type)

    def add_best_fit_poly(self,coefficients,plot_type='b-',num_sample_points=100,range=(-1,1)):
        best_fit_x = np.linspace(range[0], range[1], num_sample_points)
        best_fit_y = PolynomialMaster.get_polynomial_output(best_fit_x, coefficients)
        plt.plot(best_fit_x,best_fit_y,plot_type)

    def set_axis(self,axis):
        plt.axis(axis)

    def set_xLabel(self,label):
        plt.xlabel(label)

    def set_yLabel(self,label):
        plt.ylabel(label)

    def set_main_title(self,title):
        plt.suptitle(title)

    def show(self):
        plt.show()

    def modify_legend(self,**kwargs):
        plt.legend(**kwargs)

def initialize_data():
    '''
    This method calls the read_data method and initializes the data into the empty variables
    created at the beginning of the program
    '''
    global TRAINING_DATA
    TRAINING_DATA = read_data("Datasets/Dataset_1_train.csv")
    global VALIDATION_DATA
    VALIDATION_DATA = read_data("Datasets/Dataset_1_valid.csv")
    global TEST_DATA
    TEST_DATA = read_data("Datasets/Dataset_1_test.csv")


def read_data(filename):
    inp = []
    out = []
    with open(filename, 'r') as f:
        for line in f:
            temp_line = line.split(',')
            inp.append(float(temp_line[0]))
            out.append(float(temp_line[1]))
    return DataSet(inp, out)

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



def run_part_1():
    #Initialize data, classes, and variables
    initialize_data()
    plotter = Plotter()
    degree = 20

    #Load training data into class and generate coefficients for best fit polynomial
    poly_master = PolynomialMaster(TRAINING_DATA)
    coefficients = poly_master.get_best_fit(degree)

    #Get Mean Squared Error with validation data
    training_predictions = poly_master.get_predictions(TRAINING_DATA.x, coefficients)
    validation_predictions = poly_master.get_predictions(VALIDATION_DATA.x, coefficients)
    training_mse = poly_master.get_mean_squared_error(TRAINING_DATA.y,training_predictions)
    validation_mse = poly_master.get_mean_squared_error(VALIDATION_DATA.y,validation_predictions)
    print("The Mean Squared Error for the Training Data using a degree {} polynomial fit is {}".format(degree,training_mse))
    print("The Mean Squared Error for the Validation Data using a degree {} polynomial fit is {}".format(degree,validation_mse))

    plotter.add_plot(TRAINING_DATA.x,TRAINING_DATA.y,'ro')
    plotter.add_best_fit_poly(coefficients)
    plotter.set_axis([-1,1,-40,40])
    plotter.set_xLabel("Observations")
    plotter.set_yLabel("Results")
    plotter.set_main_title("Polynomial Curve Fitting with a {} Degree Polynomial".format(degree))
    plotter.show()


def run_part_2_a():
    # Initialize data, classes, and variables
    initialize_data()
    plotter = Plotter()
    degree = 20
    lambda_divisions = 100
    mse_training = {}
    mse_validation = {}

    poly_master = PolynomialMaster(TRAINING_DATA)

    lambda_values = divide_interval([0,1],lambda_divisions,True)
    for val in lambda_values:
        coefficients = poly_master.get_best_fit_l2(degree,val)
        training_predictions = poly_master.get_predictions(TRAINING_DATA.x,coefficients)
        validation_predictions = poly_master.get_predictions(VALIDATION_DATA.x,coefficients)

        mse_training[val] = poly_master.get_mean_squared_error(TRAINING_DATA.y,training_predictions)
        mse_validation[val] = poly_master.get_mean_squared_error(VALIDATION_DATA.y,validation_predictions)


    plot1, = plt.plot(mse_training.keys(),mse_training.values(),'b-',label="Training Data MSE")
    plot2, = plt.plot(mse_validation.keys(),mse_validation.values(),'r-',label="Validation Data MSE")
    plotter.set_axis([0,1,0,15])
    plotter.set_xLabel("Lambda Values")
    plotter.set_yLabel("Mean Squared Error")
    plotter.set_main_title("Mean Squared Error Values for Different Lambda Values")
    plotter.modify_legend()
    plotter.show()

def run_part_2_b():
    # Initialize data, classes, and variables
    initialize_data()
    plotter = Plotter()
    degree = 20

    lambda_divisions = 100
    mse_training = {}
    mse_validation = {}

    poly_master = PolynomialMaster(TRAINING_DATA)

    lambda_values = divide_interval([0, 1], lambda_divisions, True)
    for val in lambda_values:
        coefficients = poly_master.get_best_fit_l2(degree, val)
        training_predictions = poly_master.get_predictions(TRAINING_DATA.x, coefficients)
        validation_predictions = poly_master.get_predictions(VALIDATION_DATA.x, coefficients)

        mse_training[val] = poly_master.get_mean_squared_error(TRAINING_DATA.y, training_predictions)
        mse_validation[val] = poly_master.get_mean_squared_error(VALIDATION_DATA.y, validation_predictions)

    smallest_mse= find_min(mse_validation)
    print("The smallest mean squared error found for the validation "
          "data is {} with lambda value = {}".format(smallest_mse[1],smallest_mse[0]))

    print("Let's now test this with test data:")

    coefficients = poly_master.get_best_fit_l2(degree,smallest_mse[0])
    test_predictions = poly_master.get_predictions(TEST_DATA.x,coefficients)
    test_mse = poly_master.get_mean_squared_error(TEST_DATA.y,test_predictions)

    print("The M.S.E. using lambda = {} on the test data is {}".format(smallest_mse[0],test_mse))

def run_part_2_c(lambda_val):
    # Initialize data, classes, and variables
    initialize_data()
    plotter = Plotter()
    degree = 20

    # Load training data into class and generate coefficients for best fit polynomial
    poly_master = PolynomialMaster(TRAINING_DATA)
    coefficients = poly_master.get_best_fit_l2(degree,lambda_val=lambda_val)

    plotter.add_plot(TRAINING_DATA.x, TRAINING_DATA.y, 'ro')
    plotter.add_best_fit_poly(coefficients)
    plotter.set_axis([-1, 1, -40, 40])
    plotter.set_xLabel("Observations")
    plotter.set_yLabel("Results")
    plotter.set_main_title("Polynomial Curve Fitting with a {} Degree Polynomial".format(degree))
    plotter.show()


if __name__ == "__main__":
    run_part_2_b()