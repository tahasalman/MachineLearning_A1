import matplotlib.pyplot as plt
import numpy as np



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

    @staticmethod
    def get_mean_squared_error(target_values, predictions):
        output = 0
        num_predictions = len(predictions)
        for i in range(0, num_predictions):
            output += (target_values[i] - predictions[i]) ** 2

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
    def add_plot(self,x,y,plot_type='r.'):
        plt.plot(x,y,plot_type)

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



def read_data(filename):
    inp = []
    out = []
    with open(filename, 'r') as f:
        for line in f:
            temp_line = line.split(',')
            inp.append(float(temp_line[0]))
            out.append(float(temp_line[1]))
    return DataSet(inp, out)


if __name__ == "__main__":
    #Initialize data, classes, and variables
    training_data = read_data("Datasets/Dataset_1_train.csv")
    validation_data = read_data("Datasets/Dataset_1_valid.csv")
    test_data = read_data("Datasets/Dataset_1_test.csv")
    plotter = Plotter()
    degree = 20

    #Load training data into class and generate coefficients for best fit polynomial
    poly_master = PolynomialMaster(training_data)
    coefficients = poly_master.get_best_fit(degree)

    #Get Mean Squared Error with validation data
    predictions = PolynomialMaster.get_predictions(validation_data.x,coefficients)
    mse = PolynomialMaster.get_mean_squared_error(validation_data.y,predictions)
    print("The Mean Squared Error for a degree {} polynomial fit is {}".format(degree,mse))

    plotter.add_plot(training_data.x,training_data.y,'ro')
    plotter.add_best_fit_poly(coefficients)
    plotter.set_axis([-1,1,-40,40])
    plotter.set_xLabel("Observations")
    plotter.set_yLabel("Results")
    plotter.set_main_title("Polynomial Curve Fitting for the Output with a {} Degree Polynomial".format(degree))
    plotter.show()
