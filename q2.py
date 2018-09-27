import matplotlib.pyplot as plt
import numpy as np



class DataSet():
    def __init__(self,inp,out):
        self.x = inp
        self.y = out

class PolynomialFit():
    '''
    This class can be used to generate the best polynomial fits for a line
    '''

    def __init__(self,dataset,degree):
        self.dataset = dataset
        self.degree = degree


    def get_best_fit(self):
        rows = len(self.dataset.x)
        cols = self.degree + 1

        basis_array = np.zeros((rows, cols))

        for i in range(0, rows):
            for j in range(0, cols):
                basis_array[i][j] = PolynomialFit.polynomial_basis_function(j, self.dataset.x[i])

        mpps_of_basis = np.linalg.pinv(basis_array)

        return np.matmul(mpps_of_basis,np.array(self.dataset.y))

    @staticmethod
    def polynomial_basis_function(power, x):
        if power == 0:
            return 1
        else:
            return x**power

class Plotter():
    def __init__(self):
        pass

    def add_plot(self,x,y,plot_type='b-'):
        plt.plot(x,y,plot_type)

    def show_plot(self):
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


def get_poly_output(x,coefficients):
    '''
    Returns the output of the polynomial for a given x and its coefficients.
    The coefficients must be in ascending order
    '''

    degree = len(coefficients)

    output = 0
    for power in range(0,degree):
        output += coefficients[power]*(x**power)

    return output



def plot_data(dataset,coefficients,validation_data):
    '''
    This function plots the given points onto the graph
    It also takes as input a list of coefficients specifying the best fit polynomial which is also
    plotted for comparison
    '''
    plt.plot(dataset.x,dataset.y,"ro")
    plt.ylabel("Target Values")
    plt.xlabel("Observations")


    best_fit_x = np.linspace(-1,1,100)
    best_fit_y = get_poly_output(best_fit_x,coefficients)
    plt.plot(best_fit_x,best_fit_y,'b-')

    plt.axis([-1,1,-50,50])
    plt.show()

def find_mean_squared_error(test_data,predictions):
    output = 0
    num_predictions = len(predictions)
    for i in range(0,num_predictions):
        output += (test_data[i] - predictions[i])**2

    return (output/num_predictions)

if __name__ == "__main__":
    #### READ TRAINING DATA AND PLOT A BEST FIT LINE #####
    training_data = read_data("Datasets/Dataset_1_train.csv")

    poly_fit = PolynomialFit(training_data,10)
    coefficients = poly_fit.get_best_fit()
    #### END OF THIS PART ##############################


    #### READ VALIDATION DATA AND CALCULATE MEAN SQUARED ERROR ###################

    validation_data = read_data("Datasets/Dataset_1_valid.csv")
    validation_data_input = np.array(validation_data.x)
    predictions = get_poly_output(validation_data_input,coefficients)

    mean_squared_error = find_mean_squared_error(validation_data.y,predictions)
    print(mean_squared_error)


    #plot_data(training_data,coefficients,validation_data)
