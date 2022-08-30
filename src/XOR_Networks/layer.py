import numpy as np

def sigmoid(val):
    '''
    Applies the sigmoid function to the given value.

    Parameters
    ----------
    val : float
        The value to which to apply the sigmoid function.

    Returns
    ---------
    float
        The result of applying the sigmoid function to the given value.
    '''
    return (1/(1 + np.exp(-1*val)))

def derived_sigmoid(value):
    '''
    Calculates the derivative of the sigmoid function for the given value.

    Parameters
    ----------
    val : float
        The value for which to calculate the derivative of the sigmoid function.

    Returns
    ---------
    float
        The derivative of the sigmoid function at the given value.
    '''
    return sigmoid(value)*(1-sigmoid(value))


class Layer:
    '''
    This class represents a layer in a neural network
    '''
    def __init__(self, first, shape, learning_rate):
        '''
        Constructs a layer and initializes its parameters.

        Parameters
        ----------
        first : boolean
            Indicates if this is the first layer in a network.
        shape : tuple
            Indicates the number of inputs to this layer as well as the number of neurons in this layer.
        learning_rate : float
            Indicates the learning rate that should be used when updating the weight and bias terms.
        '''
        self.size = shape[1]
        self.lr = learning_rate
        self.outputs = np.zeros(self.size)
        self.inputs = np.zeros(self.size)
        if not first:
            self.weights = np.zeros(shape)
            self.bias = np.zeros(self.size)
            self.init_weights()
            self.init_bias()

    def init_weights(self):
        '''
        Initialize the weights of the layer
        '''
        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                self.weights[i, j] = np.random.rand()
    
    def init_bias(self):
        '''
        Initialize the bias terms in the layer
        '''
        for i in range(len(self.bias)):
            self.bias[i] = np.random.rand()

    def calculate_outputs(self, inputs):
        '''
        Calculate the outputs of the layer, given a set of inputs

        Parameters:
        ----------
        inputs : ndarray
            The inputs to use when calculating the outputs.
        '''
        self.inputs = np.array([self.weights[:, j].dot(inputs) for j in range(self.size)]) + self.bias
        self.outputs = np.array([sigmoid(input) for input in self.inputs])

    def set_outputs(self, outputs):
        '''
        Set the outputs of the layer to the given values.

        Parameters
        ----------
        outputs : ndarray
            The values to which to set the outputs of the layer.
        '''
        self.outputs = outputs

    def update_weights(self, deltas, prev_outs):
        '''
        Update the weights of the layer

        Parameters:
        -----------
        deltas : ndarray
            The update steps to use
        prev_outs : ndarray
            The outputs from the preceding layer.
        '''
        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                update = deltas[j]*prev_outs[i]
                self.weights[i, j] -= self.lr*update

    def update_bias(self, deltas):
        '''
        Update the bias terms of the layer

        Parameters:
        -----------
        deltas : ndarray
            The update steps to use
        '''
        for i in range(len(self.bias)):
            update = deltas[i]
            self.bias[i] -= self.lr*update