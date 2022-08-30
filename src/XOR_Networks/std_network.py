from layer import Layer, derived_sigmoid
import numpy as np

class StandardNetwork:
    '''
    This class represents a standard neural network
    '''

    def __init__(self, learning_rate, early_stopping):
        '''
        Constructs the model layers and initializes the parameters.

        Parameters:
        -----------
        learning_rate : float
            The learning rate to use for updating the weight and bias terms.
        early_stopping : boolean
            Indicates whether early stopping should be applied to the network.
        '''
        self.size = 3
        self.lr = learning_rate
        input_layer = Layer(True, (0, 2), self.lr)
        hidden_layer = Layer(False, (2, 3), self.lr)
        output_layer = Layer(False, (3, 1), self.lr)
        self.deltas = [[0, 0, 0], [0]]
        self.layers = [input_layer, hidden_layer, output_layer]
        self.es = early_stopping

    def train(self, inputs, label, max_epochs):
        '''
        Trains the network on the given data.

        Parameters:
        -----------
        inputs : ndarray
            The input data to use for the training of the neural network.
        label : ndarray
            The true labels of the samples.
        max_epochs : int
            The maximum number of epochs for which the network can be trained.

        Returns:
        ----------
        Loss
            The loss values obtained after each epoch.
        '''
        epochs = 0
        loss = []
        while epochs < max_epochs:
            cur_loss = 0
            for j in range(inputs.shape[1]):
                self.layers[0].set_outputs(inputs[:, j])
                cur_loss += self.loss(self.forward_propagate(), label[j])
                self.backward_propagate(label[j])
            cur_loss /= inputs.shape[1]
            loss.append(cur_loss)
            epochs += 1
            if cur_loss < 0.01 and self.es:
                break
        return np.array(loss)

    def loss(self, result, label):
        '''
        Calculates the loss for a given prediction.

        Parameters:
        -----------
        result : float
            The predicted value.
        label : float
            The true value.
        
        Returns:
        ----------
        Loss
            The loss calculated for the given prediction.
        '''
        return 0.5*((label - result)**2)

    def forward_propagate(self):
        '''
        Calculates the model outputs for the given input.

        Returns:
        ----------
        Output
            The model output for the given input.
        '''
        for l in range(1, self.size):
            self.layers[l].calculate_outputs(self.layers[l-1].outputs)
        return self.layers[-1].outputs[0]

    def backward_propagate(self, label):
        '''
        Propogates the loss of the current predictions backwards through the network.

        Parameter:
        ----------
        label : float
            The true label of the sample.
        '''
        self.update_deltas(label)
        self.update_weights()
        self.update_bias()

    def update_deltas(self, label):
        '''
        Update the update steps for all of the layers.

        Parameters:
        label : float
            The true label of the sample.
        '''
        for j in range(len(self.deltas[-1])):
            derivative = self.layers[-1].outputs[j] - label
            self.deltas[-1][j] = derivative*derived_sigmoid(self.layers[-1].inputs[j])
        l = self.size - 2
        while l > 0:
            for i in range(len(self.deltas[l - 1])):
                for j in range(len(self.deltas[l])):
                    delta = self.deltas[l][j]
                    weight = self.layers[l + 1].weights[i, j]
                    sig = derived_sigmoid(self.layers[l].inputs[i])
                    self.deltas[l - 1][i] = delta*weight*sig
            l -= 1
    
    def predict(self, inputs):
        '''
        Make predictions for a given set of inputs.

        Parameters:
        -----------
        inputs : ndarray
            The inputs to make predictions for.

        Returns:
        --------
        Predictions
            The predictions made for the given inputs.
        '''
        self.layers[0].set_outputs(inputs)
        return self.forward_propagate()

    def update_weights(self):
        '''
        Update the weights of the layers in the network.
        '''
        l = self.size - 1
        while l > 0:
            self.layers[l].update_weights(self.deltas[l - 1], self.layers[l - 1].outputs)
            l -= 1

    def update_bias(self):
        '''
        Update the bias terms in the layers of the network.
        '''
        l = self.size - 1
        while l > 0:
            self.layers[l].update_bias(self.deltas[l - 1])
            l -= 1

