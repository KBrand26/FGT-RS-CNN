from layer import Layer
from layer import derived_sigmoid
import numpy as np

class MergedNetwork:
    '''
    This class represents a merged XOR neural network
    '''
    def __init__(self, learning_rate, loss_weights, early_stopping):
        """Creates the network layers and initializes the various parameters.

        Parameters
        ----------
        learning_rate : float
            The learning rate to use in the weight updates.
        loss_weights : ndarray
            The weights assigned to the outputs of the XOR, NAND and OR neurons.
        early_stopping : boolean
            Indicates whether early stopping should be used.
        """
        self.size = 3
        self.lr = learning_rate
        input_layer = Layer(True, (0, 2), self.lr)
        hidden_layer = Layer(False, (2, 3), self.lr)
        output_layer = Layer(False, (3, 1), self.lr)
        self.deltas = [[0, 0, 0], [0]]
        self.layers = [input_layer, hidden_layer, output_layer]
        self.ls = loss_weights
        self.es = early_stopping

    def train(self, inputs, labels, max_epochs):
        """Trains the network on the given data.

        Parameters
        ----------
        inputs : ndarray
            The input values to use during training
        labels : ndarray
            The target feature values
        max_epochs : int
            The maximum number of epochs that are allowed during training.

        Returns:
        ----------
        Loss
            The loss values that were obtained after each epoch.
        """
        epochs = 0
        loss = []
        while epochs < max_epochs:
            cur_loss = 0
            for j in range(inputs.shape[1]):
                self.layers[0].set_outputs(inputs[:, j])
                self.forward_propagate()

                results = self.layers[-1].outputs[0]
                cur_loss += self.main_loss(results, labels[0, j])
                
                self.backward_propagate(labels[:, j])
            cur_loss /= inputs.shape[1]
            loss.append(cur_loss)
            epochs += 1
            if cur_loss < 0.01 and self.es:
                break
        return np.array(loss)

    def main_loss(self, result, label):
        """The loss function used for the main network output.

        Parameters
        ----------
        result : float
            The value predicted by the output neuron.
        label : float
            The expected value.

        Returns
        ----------
        Loss
            The loss observed for the prediction.
        """
        return 0.5*((label - result)**2)

    def loss(self, results, labels):
        """The overall loss function of the network.

        Parameters
        ----------
        results : ndarray
            The predictions made by the XOR, NAND and OR neurons.
        labels : ndarray
            The expected outputs.

        Returns
        ----------
        Loss
            The loss observed for the prediction.
        """
        loss = 0
        for i in range(3):
            loss += ((self.ls[i]/2)*((labels[i] - results[i])**2))
        return loss

    def forward_propagate(self):
        '''
        Propogates the inputs through the model.

        Returns
        ---------
        Output
            The output from the final neuron
        '''
        for l in range(1, self.size):
            self.layers[l].calculate_outputs(self.layers[l-1].outputs)
        return self.layers[-1].outputs[0]

    def backward_propagate(self, label):
        '''
        Propogates the loss backwards through the network to update the various
        weight and bias terms.

        Parameters
        ----------
        label : ndarray
            The true labels for the current sample.
        '''
        self.update_deltas(label)
        self.update_weights()
        self.update_bias()

    def update_deltas(self, labels):
        '''
        Calculates the update steps for each of the bias and weight terms.

        Parameters
        ----------
        labels : ndarray
            The true labels for the current sample
        '''
        for j in range(len(self.deltas[-1])):
            derivative = self.ls[0]*(self.layers[-1].outputs[j] - labels[0])
            self.deltas[-1][j] = derivative*derived_sigmoid(self.layers[-1].inputs[j])
        l = self.size - 2
        while l > 0:
            for i in range(len(self.deltas[l - 1])):
                for j in range(len(self.deltas[l])):
                    delta = self.deltas[l][j]
                    weight = self.layers[l + 1].weights[i, j]
                    sig = derived_sigmoid(self.layers[l].inputs[i])
                    if (l == 1) and (i == 0):
                        add_term = self.ls[1]*(self.layers[l].outputs[i] - labels[1])
                        self.deltas[l-1][i] = (weight*delta + add_term)*sig
                    elif (l == 1) and (i == 1):
                        add_term = self.ls[2]*(self.layers[l].outputs[i] - labels[2])
                        self.deltas[l-1][i] = (weight*delta + add_term)*sig
                    else:
                        self.deltas[l - 1][i] = delta*weight*sig
            l -= 1
    
    def predict(self, inputs):
        '''
        Propogates the given inputs through the model and generate predictions for each of them.

        Parameters
        ----------
        inputs : ndarray
            The inputs that should be propogated through the network.
        
        Returns
        ---------
        Predictions
            The predictions made for each sample.
        '''
        self.layers[0].set_outputs(inputs)
        self.forward_propagate()
        results = np.array([self.layers[-1].outputs[0], self.layers[-2].outputs[0], self.layers[-2].outputs[1]])
        return results

    def update_weights(self):
        '''
        Updates the weight terms of each layer.
        '''
        l = self.size - 1
        while l > 0:
            self.layers[l].update_weights(self.deltas[l - 1], self.layers[l - 1].outputs)
            l -= 1

    def update_bias(self):
        '''
        Updates the bias terms of each layer
        '''
        l = self.size - 1
        while l > 0:
            self.layers[l].update_bias(self.deltas[l - 1])
            l -= 1
