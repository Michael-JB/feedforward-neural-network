import numpy as np

class NeuralNetwork:
  def __init__(self, input_neuron_count, hidden_neuron_count, output_neuron_count):
    self.input_neuron_count = input_neuron_count
    self.hidden_neuron_count = hidden_neuron_count
    self.output_neuron_count = output_neuron_count

    # Randomly initialise biases between 0 and 1
    self.hidden_bias = np.random.rand(hidden_neuron_count)
    self.output_bias = np.random.rand(output_neuron_count)

    # Randomly initialise weights between 0 and 1
    self.hidden_weights = np.random.rand(input_neuron_count, hidden_neuron_count)
    self.output_weights = np.random.rand(hidden_neuron_count, output_neuron_count)

  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))

  def d_sigmoid(self, x):
    return np.exp(x) / (np.exp(x) + 1) ** 2

  def feed_forward(self, input_activations):
    hidden_activations = self.sigmoid(np.dot(input_activations, self.hidden_weights) + self.hidden_bias)
    output_activations = self.sigmoid(np.dot(hidden_activations, self.output_weights) + self.output_bias)
    return output_activations

  def train(self, training_data, epoch_count, learning_rate):
    # Repeat training for set number of epochs
    for _ in range(epoch_count):

      # Initialise weight and bias deltas for this epoch
      epoch_d_hidden_bias = np.zeros(self.hidden_bias.shape)
      epoch_d_output_bias = np.zeros(self.output_bias.shape)
      epoch_d_hidden_weights = np.zeros(self.hidden_weights.shape)
      epoch_d_output_weights = np.zeros(self.output_weights.shape)

      # Present each input from the training data in turn
      for input_activation, expected_output in training_data:

        # Begin feedforward pass
        # Calculate activation for hidden layer
        hidden_weighted_inputs = np.dot(input_activation, self.hidden_weights) + self.hidden_bias
        hidden_activations = self.sigmoid(hidden_weighted_inputs)
        # Calculate activation for output later (completing feed forward phase)
        output_weighted_inputs = np.dot(hidden_activations, self.output_weights) + self.output_bias
        output_activations = self.sigmoid(output_weighted_inputs)

        # Begin backpropagation
        # Calculate derivative of squared error cost function
        d_error = output_activations - expected_output
        # Calculate local error values for output layer
        delta_o = d_error * self.d_sigmoid(output_weighted_inputs)
        # Partial derivative of cost function wrt. output biases is equivalent to this delta value
        d_output_bias = delta_o
        # Calculate partial derivative of cost function wrt. output weights
        d_output_weights = np.outer(delta_o, hidden_activations).T

        # Calculate local error values for hidden layer
        delta_h = np.dot(self.output_weights, delta_o) * self.d_sigmoid(hidden_weighted_inputs)
        # Calculate partial derivatives of cost function wrt. hidden biases and weights
        d_hidden_bias = delta_h
        d_hidden_weights = np.outer(delta_h, input_activation).T

        # Increment epoch deltas with calculated values
        epoch_d_hidden_bias += d_hidden_bias
        epoch_d_output_bias += d_output_bias
        epoch_d_hidden_weights += d_hidden_weights
        epoch_d_output_weights += d_output_weights

      # Average and adjust biases and weights
      self.hidden_bias -= learning_rate * epoch_d_hidden_bias / len(training_data)
      self.output_bias -= learning_rate * epoch_d_output_bias / len(training_data)
      self.hidden_weights -= learning_rate * epoch_d_hidden_weights / len(training_data)
      self.output_weights -= learning_rate * epoch_d_output_weights / len(training_data)

def main():
  input_neuron_count, hidden_neuron_count, output_neuron_count = 2, 5, 3
  learning_rate = 0.5
  epoch_count = 10000

  # Logic training data has the structure (<input_activations>, <expected_output>)
  # <expected_output> = AND, OR, XOR
  logic_training_data = [
    ([0, 0], [0, 0, 0]),
    ([0, 1], [0, 1, 1]),
    ([1, 0], [0, 1, 1]),
    ([1, 1], [1, 1, 0])
  ]

  net = NeuralNetwork(input_neuron_count, hidden_neuron_count, output_neuron_count)
  print('Training network...')
  net.train(logic_training_data, epoch_count, learning_rate)

  for input_activation, expected_output in logic_training_data:
    print('Presenting input: ' + str(input_activation))
    print('Expected output: ' + str(expected_output))
    network_output = net.feed_forward(input_activation)
    print('Network output: ' + str(network_output))
    print('Network output (rounded): ' + str(np.rint(network_output)))
    print('')

if __name__ == '__main__':
  main()