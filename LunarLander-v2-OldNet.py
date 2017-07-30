import gym
env = gym.make('LunarLander-v2')
print(env.action_space)

print(env.observation_space)

from numpy import exp, array, random, dot
import numpy
numpy.set_printoptions(threshold=numpy.nan)

class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.neuron_count = number_of_neurons
        self.input_count = number_of_inputs_per_neuron
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def binaryStep(self, x):
        return 1 * (x>0.5)

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            output = self.think(training_set_inputs)

            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error = training_set_outputs - output

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Adjust the weights.
            self.synaptic_weights += adjustment

    # The neural network thinks.
    def think(self, inputs):
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 =  self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        return output_from_layer1, output_from_layer2



if __name__ == "__main__":

    #Intialise a single neuron neural network.


    #print ("Random starting synaptic weights: ")
   # print (neural_network.synaptic_weights)

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.

    #print(neural_network.think(observation))

    layer1 = NeuronLayer(15, 8)
    layer2 = NeuronLayer(4, 15)
    neural_network = NeuralNetwork(layer1, layer2)
    currentBestNet = neural_network
    max_reward = -9999999

    for i_episode in range(10000):
        networks_list =[]
        for i_batch_number in range(10):
            observation = env.reset()
            

        random.seed(i_episode)

        layer1.synaptic_weights = currentBestNet.layer1.synaptic_weights
        layer2.synaptic_weights = currentBestNet.layer2.synaptic_weights

        for i_mutations in range(10):
            layer1.synaptic_weights[random.randint(0,layer1.input_count),  random.randint(0,layer1.neuron_count)] += 0.5 * random.random() -0.25
            layer2.synaptic_weights[random.randint(0, layer2.input_count), random.randint(0, layer2.neuron_count)] += 0.5 * random.random() - 0.25

        test_neural_network = NeuralNetwork(layer1, layer2)
        episode_reward = 0

        for t in range(250):
            env.render()
            #print(test_neural_network.think(observation)[1])
            action = (test_neural_network.think(observation)[1]).argmax()

            observation, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                break
        if(max_reward< episode_reward):
            max_reward = episode_reward
            currentBestNet = test_neural_network

        print("Episode {} finished after {} timesteps".format(i_episode, t + 1))
        print("Test Reward: {} Maximum reward: {}".format(episode_reward, max_reward))


    #training_set_outputs = array([[0, 1, 1, 0]]).T

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    #neural_network.train(training_set_inputs, training_set_outputs, 10000)

    #print ("New synaptic weights after training: ")
    #print (neural_network.synaptic_weights)

    # Test the neural network with a new situation.
    #print ("Considering new situation [1, 0, 0] -> ?: ")
    #print (neural_network.think(array([1, 0, 0])))