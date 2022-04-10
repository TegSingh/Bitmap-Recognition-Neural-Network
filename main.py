from random import randint
import numpy as np
from input_values import *

# Initialize bitmaps
values = [] 



TEST_CASES_PER_MAP = 3
NUM_MUTATIONS = 2
INPUT_LAYER_LENGTH = 45
HIDDEN_LAYER_LENGTH = 10
OUTPUT_LAYER_LENGTH = 10


class ANN(object):
    
    # Define the constructor
    def __init__(self):
        # Declare input, output dataset
        self.input_dataset = []
        self.output_dataset = []

        # Declrae input output and hidden layers currently under concern
        self.input_layer = []
        self.hidden_layer = []
        self.output_layer = []

        # Declare weights - assumption 3 layers
        self.weights1 = []
        self.weights2 = []      

    # Method to print Neural Network for one input        
    def printNN(self):
        print("Input Layer: ", self.input_layer)
        print("Hidden Layer: ", self.hidden_layer)
        print("Output Layer: ", self.output_layer)

    # Method to mutate by flipping bits in an input layer
    def perform_mutate(self, input):
        mutation = input.copy()
        # print("Input in mutation method: ", input)
        for i in range(NUM_MUTATIONS):
            index_mutate = randint(0, 44)
            # print(index_mutate)
            if input[index_mutate] == 0:
                mutation[index_mutate] = 1
            else:
                mutation[index_mutate] = 0
                
        # print("Mutation in mutation method: ", mutation)
        return mutation

    # Method to create the dataset for training
    def generate_dataset(self): 

        # Loop through the values
        for i in range(10):
            # 1D layer to act as input
            input_layer = []

            # Loop through the 9 X 5 matrix for bitmap
            for j in range(9):
                for k in range(5):
                    input_layer.append(values[i][j][k])
            
            self.input_dataset.append(input_layer)

            for p in range(TEST_CASES_PER_MAP - 1):
                # print("Original input layer: ", input_layer)
                self.input_dataset.append(self.perform_mutate(input_layer))
            
            output_layer = []
            for p in range(10):
                if p == i:
                    output_layer.append(1)
                else:
                    output_layer.append(0)
            
            for p in range(TEST_CASES_PER_MAP):     
                self.output_dataset.append(output_layer)

        # Print Training dataset
        print("Dataset: ")
        print("Input Layers: ")
        for i in self.input_dataset:
            print(i)
        print()
        print("Output Layers: ")
        for i in self.output_dataset:
            print(i)

    # Method to initialize weights as random values
    def set_weights(self):
        
        for i in range(HIDDEN_LAYER_LENGTH):
            weights1_row = []
            for j in range(INPUT_LAYER_LENGTH):
                weights1_row.append(np.random.uniform())
            self.weights1.append(weights1_row)
        
        for i in range(OUTPUT_LAYER_LENGTH):
            weights2_row = []
            for j in range(HIDDEN_LAYER_LENGTH):
                weights2_row.append(np.random.uniform())
            self.weights2.append(weights2_row)


        print("\nWeights between Input layer and Hidden layer: ")        
        for i in self.weights1:
            print(i)
        
        print("\nWeights between Hidden Layer and Output Layer: ")
        for i in self.weights2:
            print(i)

    # Define the sigmoid function for forward propagation
    def sigmoid_forward(self, input):
        updated_input = []
        random_bias = 0.99
        for i in input:
            updated_input.append(1/(1+np.exp(-i)))
        return updated_input

    # Define the sigmoid function for backward propagation
    def sigmoid_backward(self):
        pass

    # Method to perform forward propagation one layer at a time
    def forward_propagate_single(self, layer1, weights):
        print("Individual Forward Propagation method called")

        output_layer = []

        # Perform Matric multiplication to get output for layer2
        for i in range(len(weights)):
            dot_product = 0
            for j in range(len(layer1)):
                dot_product += (weights[i][j] * layer1[j])
            output_layer.append(dot_product)

        return output_layer

    # Method to perform backward propagation
    def forward_propagate(self, input_layer): 
        print("Forward Propagation method called")
        
        # Perform the forward propagation from input layer to hidden layer and Apply Sigmoid
        self.input_layer = input_layer
        hidden_layer = self.forward_propagate_single(input_layer, self.weights1)
        hidden_layer_activated = self.sigmoid_forward(hidden_layer)
        self.hidden_layer = hidden_layer_activated
        
        # Perform forward propagation from hidden layer to output layer and apply sigmoid method
        output_layer = self.forward_propagate_single(hidden_layer, self.weights2)
        output_layer_activated = self.sigmoid_forward(output_layer)
        self.output_layer = output_layer_activated

        self.printNN()

    # Method to perform backward propagation
    def backward_propagate(self): 
        pass

    # Training method
    def train_ANN(self): 
        pass

def main():

    network = ANN()
    initialize_values(values) # Initialize values
    network.generate_dataset() # Generate input and output layers
    network.set_weights()

    # Checking output
    sum = 0
    for i in range(len(network.input_dataset[0])):
        print(network.input_dataset[0][i], network.weights1[0][i], network.input_dataset[0][i]*network.weights1[0][i])
        sum += (network.input_dataset[0][i]*network.weights1[0][i])
    print(sum)

    for i in network.input_dataset:
        network.forward_propagate(i)

if __name__ == '__main__':
    main()