from random import randint
import numpy as np

values = [] 

value0 = [[0, 1, 1, 1, 0],
[1, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[0, 1, 1, 1, 0]]

value1 = [[0, 0, 1, 0, 0],
[0, 1, 1, 0, 0], 
[1, 0, 1, 0, 0], 
[0, 0, 1, 0, 0], 
[0, 0, 1, 0, 0], 
[0, 0, 1, 0, 0], 
[0, 0, 1, 0, 0], 
[0, 0, 1, 0, 0], 
[0, 0, 1, 0, 0]]

value2 = [[0, 1, 1, 1, 0],
[1, 0, 0, 0, 1], 
[0, 0, 0, 0, 1], 
[0, 0, 0, 0, 1], 
[0, 0, 0, 1, 0], 
[0, 0, 1, 0, 0], 
[0, 1, 0, 0, 0], 
[1, 0, 0, 0, 0], 
[1, 1, 1, 1, 1]]

value3 = [[0, 1, 1, 1, 0],
[1, 0, 0, 0, 1], 
[0, 0, 0, 0, 1], 
[0, 0, 0, 0, 1], 
[0, 0, 0, 1, 0], 
[0, 0, 0, 0, 1], 
[0, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[0, 1, 1, 1, 0]]

value4 = [[0, 0, 0, 1, 0],
[0, 0, 1, 1, 0], 
[0, 0, 1, 1, 0], 
[0, 1, 0, 1, 0], 
[0, 1, 0, 1, 0], 
[1, 0, 0, 1, 0], 
[1, 1, 1, 1, 1], 
[0, 0, 0, 1, 0], 
[0, 0, 0, 1, 0]]

value5 = [[1, 1, 1, 1, 1],
[1, 0, 0, 0, 0], 
[1, 0, 0, 0, 0], 
[1, 1, 1, 1, 0], 
[1, 0, 0, 0, 1], 
[0, 0, 0, 0, 1], 
[0, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[0, 1, 1, 1, 0]]

value6 = [[0, 1, 1, 1, 0],
[1, 0, 0, 0, 1], 
[1, 0, 0, 0, 0], 
[1, 0, 0, 0, 0], 
[1, 1, 1, 1, 0], 
[1, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[0, 1, 1, 1, 0]]

value7 = [[1, 1, 1, 1, 1],
[0, 0, 0, 0, 1], 
[0, 0, 0, 1, 0], 
[0, 0, 0, 1, 0], 
[0, 0, 1, 0, 0], 
[0, 0, 1, 0, 0], 
[0, 1, 0, 0, 0], 
[0, 1, 0, 0, 0], 
[0, 1, 0, 0, 0]]

value8 = [[0, 1, 1, 1, 0],
[1, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[0, 1, 1, 1, 0], 
[1, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[0, 1, 1, 1, 0]]

value9 = [[0, 1, 1, 1, 0],
[1, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[0, 1, 1, 1, 1], 
[0, 0, 0, 0, 1], 
[0, 0, 0, 0, 1], 
[1, 0, 0, 0, 1], 
[0, 1, 1, 1, 0]]

input_dataset = []
output_dataset = []
TEST_CASES_PER_MAP = 3
NUM_MUTATIONS = 2

# Method to initialize the values for digits 0-9
def initialize_values(): 
    values.append(value0)
    values.append(value1)
    values.append(value2)
    values.append(value3)
    values.append(value4)
    values.append(value5)
    values.append(value6)
    values.append(value7)
    values.append(value8)
    values.append(value9)

# Method to mutate by flipping bits in an input layer
def perform_mutate(input):
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
def generate_dataset(): 

    # Loop through the values
    for i in range(10):
        # 1D layer to act as input
        input_layer = []

        # Loop through the 9 X 5 matrix for bitmap
        for j in range(9):
            for k in range(5):
                input_layer.append(values[i][j][k])
        
        input_dataset.append(input_layer)

        # Add corresponding expected output to the layer
        

        for p in range(TEST_CASES_PER_MAP - 1):
            # print("Original input layer: ", input_layer)
            input_dataset.append(perform_mutate(input_layer))
        
        output_layer = []
        for p in range(10):
            if p == i:
                output_layer.append(1)
            else:
                output_layer.append(0)
        
        for p in range(TEST_CASES_PER_MAP):     
            output_dataset.append(output_layer)

    print("Dataset: ")
    print("Input Layers: ")
    for i in input_dataset:
        print(i)
    print()
    print("Output Layers: ")
    for i in output_dataset:
        print(i)

def main():
    initialize_values()
    generate_dataset()

if __name__ == '__main__':
    main()