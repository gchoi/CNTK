# Import the relevant components
from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
import matplotlib.pyplot as plt

import numpy as np
import sys
import os

import cntk as C
import cntk.tests.test_utils
cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
C.cntk_py.set_fixed_random_seed(1) # fix a random seed for CNTK components

###############################################################################
# DATA GENERATION
###############################################################################
# Ensure we always get the same amount of randomness
np.random.seed(0)

# Define the data dimensions
input_dim = 2
num_output_classes = 2

# Helper function to generate a random data sample
def generate_random_data_sample(sample_size, feature_dim, num_classes):
    # Create synthetic data using NumPy. 
    Y = np.random.randint(size=(sample_size, 1), low=0, high=num_classes)

    # Make sure that the data is separable
    X = (np.random.randn(sample_size, feature_dim)+3) * (Y+1)
    X = X.astype(np.float32)    
    # converting class 0 into the vector "1 0 0", 
    # class 1 into vector "0 1 0", ...
    class_ind = [Y==class_number for class_number in range(num_classes)]
    Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
    return X, Y

# Create the input variables denoting the features and the label data. Note: the input 
# does not need additional info on number of observations (Samples) since CNTK first create only 
# the network tooplogy first 
mysamplesize = 64
features, labels = generate_random_data_sample(mysamplesize, input_dim, num_output_classes)

# Plot the data 
import matplotlib.pyplot as plt

# given this is a 2 class 
colors = ['r' if l == 0 else 'b' for l in labels[:,0]]

plt.scatter(features[:,0], features[:,1], c=colors)
plt.xlabel("Scaled age (in yrs)")
plt.ylabel("Tumor size (in cm)")
plt.show()

num_hidden_layers = 2
hidden_layers_dim = 50

# The input variable (representing 1 observation, in our example of age and size) x, which 
# in this case has a dimension of 2. 
#
# The label variable has a dimensionality equal to the number of output classes in our case 2. 

input = C.input_variable(input_dim)
label = C.input_variable(num_output_classes)


def linear_layer(input_var, output_dim):
    input_dim = input_var.shape[0]
    
    weight = C.parameter(shape=(input_dim, output_dim))
    bias = C.parameter(shape=(output_dim))

    return bias + C.times(input_var, weight)


def dense_layer(input_var, output_dim, nonlinearity):
    l = linear_layer(input_var, output_dim)
    
    return nonlinearity(l)


# Define a multilayer feedforward classification model
def fully_connected_classifier_net(input_var, num_output_classes, hidden_layer_dim, num_hidden_layers, nonlinearity):
    
    h = dense_layer(input_var, hidden_layer_dim, nonlinearity)
    for i in range(1, num_hidden_layers):
        h = dense_layer(h, hidden_layer_dim, nonlinearity)
    
    return linear_layer(h, num_output_classes)

# Create the fully connected classfier
z = fully_connected_classifier_net(input, num_output_classes, hidden_layers_dim, num_hidden_layers, C.sigmoid)













