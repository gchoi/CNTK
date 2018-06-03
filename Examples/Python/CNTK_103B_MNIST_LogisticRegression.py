# Import the relevant components
from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

import cntk as C
import cntk.tests.test_utils

cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
C.cntk_py.set_fixed_random_seed(1) # fix the random seed so that LR examples are repeatable


###############################################################################
# INITIALIZATION
###############################################################################
# Define the data dimensions
input_dim = 784
num_output_classes = 10


###############################################################################
# DATA READING
###############################################################################
# Read a CTF formatted text (as mentioned above) using the CTF deserializer from a file
def create_reader(path, is_training, input_dim, num_label_classes):
    labelStream = C.io.StreamDef(field='labels', shape=num_label_classes, is_sparse=False)
    featureStream = C.io.StreamDef(field='features', shape=input_dim, is_sparse=False)
    
    deserailizer = C.io.CTFDeserializer(path, C.io.StreamDefs(labels = labelStream, features = featureStream))
            
    return C.io.MinibatchSource(deserailizer,
                                randomize = is_training,
                                max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)













###############################################################################
# @ main
###############################################################################
if __name__ == '__main__':
    # Ensure the training and test data is generated and available for this tutorial.
    # We search in two locations in the toolkit for the cached MNIST data set.
    data_found = False
    
    for data_dir in [os.path.join("..", "Examples", "Image", "DataSets", "MNIST"),
                     os.path.join("data", "MNIST")]:
        train_file = os.path.join(data_dir, "Train-28x28_cntk_text.txt")
        test_file = os.path.join(data_dir, "Test-28x28_cntk_text.txt")
        
        if os.path.isfile(train_file) and os.path.isfile(test_file):
            data_found = True
            break
            
    if not data_found:
        raise ValueError("Please generate the data by completing CNTK 103 Part A")
        
    print("Data directory is {0}".format(data_dir))


