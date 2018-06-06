from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

import cntk as C
import cntk.tests.test_utils

import helper

cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
C.cntk_py.set_fixed_random_seed(1) # fix a random seed for CNTK components


###############################################################################
# DATA READING
###############################################################################
# Define the data dimensions
input_dim_model = (1, 28, 28)    # images are 28 x 28 with 1 channel of color (gray)
input_dim = 28*28                # used by readers to treat input data as a vector
num_output_classes = 10


###############################################################################
# BUILDING CNN MODELS
###############################################################################
x = C.input_variable(input_dim_model)
y = C.input_variable(num_output_classes)


###############################################################################
# @FUNCTION : create_model
###############################################################################
# function to build model
def create_model(features):
    with C.layers.default_options(init=C.glorot_uniform(), activation=C.relu):
        h = features
        h = C.layers.Convolution2D(filter_shape=(5,5), 
                                   num_filters=8, 
                                   strides=(2,2), 
                                   pad=True, name='first_conv')(h)
        h = C.layers.Convolution2D(filter_shape=(5,5), 
                                   num_filters=16, 
                                   strides=(2,2), 
                                   pad=True, name='second_conv')(h)
        r = C.layers.Dense(num_output_classes, activation=None, name='classify')(h)
        return r


# Create the model
z = create_model(x)

# Print the output shapes / parameters of different components
print("Output Shape of the first convolution layer:", z.first_conv.shape)
print("Output Shape of the second convolution layer:", z.second_conv.shape)
print("Bias value of the last dense layer:", z.classify.b.value)

# Number of parameters in the network
C.logging.log_number_of_parameters(z)


###############################################################################
# TRAINING
###############################################################################
def create_criterion_function(model, labels):
    loss = C.cross_entropy_with_softmax(model, labels)
    errs = C.classification_error(model, labels)
    return loss, errs # (model, labels) -> (loss, error metric)


###############################################################################
# CONFIGURE TRAINING
###############################################################################
def train_test(train_reader, test_reader, model_func, num_sweeps_to_train_with=10):
    
    # Instantiate the model function; x is the input (feature) variable 
    # We will scale the input image pixels within 0-1 range by dividing all input value by 255.
    model = model_func(x/255)
    
    # Instantiate the loss and error function
    loss, label_error = create_criterion_function(model, y)
    
    # Instantiate the trainer object to drive the model training
    learning_rate = 0.2
    lr_schedule = C.learning_parameter_schedule(learning_rate)
    learner = C.sgd(z.parameters, lr_schedule)
    trainer = C.Trainer(z, (loss, label_error), [learner])
    
    # Initialize the parameters for the trainer
    minibatch_size = 64
    num_samples_per_sweep = 60000
    num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size
    
    # Map the data streams to the input and labels.
    input_map={
        y  : train_reader.streams.labels,
        x  : train_reader.streams.features
    } 
    
    # Uncomment below for more detailed logging
    training_progress_output_freq = 500
     
    # Start a timer
    start = time.time()

    for i in range(0, int(num_minibatches_to_train)):
        # Read a mini batch from the training data file
        data=train_reader.next_minibatch(minibatch_size, input_map=input_map) 
        trainer.train_minibatch(data)
        helper.print_training_progress(trainer, i, training_progress_output_freq, verbose=1)
     
    # Print training time
    print("Training took {:.1f} sec".format(time.time() - start))
    
    # Test the model
    test_input_map = {
        y  : test_reader.streams.labels,
        x  : test_reader.streams.features
    }

    # Test data for trained model
    test_minibatch_size = 512
    num_samples = 10000
    num_minibatches_to_test = num_samples // test_minibatch_size

    test_result = 0.0   

    for i in range(num_minibatches_to_test):
    
        # We are loading test data in batches specified by test_minibatch_size
        # Each data point in the minibatch is a MNIST digit image of 784 dimensions 
        # with one pixel per dimension that we will encode / decode with the 
        # trained model.
        data = test_reader.next_minibatch(test_minibatch_size, input_map=test_input_map)
        eval_error = trainer.test_minibatch(data)
        test_result = test_result + eval_error

    # Average of evaluation errors of all test minibatches
    print("Average test error: {0:.2f}%".format(test_result*100 / num_minibatches_to_test))


###############################################################################
# RUN THE TRAINER AND TEST MODEL
###############################################################################
def do_train_test():
    global z
    z = create_model(x)
    reader_train = helper.create_reader(train_file, True, input_dim, num_output_classes)
    reader_test = helper.create_reader(test_file, False, input_dim, num_output_classes)
    train_test(reader_train, reader_test, z)
    
do_train_test()


print("Bias value of the last dense layer:", z.classify.b.value)










