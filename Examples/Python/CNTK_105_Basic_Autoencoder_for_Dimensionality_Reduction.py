# Import the relevant modules
from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
import numpy as np

# Import CNTK 
import cntk as C
import cntk.tests.test_utils

import helper

cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
C.cntk_py.set_fixed_random_seed(1) # fix a random seed for CNTK components

isFast = True

###############################################################################
# DATA READING
###############################################################################
train_file, test_file = helper.ensure_data_dir()


###############################################################################
# MODEL CREATION
###############################################################################
input_dim = 784
encoding_dim = 32
output_dim = input_dim

def create_model(features):
    with C.layers.default_options(init = C.glorot_uniform()):
        # We scale the input pixels to 0-1 range
        encode = C.layers.Dense(encoding_dim, activation = C.relu)(features/255.0)
        decode = C.layers.Dense(input_dim, activation = C.sigmoid)(encode)

    return decode


###############################################################################
# TRAIN AND TEST THE MODEL
###############################################################################
def train_and_test(reader_train, reader_test, model_func):
    
    ###############################################
    # Training the model
    ###############################################
    
    # Instantiate the input and the label variables
    input = C.input_variable(input_dim)
    label = C.input_variable(input_dim)
    
    # Create the model function
    model = model_func(input)
    
    # The labels for this network is same as the input MNIST image.
    # Note: Inside the model we are scaling the input to 0-1 range
    # Hence we rescale the label to the same range
    # We show how one can use their custom loss function
    # loss = -(y* log(p)+ (1-y) * log(1-p)) where p = model output and y = target
    # We have normalized the input between 0-1. Hence we scale the target to same range
    
    target = label/255.0 
    loss = -(target * C.log(model) + (1 - target) * C.log(1 - model))
    label_error  = C.classification_error(model, target)
    
    # training config
    epoch_size = 30000        # 30000 samples is half the dataset size 
    minibatch_size = 64
    num_sweeps_to_train_with = 5 if isFast else 100
    num_samples_per_sweep = 60000
    num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) // minibatch_size
 
    
    # Instantiate the trainer object to drive the model training
    lr_per_sample = [0.00003]
    lr_schedule = C.learning_parameter_schedule_per_sample(lr_per_sample, epoch_size)
    
    # Momentum which is applied on every minibatch_size = 64 samples
    momentum_schedule = C.momentum_schedule(0.9126265014311797, minibatch_size)
    
    # We use a variant of the Adam optimizer which is known to work well on this dataset
    # Feel free to try other optimizers from 
    # https://www.cntk.ai/pythondocs/cntk.learner.html#module-cntk.learner
    learner = C.fsadagrad(model.parameters,
                         lr=lr_schedule,
                         momentum=momentum_schedule) 
    
    # Instantiate the trainer
    progress_printer = C.logging.ProgressPrinter(0)
    trainer = C.Trainer(model, (loss, label_error), learner, progress_printer)
    
    # Map the data streams to the input and labels.
    # Note: for autoencoders input == label
    input_map = {
        input  : reader_train.streams.features,
        label  : reader_train.streams.features
    } 
    
    aggregate_metric = 0
    for i in range(num_minibatches_to_train):
        # Read a mini batch from the training data file
        data = reader_train.next_minibatch(minibatch_size, input_map = input_map)
        
        # Run the trainer on and perform model training
        trainer.train_minibatch(data)
        samples = trainer.previous_minibatch_sample_count
        aggregate_metric += trainer.previous_minibatch_evaluation_average * samples
        
    train_error = (aggregate_metric*100.0) / (trainer.total_number_of_samples_seen)
    print("Average training error: {0:0.2f}%".format(train_error))
        
    #############################################################################
    # Testing the model
    # Note: we use a test file reader to read data different from a training data
    #############################################################################
        
    # Test data for trained model
    test_minibatch_size = 32
    num_samples = 10000
    num_minibatches_to_test = num_samples / test_minibatch_size
    
    # Test error metric calculation
    metric_numer    = 0
    metric_denom    = 0

    test_input_map = {
        input  : reader_test.streams.features,
        label  : reader_test.streams.features
    }

    for i in range(0, int(num_minibatches_to_test)):
        
        # We are loading test data in batches specified by test_minibatch_size
        # Each data point in the minibatch is a MNIST digit image of 784 dimensions 
        # with one pixel per dimension that we will encode / decode with the 
        # trained model.
        data = reader_test.next_minibatch(test_minibatch_size,
                                       input_map = test_input_map)

        # Specify the mapping of input variables in the model to actual
        # minibatch data to be tested with
        eval_error = trainer.test_minibatch(data)
        
        # minibatch data to be trained with
        metric_numer += np.abs(eval_error * test_minibatch_size)
        metric_denom += test_minibatch_size

    # Average of evaluation errors of all test minibatches
    test_error = (metric_numer*100.0) / (metric_denom) 
    print("Average test error: {0:0.2f}%".format(test_error))
    
    return model, train_error, test_error


num_label_classes = 10
reader_train = helper.create_reader(train_file, True, input_dim, num_label_classes)
reader_test = helper.create_reader(test_file, False, input_dim, num_label_classes)
model, simple_ae_train_error, simple_ae_test_error = train_and_test(reader_train, 
                                                                    reader_test, 
                                                                    model_func = create_model )


###############################################################################
# VISUALIZE SIMPLE AE RESULTS
###############################################################################
# Read some data to run the eval
num_label_classes = 10
reader_eval = helper.create_reader(test_file, False, input_dim, num_label_classes)

eval_minibatch_size = 50
eval_input_map = { input  : reader_eval.streams.features }    
    
eval_data = reader_eval.next_minibatch(eval_minibatch_size,
                                       input_map = eval_input_map)

img_data = eval_data[input].asarray()

# Select a random image
np.random.seed(0)
idx = np.random.choice(eval_minibatch_size)

orig_image = img_data[idx,:,:]
decoded_image = model.eval(orig_image)[0]*255


    
# Print original image
helper.print_image_stats(orig_image, "Original image statistics:")

# Print decoded image
helper.print_image_stats(decoded_image, "Decoded image statistics:")

# Plot the original and the decoded image
img1 = orig_image.reshape(28,28)
text1 = 'Original image'

img2 = decoded_image.reshape(28,28)
text2 = 'Decoded image'

helper.plot_image_pair(img1, text1, img2, text2)

##################################################################################################
#                          DEEP AE
##################################################################################################

###############################################################################
# MODEL CREATION
###############################################################################
input_dim = 784
encoding_dims = [128,64,32]
decoding_dims = [64,128]

encoded_model = None

def create_deep_model(features):
    with C.layers.default_options(init = C.layers.glorot_uniform()):
        encode = C.element_times(C.constant(1.0/255.0), features)

        for encoding_dim in encoding_dims:
            encode = C.layers.Dense(encoding_dim, activation = C.relu)(encode)

        global encoded_model
        encoded_model = encode
        
        decode = encode
        for decoding_dim in decoding_dims:
            decode = C.layers.Dense(decoding_dim, activation = C.relu)(decode)

        decode = C.layers.Dense(input_dim, activation = C.sigmoid)(decode)
        return decode


num_label_classes = 10
reader_train = helper.create_reader(train_file, True, input_dim, num_label_classes)
reader_test = helper.create_reader(test_file, False, input_dim, num_label_classes)

model, deep_ae_train_error, deep_ae_test_error = train_and_test(reader_train, 
                                                                reader_test, 
                                                                model_func = create_deep_model)


# Print original image
helper.print_image_stats(orig_image, "Original image statistics:")

# Print decoded image
helper.print_image_stats(decoded_image, "Decoded image statistics:")

# Plot the original and the decoded image
img1 = orig_image.reshape(28,28)
text1 = 'Original image'

img2 = decoded_image.reshape(28,28)
text2 = 'Decoded image'

helper.plot_image_pair(img1, text1, img2, text2)


# Read some data to run get the image data and the corresponding labels
num_label_classes = 10
reader_viz = helper.create_reader(test_file, False, input_dim, num_label_classes)

image = C.input_variable(input_dim)
image_label = C.input_variable(num_label_classes)

viz_minibatch_size = 50

viz_input_map = { 
    image  : reader_viz.streams.features,
    image_label  : reader_viz.streams.labels
}
    
viz_data = reader_eval.next_minibatch(viz_minibatch_size,
                                      input_map = viz_input_map)

img_data   = viz_data[image].asarray()
imglabel_raw = viz_data[image_label].asarray()

# Map the image labels into indices in minibatch array
img_labels = [np.argmax(imglabel_raw[i,:,:]) for i in range(0, imglabel_raw.shape[0])]       
        
from collections import defaultdict

label_dict=defaultdict(list)
for img_idx, img_label, in enumerate(img_labels):
    label_dict[img_label].append(img_idx)        
    
# Print indices corresponding to 3 digits
randIdx = [1, 3, 9]
for i in randIdx:
    print("{0}: {1}".format(i, label_dict[i]))

from scipy import spatial

def image_pair_cosine_distance(img1, img2):
    if img1.size != img2.size:
        raise ValueError("Two images need to be of same dimension")
    return 1 - spatial.distance.cosine(img1, img2)

# Let s compute the distance between two images of the same number
digit_of_interest = 6

digit_index_list = label_dict[digit_of_interest]

if len(digit_index_list) < 2:
    print("Need at least two images to compare")
else:
    imgA = img_data[digit_index_list[0],:,:][0] 
    imgB = img_data[digit_index_list[1],:,:][0]
    
    # Print distance between original image
    imgA_B_dist = image_pair_cosine_distance(imgA, imgB)
    print("Distance between two original image: {0:.3f}".format(imgA_B_dist))
    
    # Plot the two images
    img1 = imgA.reshape(28,28)
    text1 = 'Original image 1'

    img2 = imgB.reshape(28,28)
    text2 = 'Original image 2'

    helper.plot_image_pair(img1, text1, img2, text2)
    
    # Decode the encoded stream 
    imgA_decoded =  model.eval([imgA])[0]
    imgB_decoded =  model.eval([imgB])   [0]    
    imgA_B_decoded_dist = image_pair_cosine_distance(imgA_decoded, imgB_decoded)

    # Print distance between original image
    print("Distance between two decoded image: {0:.3f}".format(imgA_B_decoded_dist))
    
    # Plot the two images
    # Plot the original and the decoded image
    img1 = imgA_decoded.reshape(28,28)
    text1 = 'Decoded image 1'

    img2 = imgB_decoded.reshape(28,28)
    text2 = 'Decoded image 2'

    helper.plot_image_pair(img1, text1, img2, text2)


imgA = img_data[digit_index_list[0],:,:][0] 
imgA_encoded =  encoded_model.eval([imgA])

print("Length of the original image is {0:3d} and the encoded image is {1:3d}".format(len(imgA), 
                                                                                      len(imgA_encoded[0])))
print("\nThe encoded image: ")
print(imgA_encoded[0])


digitA = 3
digitB = 8

digitA_index = label_dict[digitA]
digitB_index = label_dict[digitB]

imgA = img_data[digitA_index[0],:,:][0] 
imgB = img_data[digitB_index[0],:,:][0]

# Print distance between original image
imgA_B_dist = image_pair_cosine_distance(imgA, imgB)
print("Distance between two original image: {0:.3f}".format(imgA_B_dist))
    
# Plot the two images
img1 = imgA.reshape(28,28)
text1 = 'Original image 1'

img2 = imgB.reshape(28,28)
text2 = 'Original image 2'

helper.plot_image_pair(img1, text1, img2, text2)
    
# Decode the encoded stream 
imgA_decoded =  model.eval([imgA])[0]
imgB_decoded =  model.eval([imgB])[0]    
imgA_B_decoded_dist = image_pair_cosine_distance(imgA_decoded, imgB_decoded)

#Print distance between original image
print("Distance between two decoded image: {0:.3f}".format(imgA_B_decoded_dist))

# Plot the original and the decoded image
img1 = imgA_decoded.reshape(28,28)
text1 = 'Decoded image 1'

img2 = imgB_decoded.reshape(28,28)
text2 = 'Decoded image 2'

helper.plot_image_pair(img1, text1, img2, text2)


# Simple autoencoder test error
print(simple_ae_test_error)

# Deep autoencoder test error
print(deep_ae_test_error)