from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
import os
import cntk as C


###############################################################################
# @FUNCTION : create_reader
###############################################################################
# Read a CTF formatted text (as mentioned above) using the CTF deserializer from a file
def create_reader(path, is_training, input_dim, num_label_classes):
    
    ctf = C.io.CTFDeserializer(path, C.io.StreamDefs(
          labels=C.io.StreamDef(field='labels', shape=num_label_classes, is_sparse=False),
          features=C.io.StreamDef(field='features', shape=input_dim, is_sparse=False)))
                          
    return C.io.MinibatchSource(ctf,
        randomize = is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)


###############################################################################
# @FUNCTION : moving_average
###############################################################################
# Define a utility function to compute the moving average sum.
# A more efficient implementation is possible with np.cumsum() function
def moving_average(a, w=5):
    if len(a) < w:
        return a[:]    # Need to send a copy of the array
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]


###############################################################################
# @FUNCTION : print_training_progress
###############################################################################
# Defines a utility that prints the training progress
def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss = "NA"
    eval_error = "NA"

    if mb%frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose: 
            print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(mb, training_loss, eval_error*100))
        
    return mb, training_loss, eval_error


###############################################################################
# @FUNCTION : print_training_progress
###############################################################################
# Ensure the training and test data is available for this tutorial.
# We search in two locations in the toolkit for the cached MNIST data set.
def ensure_data_dir():
    data_found = False # A flag to indicate if train/test data found in local cache
    for data_dir in [os.path.join("..", "Examples", "Image", "DataSets", "MNIST"),
                     os.path.join("data", "MNIST")]:
        
        train_file = os.path.join(data_dir, "Train-28x28_cntk_text.txt")
        test_file = os.path.join(data_dir, "Test-28x28_cntk_text.txt")
        
        if os.path.isfile(train_file) and os.path.isfile(test_file):
            data_found=True
            break
            
    if not data_found:
        raise ValueError("Please generate the data by completing CNTK 103 Part A")
        
    print("Data directory is {0}".format(data_dir))
    
    return train_file, test_file