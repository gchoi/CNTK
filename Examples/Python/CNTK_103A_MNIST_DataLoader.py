# Import the relevant modules to be used later
from __future__ import print_function
import gzip
import matplotlib.pyplot as plt
import numpy as np
import os
import struct

try: 
    from urllib.request import urlretrieve 
except ImportError: 
    from urllib import urlretrieve

###############################################################################
# DATA DOWNLOAD
###############################################################################
# Functions to load MNIST images and unpack into train and test set.
# - loadData reads a image and formats it into a 28x28 long array
# - loadLabels reads the corresponding label data, one for each image
# - load packs the downloaded image and label data into a combined format to be read later by 
#   the CNTK text reader 

def loadData(src, cimg):
    print ('Downloading ' + src)
    gzfname, h = urlretrieve(src, './delete.me')
    print ('Done.')
    try:
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))
            # Read magic number.
            if n[0] != 0x3080000:
                raise Exception('Invalid file: unexpected magic number.')
            # Read number of entries.
            n = struct.unpack('>I', gz.read(4))[0]
            if n != cimg:
                raise Exception('Invalid file: expected {0} entries.'.format(cimg))
            crow = struct.unpack('>I', gz.read(4))[0]
            ccol = struct.unpack('>I', gz.read(4))[0]
            if crow != 28 or ccol != 28:
                raise Exception('Invalid file: expected 28 rows/cols per image.')
            # Read data.
            res = np.fromstring(gz.read(cimg * crow * ccol), dtype = np.uint8)
    finally:
        os.remove(gzfname)
    return res.reshape((cimg, crow * ccol))


def loadLabels(src, cimg):
    print ('Downloading ' + src)
    gzfname, h = urlretrieve(src, './delete.me')
    print ('Done.')
    try:
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))
            # Read magic number.
            if n[0] != 0x1080000:
                raise Exception('Invalid file: unexpected magic number.')
            # Read number of entries.
            n = struct.unpack('>I', gz.read(4))
            if n[0] != cimg:
                raise Exception('Invalid file: expected {0} rows.'.format(cimg))
            # Read labels.
            res = np.fromstring(gz.read(cimg), dtype = np.uint8)
    finally:
        os.remove(gzfname)
    return res.reshape((cimg, 1))


def try_download(dataSrc, labelsSrc, cimg):
    data = loadData(dataSrc, cimg)
    labels = loadLabels(labelsSrc, cimg)
    return np.hstack((data, labels))


# Save the data files into a format compatible with CNTK text reader
def savetxt(filename, ndarray):
    dir = os.path.dirname(filename)

    if not os.path.exists(dir):
        os.makedirs(dir)

    if not os.path.isfile(filename):
        print("Saving", filename )
        with open(filename, 'w') as f:
            labels = list(map(' '.join, np.eye(10, dtype=np.uint).astype(str)))
            for row in ndarray:
                row_str = row.astype(str)
                label_str = labels[row[-1]]
                feature_str = ' '.join(row_str[:-1])
                f.write('|labels {} |features {}\n'.format(label_str, feature_str))
    else:
        print("File already exists", filename)

###############################################################################
# @ main
###############################################################################
if __name__ == '__main__':
    ###############################################################################
    # DOWNLOAD THE DATA
    ###############################################################################
    # URLs for the train image and label data
    url_train_image = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    url_train_labels = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
    num_train_samples = 60000
    
    print("Downloading train data")
    train = try_download(url_train_image, url_train_labels, num_train_samples)
    
    # URLs for the test image and label data
    url_test_image = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
    url_test_labels = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    num_test_samples = 10000
    
    print("Downloading test data")
    test = try_download(url_test_image, url_test_labels, num_test_samples)    
    
    ###############################################################################
    # VISUALIZE DATA
    ###############################################################################
    
    # Plot a random image
    sample_number = 5001
    plt.imshow(train[sample_number,:-1].reshape(28,28), cmap="gray_r")
    plt.axis('off')
    print("Image Label: ", train[sample_number,-1])
    
    # Save the train and test files (prefer our default path for the data)
    data_dir = os.path.join("..", "Examples", "Image", "DataSets", "MNIST")
    if not os.path.exists(data_dir):
        data_dir = os.path.join("data", "MNIST")
    
    print ('Writing train text file...')
    savetxt(os.path.join(data_dir, "Train-28x28_cntk_text.txt"), train)
    
    print ('Writing test text file...')
    savetxt(os.path.join(data_dir, "Test-28x28_cntk_text.txt"), test)
    
    print('Done')