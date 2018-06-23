from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
import os
import cntk as C
import numpy as np
import matplotlib.pyplot as plt
import requests
import h5py

###############################################################################
# @FUNCTION : create_reader
###############################################################################
# Read a CTF formatted text (as mentioned above) using the CTF deserializer from a file
def create_reader(path, is_training, input_dim, num_label_classes):
    
    ctf = C.io.CTFDeserializer(path, C.io.StreamDefs(
            labels = C.io.StreamDef(field='labels', shape=num_label_classes, is_sparse=False),
            features = C.io.StreamDef(field='features', shape=input_dim, is_sparse=False)))
                          
    return C.io.MinibatchSource(ctf,
                                randomize = is_training,
                                max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)

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


###############################################################################
# @FUNCTION : print_image_stats
###############################################################################
# Print image statistics
def print_image_stats(img, text):
    print(text)
    print("Max: {0:.2f}, Median: {1:.2f}, Mean: {2:.2f}, Min: {3:.2f}".format(np.max(img),
                                                                              np.median(img),
                                                                              np.mean(img),
                                                                              np.min(img)))


###############################################################################
# @FUNCTION : plot_image_pair
###############################################################################
# Define a helper function to plot a pair of images
def plot_image_pair(img1, text1, img2, text2):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 6))

    axes[0].imshow(img1, cmap="gray")
    axes[0].set_title(text1)
    axes[0].axis("off")

    axes[1].imshow(img2, cmap="gray")
    axes[1].set_title(text2)
    axes[1].axis("off")


###############################################################################
# @FUNCTION : download
###############################################################################
def download(url, filename):
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as handle:
        for data in response.iter_content(chunk_size=2**20):
            if data: handle.write(data)


###############################################################################
# @FUNCTION : load_vgg
###############################################################################
def load_vgg(path):
    f = h5py.File(path)
    layers = []
    for k in range(f.attrs['nb_layers']):
        g = f['layer_{}'.format(k)]
        n = g.attrs['nb_params']
        layers.append([g['param_{}'.format(p)][:] for p in range(n)])
    f.close()
    return layers


###############################################################################
# @FUNCTION : downloadCifar10
###############################################################################
def downloadCifar10():
    from PIL import Image
    import numpy as np
    import pickle as cp
    import os
    import sys
    import tarfile
    import xml.etree.cElementTree as et
    import xml.dom.minidom
    
    try: 
        from urllib.request import urlretrieve 
    except ImportError: 
        from urllib import urlretrieve
    
    #### DATA DOWNLOAD
    # CIFAR Image data
    imgSize = 32
    numFeature = imgSize * imgSize * 3
    
    def readBatch(src):
        with open(src, 'rb') as f:
            if sys.version_info[0] < 3: 
                d = cp.load(f) 
            else:
                d = cp.load(f, encoding='latin1')
            data = d['data']
            feat = data
        res = np.hstack((feat, np.reshape(d['labels'], (len(d['labels']), 1))))
        return res.astype(np.int)
    
    def loadData(src):
        print ('Downloading ' + src)
        fname, h = urlretrieve(src, './delete.me')
        print ('Done.')
        try:
            print ('Extracting files...')
            with tarfile.open(fname) as tar:
                tar.extractall()
            print ('Done.')
            print ('Preparing train set...')
            trn = np.empty((0, numFeature + 1), dtype=np.int)
            for i in range(5):
                batchName = './cifar-10-batches-py/data_batch_{0}'.format(i + 1)
                trn = np.vstack((trn, readBatch(batchName)))
            print ('Done.')
            print ('Preparing test set...')
            tst = readBatch('./cifar-10-batches-py/test_batch')
            print ('Done.')
        finally:
            os.remove(fname)
        return (trn, tst)
    
    def saveTxt(filename, ndarray):
        with open(filename, 'w') as f:
            labels = list(map(' '.join, np.eye(10, dtype=np.uint).astype(str)))
            for row in ndarray:
                row_str = row.astype(str)
                label_str = labels[row[-1]]
                feature_str = ' '.join(row_str[:-1])
                f.write('|labels {} |features {}\n'.format(label_str, feature_str))
    
    
    #### SAVE IMAGES
    def saveImage(fname, data, label, mapFile, regrFile, pad, **key_parms):
        # data in CIFAR-10 dataset is in CHW format.
        pixData = data.reshape((3, imgSize, imgSize))
        if ('mean' in key_parms):
            key_parms['mean'] += pixData
    
        if pad > 0:
            pixData = np.pad(pixData, ((0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=128) 
    
        img = Image.new('RGB', (imgSize + 2 * pad, imgSize + 2 * pad))
        pixels = img.load()
        for x in range(img.size[0]):
            for y in range(img.size[1]):
                pixels[x, y] = (pixData[0][y][x], pixData[1][y][x], pixData[2][y][x])
        img.save(fname)
        mapFile.write("%s\t%d\n" % (fname, label))
        
        # compute per channel mean and store for regression example
        channelMean = np.mean(pixData, axis=(1,2))
        regrFile.write("|regrLabels\t%f\t%f\t%f\n" % (channelMean[0]/255.0, channelMean[1]/255.0, channelMean[2]/255.0))
        
    def saveMean(fname, data):
        root = et.Element('opencv_storage')
        et.SubElement(root, 'Channel').text = '3'
        et.SubElement(root, 'Row').text = str(imgSize)
        et.SubElement(root, 'Col').text = str(imgSize)
        meanImg = et.SubElement(root, 'MeanImg', type_id='opencv-matrix')
        et.SubElement(meanImg, 'rows').text = '1'
        et.SubElement(meanImg, 'cols').text = str(imgSize * imgSize * 3)
        et.SubElement(meanImg, 'dt').text = 'f'
        et.SubElement(meanImg, 'data').text = ' '.join(['%e' % n for n in np.reshape(data, (imgSize * imgSize * 3))])
    
        tree = et.ElementTree(root)
        tree.write(fname)
        x = xml.dom.minidom.parse(fname)
        with open(fname, 'w') as f:
            f.write(x.toprettyxml(indent = '  '))
    
    
    def saveTrainImages(filename, foldername):
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        data = {}
        dataMean = np.zeros((3, imgSize, imgSize)) # mean is in CHW format.
        with open('train_map.txt', 'w') as mapFile:
            with open('train_regrLabels.txt', 'w') as regrFile:
                for ifile in range(1, 6):
                    with open(os.path.join('./cifar-10-batches-py', 'data_batch_' + str(ifile)), 'rb') as f:
                        if sys.version_info[0] < 3: 
                            data = cp.load(f)
                        else: 
                            data = cp.load(f, encoding='latin1')
                        for i in range(10000):
                            fname = os.path.join(os.path.abspath(foldername), ('%05d.png' % (i + (ifile - 1) * 10000)))
                            saveImage(fname, data['data'][i, :], data['labels'][i], mapFile, regrFile, 4, mean=dataMean)
        dataMean = dataMean / (50 * 1000)
        saveMean('CIFAR-10_mean.xml', dataMean)
    
    def saveTestImages(filename, foldername):
        if not os.path.exists(foldername):
          os.makedirs(foldername)
        with open('test_map.txt', 'w') as mapFile:
            with open('test_regrLabels.txt', 'w') as regrFile:
                with open(os.path.join('./cifar-10-batches-py', 'test_batch'), 'rb') as f:
                    if sys.version_info[0] < 3: 
                        data = cp.load(f)
                    else: 
                        data = cp.load(f, encoding='latin1')
                    for i in range(10000):
                        fname = os.path.join(os.path.abspath(foldername), ('%05d.png' % i))
                        saveImage(fname, data['data'][i, :], data['labels'][i], mapFile, regrFile, 0)
    
    #### SAVE LABLES AND FEATURES
    data_dir = './data/CIFAR-10/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    try:
        os.chdir(data_dir)   
        trn, tst= loadData('http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
        print ('Writing train text file...')
        saveTxt(r'./Train_cntk_text.txt', trn)
        print ('Done.')
        print ('Writing test text file...')
        saveTxt(r'./Test_cntk_text.txt', tst)
        print ('Done.')
        print ('Converting train data to png images...')
        saveTrainImages(r'./Train_cntk_text.txt', 'train')
        print ('Done.')
        print ('Converting test data to png images...')
        saveTestImages(r'./Test_cntk_text.txt', 'test')
        print ('Done.')
    finally:
        os.chdir("../..")