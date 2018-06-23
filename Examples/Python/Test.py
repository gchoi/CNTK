class Cifar10:
    def __init__(self):
        self.result = 0
        self.imgSize = 32
        self.numFeature = self.imgSize * self.imgSize * 3
        
    def readBatch(self, src):
        import numpy as np
        import sys
        import pickle as cp
        
        with open(src, 'rb') as f:
            if sys.version_info[0] < 3: 
                d = cp.load(f) 
            else:
                d = cp.load(f, encoding='latin1')
            data = d['data']
            feat = data
        res = np.hstack((feat, np.reshape(d['labels'], (len(d['labels']), 1))))
        return res.astype(np.int)
        
    def loadData(self, src):
        import tarfile
        import numpy as np
        import os
        
        try:
            from urllib.request import urlretrieve
        except ImportError:
            from urllib import urlretrieve
        
        print ('Downloading ' + src)
        fname, h = urlretrieve(src, './delete.me')
        print ('Done.')
        try:
            print ('Extracting files...')
            with tarfile.open(fname) as tar:
                tar.extractall()
            print ('Done.')
            print ('Preparing train set...')
            trn = np.empty((0, self.numFeature + 1), dtype=np.int)
            for i in range(5):
                batchName = './cifar-10-batches-py/data_batch_{0}'.format(i + 1)
                trn = np.vstack((trn, self.readBatch(batchName)))
            print ('Done.')
            print ('Preparing test set...')
            tst = self.readBatch('./cifar-10-batches-py/test_batch')
            print ('Done.')
        finally:
            os.remove(fname)
        return (trn, tst)
        
    def saveTxt(self, filename, ndarray):
        import numpy as np
        
        filename = './data/CIFAR-10/Train_cntk_text.txt'
        
        with open(filename, 'w') as f:
            labels = list(map(' '.join, np.eye(10, dtype=np.uint).astype(str)))
            for row in ndarray:
                row_str = row.astype(str)
                label_str = labels[row[-1]]
                feature_str = ' '.join(row_str[:-1])
                f.write('|labels {} |features {}\n'.format(label_str, feature_str))

    def saveMean(self, fname, data):
        import xml.etree.cElementTree as et
        import numpy as np
        import xml.dom.minidom
        
        root = et.Element('opencv_storage')
        et.SubElement(root, 'Channel').text = '3'
        et.SubElement(root, 'Row').text = str(self.imgSize)
        et.SubElement(root, 'Col').text = str(self.imgSize)
        meanImg = et.SubElement(root, 'MeanImg', type_id='opencv-matrix')
        et.SubElement(meanImg, 'rows').text = '1'
        et.SubElement(meanImg, 'cols').text = str(self.imgSize * self.imgSize * 3)
        et.SubElement(meanImg, 'dt').text = 'f'
        et.SubElement(meanImg, 'data').text = ' '.join(['%e' % n for n in np.reshape(data, (self.imgSize * self.imgSize * 3))])
    
        tree = et.ElementTree(root)
        tree.write(fname)
        x = xml.dom.minidom.parse(fname)
        with open(fname, 'w') as f:
            f.write(x.toprettyxml(indent = '  '))

    def saveTrainImages(self, filename, foldername):
        import numpy as np
        import pickle as cp
        import os
        import sys

        if not os.path.exists(foldername):
            os.makedirs(foldername)
        
        data = {}
        dataMean = np.zeros((3, self.imgSize, self.imgSize)) # mean is in CHW format.
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
                            self.saveImage(fname, data['data'][i, :], data['labels'][i], mapFile, regrFile, 4, mean=dataMean)
        dataMean = dataMean / (50 * 1000)
        self.saveMean('CIFAR-10_mean.xml', dataMean)        

    def saveTestImages(self, filename, foldername):
        import pickle as cp
        import os
        import sys
        
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
                        self.saveImage(fname, data['data'][i, :], data['labels'][i], mapFile, regrFile, 0)

    def downloadImage(self,
                      train_filename = 'Train_cntk_text.txt',
                      test_filename = 'Test_cntk_text.txt'):
        import os
        
        # URLs for the train image and labels data
        url_cifar_data = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        
        # Paths for saving the text files
        data_dir = './data/CIFAR-10/'
        train_filename = data_dir + '/' + train_filename
        test_filename = data_dir + '/' + test_filename
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        try:
            os.chdir(data_dir)   
            trn, tst = self.loadData(url_cifar_data)
            
            print ('Writing train text file...')
            self.saveTxt(train_filename, trn)
            print ('Done.')
            
            print ('Writing test text file...')
            self.saveTxt(test_filename, tst)
            print ('Done.')
            
            print ('Converting train data to png images...')
            self.saveTrainImages(train_filename, 'train')
            print ('Done.')
            
            print ('Converting test data to png images...')
            self.saveTestImages(test_filename, 'test')
            print ('Done.')
        finally:
            os.chdir("../..")

xx = Cifar10()
xx.downloadImage()
