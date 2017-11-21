#extract features with caffe python


import matplotlib.pyplot as plt


caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys

sys.path.insert(0, caffe_root + 'python')

import caffe

import os

if not os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print("Downloading pre-trained CaffeNet model...")
    !../ scripts / download_model_binary.py.. / models / bvlc_reference_caffenet

caffe.set_mode_cpu()
net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))  #
transformer.set_mean('data',
                     np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))  # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

net.blobs['data'].reshape(1, 3, 227, 227)
net.blobs['data'].data[...] = transformer.preprocess('data',
                                                     caffe.io.load_image(caffe_root + 'examples/images/cat.jpg'))
out = net.forward()
print("First 5-dim of predicted probability is \n #{}.".format(out['prob'][0][0:5]))

# load labels
imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
try:
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
except:
    !../ data / ilsvrc12 / get_ilsvrc_aux.sh
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

# sort top k predictions from softmax output
top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
print
"\n Top 5 classes: \n", labels[top_k]






##########################
#extract features with C++

import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2

caffe_root = '../'


def save2txt(db_name):
    img_db = lmdb.open(db_name)
    txn = img_db.begin()
    cursor = txn.cursor()
    cursor.iternext()

    count = 0
    train = {}

    datum = caffe_pb2.Datum()
    count = 0
    train = {}
    for key, value in cursor:
        datum.ParseFromString(value)
        data = caffe.io.datum_to_array(datum)
        data = np.reshape(data, (1, np.product(data.shape)))[0]
        train[count] = data
        count += 1
    return train