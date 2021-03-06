#Use the pre-trained CaffeNet model to extract features
#It's common to use the fc7 layer. Each feature vector will be 4096-dim.
#I also tried the last output layer. Each feature will be a 1000-dim probability vector.



caffe_root = '/usr/bin/caffe/'
data_root = '/mnt/Data/Data/Kaggle/Yelp/train_photos/'

import numpy as np
import sys

sys.path.insert(0, caffe_root + 'python')

import caffe
import os

#todo here download models & eveyting ?
if not os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print("Downloading pre-trained CaffeNet model...")
    !caffe_root / scripts / download_model_binary.py.. / models / bvlc_reference_caffenet

## Use GPU
#caffe.set_device(0)
#caffe.set_mode_gpu()


def extract_features(images, layer='fc7'):
    net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                    caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                    caffe.TEST)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(
        1))  # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB]]

    num_images = len(images)
    net.blobs['data'].reshape(num_images, 3, 227, 227)
    net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data', caffe.io.load_image(x)), images)
    out = net.forward()

    return net.blobs[layer].data



# extract image features and save it to .h5

# Initialize files
import h5py
#f.close()
f = h5py.File(data_root+'train_image_fc7features.h5','w')
filenames = f.create_dataset('photo_id',(0,), maxshape=(None,),dtype='|S54')
feature = f.create_dataset('feature',(0,4096), maxshape = (None,4096))
f.close()

import pandas as pd
train_photos = pd.read_csv(data_root+'train_photo_to_biz_ids.csv')
train_folder = data_root+'train_photos/'
train_images = [os.path.join(train_folder, str(x)+'.jpg') for x in train_photos['photo_id']]  # get full filename

num_train = len(train_images)
print "Number of training images: ", num_train
batch_size = 500

# Training Images
for i in range(0, num_train, batch_size):
    images = train_images[i: min(i+batch_size, num_train)]
    features = extract_features(images, layer='fc7')
    num_done = i+features.shape[0]
    f= h5py.File(data_root+'train_image_fc7features.h5','r+')
    f['photo_id'].resize((num_done,))
    f['photo_id'][i: num_done] = np.array(images)
    f['feature'].resize((num_done,features.shape[1]))
    f['feature'][i: num_done, :] = features
    f.close()
    if num_done%20000==0 or num_done==num_train:
        print "Train images processed: ", num_done



#######################
### Check the file content

f = h5py.File(data_root + 'train_image_fc7features.h5', 'r')
print
'train_image_features.h5:'
for key in f.keys():
    print
    key, f[key].shape

print
"\nA photo:", f['photo_id'][0]
print
"Its feature vector (first 10-dim): ", f['feature'][0][0:10], " ..."
f.close()


######################
import h5py
import pandas as pd

batch_size = 500

f = h5py.File(data_root + 'test_image_fc7features.h5', 'w')
filenames = f.create_dataset('photo_id', (0,), maxshape=(None,), dtype='|S54')
feature = f.create_dataset('feature', (0, 4096), maxshape=(None, 4096))
f.close()

test_photos = pd.read_csv(data_root + 'test_photo_to_biz.csv')
test_folder = data_root + 'test_photos/'
test_images = [os.path.join(test_folder, str(x) + '.jpg') for x in test_photos['photo_id'].unique()]
num_test = len(test_images)
print
"Number of test images: ", num_test

# Test Images
for i in range(0, num_test, batch_size):
    images = test_images[i: min(i + batch_size, num_test)]
    features = extract_features(images, layer='fc7')
    num_done = i + features.shape[0]

    f = h5py.File(data_root + 'test_image_fc7features.h5', 'r+')
    f['photo_id'].resize((num_done,))
    f['photo_id'][i: num_done] = np.array(images)
    f['feature'].resize((num_done, features.shape[1]))
    f['feature'][i: num_done, :] = features
    f.close()
    if num_done % 20000 == 0 or num_done == num_test:
        print
        "Test images processed: ", num_done



### Check the file content
f = h5py.File(data_root+'test_image_fc7features.h5','r')
for key in f.keys():
    print key, f[key].shape
print "\nA photo:", f['photo_id'][0]
print "feature vector: (first 10-dim)", f['feature'][0][0:10], " ..."
f.close()