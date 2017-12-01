import mxnet as mx
import pandas as pd
import logging
import time
import numpy as np
from scipy.misc import imread, imresize
import os
src = 'D:\\Data\\pFabre\\data\\'




model_dir = 'D:\\Data\\mxnet\\models\\'
model_prefix = os.path.join(model_dir, 'Inception')
num_round = 9

"""
model_dir = 'D:\\Data\\mxnet\\models\\'
model_prefix = os.path.join(model_dir, 'Inception-BN')
num_round = 126
"""

model = mx.model.FeedForward.load(model_prefix, num_round, numpy_batch_size=1)

"""
dataiterSample = mx.io.ImageRecordIter(
    path_imgrec=src+'trSample.rec',
    data_shape=(3,224,224),
    batch_size=10,
    mean_r=117.0,
    mean_g=117.0,
    mean_b=117.0,
    preprocess_threads=6,
    prefetch_buffer=1)
"""



#### LOAD IMAGE
# img :  3 * 224 * 224 array
df = pd.DataFrame()
trainLst = os.listdir(src+'Donnees_apprentissage\\Donnees_apprentissage\\')
batch = 10


start_time = time.time()
df = pd.DataFrame()
for k in np.arange(int((len(trainLst)-1)/batch)+1):
#for k in np.arange(20):
    #todo print number of plots
    #todo selection of features
    recs = trainLst[k * batch:(k+1) * batch]
    img = np.array([imresize(imread(src + 'Donnees_apprentissage\\Donnees_apprentissage\\'+file),(224,224)) for file in recs])
    img = img.swapaxes(1,3)
    prob = model.predict(img)
    df = df.append(pd.concat([pd.DataFrame({'photo_id':recs}), pd.DataFrame(prob, columns=['feat'+str(j+1) for j in range(prob.shape[1])])], axis=1))
    if (k + 1) % 50 == 0:
        print('%d processed: %ds' % ((k + 1), time.time() - start_time))



### ITER IMAGE
dataiter = mx.io.ImageRecordIter(
    path_imgrec=src+'tr.rec',
    data_shape=(3,224,224),
    batch_size=100,
    mean_r=117.0,
    mean_g=117.0,
    mean_b=117.0,
    preprocess_threads=1,
    prefetch_buffer=1)

#col_sel = pd.read_csv(tgt+'col.sel.csv')

NUMBATCH = 2349
first = True
for i in range(NUMBATCH):
    batch = dataiter.next()
    idx = batch.index
    print(idx)
    pad = batch.pad
    data = batch.data[0]

    prob = model.predict(data)

    #todo
    """
    redness,
      automatic segmentation and measure of sphericality
      
      bulging,  and tympanic membrane perforation may suggest an OM
condition."""
  # another way to load images
  # 2 understand sructure of submissions & everything
  # 3 select which pretrained model is the best one
  #machine learning stuff
  # 10 col selection
  #prob = prob[:, col_sel.col-1]

  if pad > 0:
    N = len(idx) - pad
    idx = idx[:N]
    prob = prob[:N]

  df = pd.concat([pd.DataFrame({'photo_id':idx}), pd.DataFrame(prob, columns=['feat'+str(j+1) for j in range(prob.shape[1])])], axis=1)

  header = first
  mode = 'w' if first else 'a'
  df.to_csv(src+'outfileTr.csv', index=False, header=header, mode=mode)
  first = False

  if (i+1) % 100 == 0:
    print('%d processed: %ds' % ((i+1), time.time() - start_time))

print('done; elapsed = %ds' % (time.time() - start_time))

