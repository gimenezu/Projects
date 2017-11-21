# mxnet prediction using pretrained model

import mxnet as mx
import pandas as pd
import logging

import os
import time

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

BATCHSIZE = 100
tgt = '/mnt/Data/Data/Kaggle/Yelp/' + 'third/'
src = '/mnt/Data/Data/Kaggle/Yelp/train_photos/'

# for train data
data_prefix = '/mnt/Data/Data/Kaggle/Yelp/tr'
NUMBATCH = 2349
outfile = 'feat03-21k.tr.csv'
# for test data
#data_prefix = 'te'
#NUMBATCH = 2372
#outfile = 'feat03-21k.te.csv'

model_dir = '/mnt/Data/Data/mxnet/models/inception-21k/model/'
model_prefix = os.path.join(model_dir, 'Inception')
num_round = 9



#todo add and load model in mxnet
start_time = time.time()

model = mx.model.FeedForward.load(model_prefix, num_round, numpy_batch_size=1)

#load data previously saved in rec (better for processing)

dataiter = mx.io.ImageRecordIter(
    path_imgrec='/mnt/Data/Data/Kaggle/Yelp/tr.rec',
    data_shape=(3,224,224),
    batch_size=10,
    mean_r=117.0,
    mean_g=117.0,
    mean_b=117.0,
    preprocess_threads=6,
    prefetch_buffer=1)
# to avoid large output file containing all the irrelevant predictions,
# i have sampled photo ids and saved column numbers of 2000 columns
# which have larger std dev in 'col.sel.csv'
col_sel = pd.read_csv(tgt+'col.sel.csv')


first = True
for i in range(NUMBATCH):
  batch = dataiter.next()
  idx = batch.index
  pad = batch.pad
  data = batch.data[0]

  prob = model.predict(data)
  #ok until here, todo understand here
  #todo
  # 1 installation of everything on other computer
  # 2 understand sructure of submissions & everything
  # 3 select which pretrained model is the best one
  # 4 run on yelp
  # 5 run on another
  # 6 wndows azure
  # 7 set time time to make decisions
  # 8 get whole script to run everyting.
  # 9 understand other uses of mxnet
  prob = prob[:, col_sel.col-1]

  if pad > 0:
    N = len(idx) - pad
    idx = idx[:N]
    prob = prob[:N]

  df = pd.concat([pd.DataFrame({'photo_id':idx}), pd.DataFrame(prob, columns=['feat'+str(j+1) for j in range(prob.shape[1])])], axis=1)

  header = first
  mode = 'w' if first else 'a'
  df.to_csv(outfile % SEQ, index=False, header=header, mode=mode)
  first = False

  if (i+1) % 100 == 0:
    print('%d processed: %ds' % ((i+1), time.time() - start_time))

print('done; elapsed = %ds' % (time.time() - start_time))

