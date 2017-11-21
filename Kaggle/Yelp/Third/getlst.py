import pandas as pd

# read data
tgt = '/mnt/Data/Data/Kaggle/Yelp/third/'
p2b_tr = pd.read_csv('/mnt/Data/Data/Kaggle/Yelp/train_photo_to_biz_ids.csv/train_photo_to_biz_ids.csv')
p2b_te = pd.read_csv('/mnt/Data/Data/Kaggle/Yelp/test_photo_to_biz.csv')

rows = []
for pid in set(p2b_tr.photo_id):
  rowdict = { 'photo_id':pid, 'class':0, 'filename':'%d.jpg' % pid }
  rows.append(rowdict)
df_train = pd.DataFrame(rows)
df_train.to_csv(tgt + 'tr.lst', columns=['photo_id', 'class', 'filename'], sep='\t', header=False, index=False)

rows = []
for pid in set(p2b_te.photo_id):
  rowdict = { 'photo_id':pid, 'class':0, 'filename':'%d.jpg' % pid }
  rows.append(rowdict)
df_test = pd.DataFrame(rows)
df_test.to_csv(tgt + 'te.lst', columns=['photo_id', 'class', 'filename'], sep='\t', header=False, index=False)


