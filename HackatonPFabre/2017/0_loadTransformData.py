src = 'D:\\Data\\pFabre\\data\\'


#todo
trainLst = os.listdir(src+'Donnees_apprentissage\\Donnees_apprentissage\\')
trainLstSample = os.listdir(src+'TestImg\\')#todo
#testLst

rows = []
for pid in set(trainLst):
    pid = int(pid[:-4])#todo suppress
    rowdict = { 'photo_id':pid, 'class':0, 'filename':'%d.png' % pid }
    rows.append(rowdict)
df_train = pd.DataFrame(rows)
df_train.to_csv(src + 'tr.lst', columns=['photo_id', 'class', 'filename'], sep='\t', header=False, index=False)


rows = []
for pid in set(trainLstSample):
    pid = int(pid[:-4])#todo suppress
    rowdict = { 'photo_id':pid, 'class':0, 'filename':'%d.png' % pid }
    rows.append(rowdict)
df_trainS = pd.DataFrame(rows)
df_trainS.to_csv(src + 'trSample.lst', columns=['photo_id', 'class', 'filename'], sep='\t', header=False, index=False)



rows = []
for pid in set(testLst):
  rowdict = { 'photo_id':pid, 'class':0, 'filename':'%d.jpg' % pid }
  rows.append(rowdict)
df_test = pd.DataFrame(rows)
df_test.to_csv(src + 'te.lst', columns=['photo_id', 'class', 'filename'], sep='\t', header=False, index=False)


#todo on terminal in data folder
# activate pyt3.6.3
# python im2rec.py --resize 224 tr train_photos