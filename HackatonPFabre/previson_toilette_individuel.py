import numpy as np
from matplotlib.pyplot import *
import csv
import datetime
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score



#TODO in LoadLine
DataSrc = 'C:\\Users\\UlysseGIMENEZ\\Desktop\\Urotech\\'
newData=open(DataSrc+'pipi').read()
wholeData=open(DataSrc+'pipi').read()

def loadLine(dataLine):
    Date = 0;
    Hour = 0;
    hurt = 0;
    success = 0;
    Pipi_frequence = 0;
    volume = 0;
    effort = 0;
    Drink_0 = 0;
    Drink_1 = 0;
    Drink_2 = 0;
    Drink_3 = 0;
    success=0;
    hurt=0;
    Aliment_0=0;
    Aliment_1=0;
    Aliment_2=0;
    Aliment_3=0;
    Aliment_4=0;




    if dataLine.find(':"2016-')>=0:Date= dataLine[dataLine.find(':"2016-')+len(':"2016-'):dataLine.find('T')]
    if dataLine.find('T')>0:Hour= int(dataLine[dataLine.find('T')+len('T'):dataLine[dataLine.find('T'):].find(':')+dataLine.find('T')])
    if dataLine.find('"hurt":')>0:hurt = int(dataLine[dataLine.find('"hurt":')+len('"hurt":'):dataLine.find(',"success"')])
    if dataLine.find('"success":')>0:
        success = dataLine[dataLine.find('"success":')+len('"success":'):dataLine.find(',"volume":')]
        success = success[0].upper()+success[1:]
        if success == 'True':
            Pipi_frequence = int(1)
    if dataLine.find('"volume":')>0: Volume = float(dataLine[dataLine.find('"volume":')+len('"volume":'):dataLine.find('}}')])
    if dataLine.find('"effort":')>0:
        effort = dataLine[dataLine.find('"effort":')+len('"effort":'):dataLine.find('}}')]
        effort = effort[0].upper()+effort[1:]
    if dataLine.find('"boissonType":0, "volume":')>0: Drink_0 = float(dataLine[dataLine.find('"boissonType":0, "volume":')+len('"boissonType":0, "volume":'):dataLine.find('}}')])
    if dataLine.find('"boissonType":1, "volume":')>0: Drink_1 = float(dataLine[dataLine.find('"boissonType":1, "volume":')+len('"boissonType":1, "volume":'):dataLine.find('}}')])
    if dataLine.find('"boissonType":2, "volume":')>0: Drink_2 = float(dataLine[dataLine.find('"boissonType":2, "volume":')+len('"boissonType":2, "volume":'):dataLine.find('}}')])
    if dataLine.find('"boissonType":3, "volume":')>0: Drink_3 = float(dataLine[dataLine.find('"boissonType":3, "volume":')+len('"boissonType":3, "volume":'):dataLine.find('}}')])



    df=pd.DataFrame({'Date':[Date],'Hour':[Hour],'hurt':[hurt],'success':[success],'Pipi_frequence':[Pipi_frequence],'Volume':[Volume],
                     'effort':[effort],'Drink_0':[Drink_0],'Drink_1':[Drink_1],'Drink_2':[Drink_2],'Drink_3':[Drink_3],
                     'Aliment_0': [Aliment_0],'Aliment_1': [Aliment_1],'Aliment_2': [Aliment_2],'Aliment_3': [Aliment_3],'Aliment_4': [Aliment_4]})
    return df
def newHourdf(dfLine):
# Data engineering, 1 ligne par heure => frequence
    df_new_Hour = pd.DataFrame()
    df_new_Hour = pd.concat([df_new_Hour,pd.DataFrame(data=np.array([max(0,dfLine['hurt'][dfLine['hurt']>0].mean())]),columns=['hurt'])],axis=1)
    df_new_Hour = pd.concat([df_new_Hour,pd.DataFrame(data=np.array([(dfLine['Hour'].iloc[0]<8 or dfLine['Hour'].iloc[0]>22)*1]),columns=['Nuit'])],axis=1)
    df_new_Hour = pd.concat([df_new_Hour,pd.DataFrame(data=np.array([dfLine['Hour'].iloc[0]]),columns=['Hour'])],axis=1)
    df_new_Hour = pd.concat([df_new_Hour,pd.DataFrame(data=np.array([dfLine['Date'].iloc[0]]),columns=['Date'])],axis=1)
    df_new_Hour = pd.concat([df_new_Hour,pd.DataFrame(data=np.array([max(0,dfLine['Drink_0'][dfLine['Drink_0']>0].mean())]),columns=['Drink_0'])],axis=1)
    df_new_Hour = pd.concat([df_new_Hour,pd.DataFrame(data=np.array([max(0,dfLine['Drink_1'][dfLine['Drink_1']>0].mean())]),columns=['Drink_1'])],axis=1)
    df_new_Hour = pd.concat([df_new_Hour,pd.DataFrame(data=np.array([max(0,dfLine['Drink_2'][dfLine['Drink_2']>0].mean())]),columns=['Drink_2'])],axis=1)
    df_new_Hour = pd.concat([df_new_Hour,pd.DataFrame(data=np.array([max(0,dfLine['Drink_3'][dfLine['Drink_3']>0].mean())]),columns=['Drink_3'])],axis=1)
    df_new_Hour = pd.concat([df_new_Hour,pd.DataFrame(data=np.array([dfLine['Volume'].iloc[-1]]),columns=['Volume'])],axis=1)
    df_new_Hour = pd.concat([df_new_Hour,pd.DataFrame(data=np.array([dfLine['Pipi_frequence'].sum()]),columns=['Pipi_frequence'])],axis=1)
    df_new_Hour = pd.concat([df_new_Hour,pd.DataFrame(data=np.array([dfLine['effort'][dfLine['effort']==True].sum()]),columns=['accident_effort'])],axis=1)
    df_new_Hour = pd.concat([df_new_Hour,pd.DataFrame(data=np.array([dfLine['effort'][dfLine['effort']==False].sum()]),columns=['accident_passif'])],axis=1)
    df_new_Hour = pd.concat([df_new_Hour,pd.DataFrame(data=np.array([dfLine['success'][dfLine['success']==True].sum()]),columns=['success'])],axis=1)
    df_new_Hour = pd.concat([df_new_Hour,pd.DataFrame(data=np.array([dfLine['success'][dfLine['success']==False].sum()]),columns=['non_Predicted'])],axis=1)
    df_new_Hour = pd.concat([df_new_Hour,pd.DataFrame(data=np.array([max(0,dfLine['Aliment_0'][dfLine['Aliment_0']>0].mean())]),columns=['Aliment_0'])],axis=1)
    df_new_Hour = pd.concat([df_new_Hour,pd.DataFrame(data=np.array([max(0,dfLine['Aliment_1'][dfLine['Aliment_1']>0].mean())]),columns=['Aliment_1'])],axis=1)
    df_new_Hour = pd.concat([df_new_Hour,pd.DataFrame(data=np.array([max(0,dfLine['Aliment_2'][dfLine['Aliment_2']>0].mean())]),columns=['Aliment_2'])],axis=1)
    df_new_Hour = pd.concat([df_new_Hour,pd.DataFrame(data=np.array([max(0,dfLine['Aliment_3'][dfLine['Aliment_3']>0].mean())]),columns=['Aliment_3'])],axis=1)
    df_new_Hour = pd.concat([df_new_Hour,pd.DataFrame(data=np.array([max(0,dfLine['Aliment_4'][dfLine['Aliment_4']>0].mean())]),columns=['Aliment_4'])],axis=1)
    return df_new_Hour
def RF(df, listFeature,Y,df2):
 train, test = train_test_split(df, test_size=0.3)
 X_train = train[listFeature]
 X_test = test[listFeature]
 Y_train = train[Y]
 Y_test = test[Y].values
 Y_true = Y_test.transpose()
 model = RandomForestClassifier()
 model.fit(X_train, Y_train)
 predicted = model.predict(X_test)

#todo change
 score_normalized = mean_squared_error(Y_true, predicted)
 r2 = r2_score(Y_true, predicted)
 z=model.predict(df2)

 return z*score_normalized
 #return featuresDf, performance,




#todo tout d'un coup, il faut séparer les heures
# Nouvelle heure chargement des donnees JSON, 1 fois par heure
data_online = pd.DataFrame()
actualHour = wholeData.split('{"date"')[1][13:15]
dfLine = loadLine(wholeData.split('{"date"')[1])
for line in wholeData.split('{"date"')[2:]:
    if line[13:15]==actualHour:
        dfLine = pd.concat([dfLine, loadLine(line)], axis=0)
    else:
        data_online = pd.concat([data_online, newHourdf(dfLine)])
        dfLine=loadLine(line)
        actualHour=line[13:15]



'''
newData=open(DataSrc+'pipi').read()
dfLine = pd.DataFrame()
for dataLine in newData.split('{"date"')[1:]:
    dfLine = pd.concat([dfLine,loadLine(dataLine)],axis=0)






#todo suppress
DataSrc = 'C:\\Users\\UlysseGIMENEZ\\Desktop\\Urotech\\'
data_online=pd.read_csv(DataSrc + 'data_test_online.csv',sep=';', header=0,index_col=0)
data_online=pd.concat([data_online,df_new_Hour])
#data_online.to_csv(DataSrc + 'data_test_online.csv',sep=';')
'''



#data_online=pd.read_csv(DataSrc + 'data_test_online.csv',sep=';', header=0,index_col=0)
# Prendre en compte les 3 dernieres heures
data_online_3h = pd.DataFrame()
for k in np.arange(len(data_online))[3:]:
    data_online_3hTemp = pd.concat([data_online.iloc[k-2].to_frame().T,data_online.iloc[k-1].to_frame().T,data_online.iloc[k].to_frame().T],axis=1)
    data_online_3hTemp.columns = np.concatenate((data_online.columns.values + np.array(['_H2']),data_online.columns.values + np.array(['_H1']),data_online.columns.values))
    data_online_3h = pd.concat([data_online_3h,data_online_3hTemp],axis=0)
data_online_3h.index=data_online_3h['Date']


#to apply
#A = pd.DataFrame({'Pipi_+2':data_online_3h['Pipi_frequence'].iloc[2:-1]})


df = data_online_3h.iloc[:-3].copy()
'''
A = pd.DataFrame({'Pipi_+1':data_online_3h['Pipi_frequence'].iloc[1:-2]})
A.index=df.index
df = pd.concat([df,A],axis=1)
A = pd.DataFrame({'Pipi_+2':data_online_3h['Pipi_frequence'].iloc[2:-1]})
A.index=df.index
df = pd.concat([df,A],axis=1)
A = pd.DataFrame({'Pipi_+3':data_online_3h['Pipi_frequence'].iloc[3:]})
A.index=df.index
df = pd.concat([df,A],axis=1)
'''
df = pd.concat([df,pd.DataFrame(data=data_online_3h['Pipi_frequence'].iloc[1:-2].values,columns=['Pipi_+1'],index=df.index)],axis=1)
df = pd.concat([df,pd.DataFrame(data=data_online_3h['Pipi_frequence'].iloc[2:-1].values,columns=['Pipi_+2'],index=df.index)],axis=1)
df = pd.concat([df,pd.DataFrame(data=data_online_3h['Pipi_frequence'].iloc[3:].values,columns=['Pipi_+3'],index=df.index)],axis=1)

df['Pipi_+1']=pd.to_numeric(df['Pipi_+1'])
df['Pipi_+2']=pd.to_numeric(df['Pipi_+2'])
df['Pipi_+3']=pd.to_numeric(df['Pipi_+3'])


listFeature=['Nuit', 'Aliment_0','Aliment_1', 'Aliment_2', 'Aliment_3', 'Aliment_4',
    'Drink_0', 'Drink_1', 'Drink_2', 'Drink_3','Volume','Pipi_frequence',
    'Nuit_H1', 'Aliment_0_H1','Aliment_1_H1', 'Aliment_2_H1', 'Aliment_3_H1', 'Aliment_4_H1',
    'Drink_0_H1', 'Drink_1_H1', 'Drink_2_H1', 'Drink_3', 'Volume', 'Pipi_frequence',
    'Nuit_H2', 'Aliment_0_H2','Aliment_1_H2', 'Aliment_2_H2', 'Aliment_3_H2', 'Aliment_4_H2',
    'Drink_0_H2', 'Drink_1_H2', 'Drink_2_H2', 'Drink_3_H2', 'Volume_H2', 'Pipi_frequence_H2'
    ]


z1 = RF(df, listFeature,'Pipi_+1',data_online_3h.iloc[-1:][listFeature])
z2 = RF(df, listFeature,'Pipi_+2',data_online_3h.iloc[-1:][listFeature])
z3 = RF(df, listFeature,'Pipi_+3',data_online_3h.iloc[-1:][listFeature])


uroPred=dict()
uroPred['H1']=int(z1)
uroPred['H2']=int(z2)
uroPred['H3']=int(z3)
