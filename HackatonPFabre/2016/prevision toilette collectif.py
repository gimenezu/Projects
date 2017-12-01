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
import seaborn



def RF(df, listFeature,Y):
 # todo perform gridsearch
 train, test = train_test_split(df, test_size=0.3)
 X_train = train[listFeature]
 X_test = test[listFeature]
 Y_train = train[Y]
 Y_test = test[Y].values
 Y_true = Y_test.transpose()
 model = RandomForestClassifier()
 model.fit(X_train, Y_train)
 predicted = model.predict(X_test)

 mse = mean_squared_error(Y_true, predicted)
 r2 = r2_score(Y_true, predicted)

 return model.feature_importances_, predicted,mse
 #return featuresDf, performance,
def plot_importance(importance,Y,DataSrc,titleEnd):
    importance=importance.sort('Importance')
    _COLOR_THEME = 'coolwarm'
    labels = np.array(importance.index.values)
    values = np.array(importance.ix[:, 0].values)

    seaborn.set(style="white", context="talk")
    fig = figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    b = seaborn.barplot(
        labels,
        values,
        ci=None,
        palette=_COLOR_THEME,
        hline=0,
        ax=ax,
        x_order=labels)
    ax.set_xticklabels(labels, rotation=30,fontsize=16)
    title('Influence sur ' + titleEnd, fontsize=22)
    savefig(DataSrc + 'figs\\' + Y + '_predictors.png', dpi=300)


'''
#Data engineering : mettre a jour regulierement toutes les 24 h dans un seul fichier
# 24 h : nouveau chargement des données JSON pour tous les patiens
for patient in ListPatients:

#ajout douleur
#ajout drugs
patients
oxybutynine
tolterodine
Hurt
weight
height
IMC
age
smoking
activity
sex
modele_Name
Performance
Accident
Best_feature_1
Best_feature_2
Best_feature_3
'''

#todo Load from CSV
DataSrc = 'C:\\Users\\UlysseGIMENEZ\\Desktop\\Urotech\\'
data_global=pd.read_csv(DataSrc + 'data_test_fixe.csv',sep='\t', header=0,index_col=0)




# Plot meilleur feature
BestFeat = pd.concat([data_global['Best_feature_1'],data_global['Best_feature_1'],data_global['Best_feature_1'], data_global['Best_feature_2'],data_global['Best_feature_2'] ,data_global['Best_feature_3']])
BestFeat.value_counts()
#todo tracé des bars ici


#Meilleur modele
Best_model = data_global['modele_Name'].value_counts().index[0]
#todo add paremetres, performance ? A qui ?


# Plot selon medicament ?




###### Random Forest Accident  ########
#liste des features ? Ajouter Aliments etc ?

Fimportance, predicted,score  = RF(data_global,
                  ['oxybutynine', 'tolterodine', 'Hurt', 'weight', 'height', 'IMC', 'age', 'smoking',
                   'activity', 'sex'],
                  'Accident'
                  )
importance = pd.DataFrame(Fimportance,
                          index=['oxybutynine', 'tolterodine', 'Hurt', 'weight', 'height', 'IMC', 'age', 'smoking',
                   'activity', 'sex'],
                          columns=["Importance"])

plot_importance(importance,'Accident',DataSrc,'les accidents')



###### Random Forest Hurt  ########
#liste des features ? Ajouter Aliments etc ?



Fimportance, predicted,score = RF(data_global,
                  ['oxybutynine', 'tolterodine', 'weight', 'height', 'IMC', 'age', 'smoking',
                   'activity', 'sex'],
                  'Hurt'
                  )
importance = pd.DataFrame(Fimportance,
                          index=['oxybutynine', 'tolterodine', 'weight', 'height', 'IMC', 'age', 'smoking',
                   'activity', 'sex'],
                          columns=["Importance"])

plot_importance(importance,'Hurt',DataSrc, 'la douleur')






###### Random Forest Performance  ########





#tests
'''
fig = figure()
ax = fig.add_subplot(111)
bar(np.arange(len(importance.ix[:, 0])), importance.ix[:, 0].values, align="center")
xTickMarks = importance.index.values
ax.set_xticks(np.arange(len(importance.ix[:, 0])))
xtickNames = ax.set_xticklabels(xTickMarks)
setp(xtickNames, rotation=30, fontsize=10)
show()
'''