from surprise import SVD
from surprise import Dataset
from surprise import evaluate, print_perf
from surprise.dataset import Reader
import pandas as pd

src = 'D:\\Delicieuse\\'
sub = 'sampleDf.csv'
df = pd.read_csv(src+sub, sep=';')
# Load the movielens-100k dataset (download it if needed),
# and split it into 3 folds for cross-validation.
#data = Dataset.load_builtin('ml-100k')
reader = Reader(line_format='user item rating')
data = Dataset.load_from_df(dfStacked, reader)
data.split(n_folds=3)

# We'll use the famous SVD algorithm.
algo = SVD()

# Evaluate performances of our algorithm on the dataset.
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
print_perf(perf)






##### LOAD OF DATA  ########
"""
Replace
"Bosq ft. Nicole Willis "Bad For Me (Late Night Dub)""
"Sade "Give It Up (Vin Sol & Matrixxman Edit)""
"Robosonic - Good Old Feel (12" dub version)"
"Tosca - "Stuttgart" feat. Lucas Santtana (Marlow & Tr\\u00c3\\u00bcby refix)"
"Two Man Sound - Que Tal America (Dj "S" Bootleg Extended Dance Re-Mix)"
"Lake Powel "Bright Eyes, Dirty Hair (Acid Pauli & NU Rmx)" (snippet)"
"Lescop - La Foret (Joakim "Balenciaga" Remix)"
"Chantez Les Gars! 7' Killer Disco Club 45, France 1977"
"Ce Ce Peniston - Finally (12" Choice Mix) "
"Desert Sound Colony "The Way I Began" [clip]"
"Sade "Gi"
"The Nightjar - Flowers on Bare Wood" - Boiler Room Debuts"
"Mac DeMarco "Passing Out Pieces""
"Still Going - "Spaghetti Circus""
"Sufjan Stevens - Should Have Known Better""
"Mac DeMarco "Passing Out Pieces""
"DJ Premier - Classic (Ft. Rakim, Nas,Kanye west,KRS-One)",Kanye west,KRS-One)"


"Free and Easy - Apple and the Three Oranges - I'll Give You A Ring (When I Come]
"Tamba Trio - Mas]
"Still Going - "Spa"]
"Detroit Swindle - You,me,here

""
,"]
, "]

"""

import json
src = 'C:\\Users\\UlysseGIMENEZ\\OneDrive - BioSerenity\\UGIT\\Delicieuse\\users_liked_music_corrected.json'
with open(src) as json_data:
    d = json.load(json_data)


# Get lists
X = dict()
dfStacked = pd.DataFrame(columns=['usr','song','like'])
usrLst = []
musicLst = []
for k in d:
    if k['music'] and k['music'][0] is not None:

        X[k['user']] = np.array(k['music'])
        usrLst.append(k['user'])
        musicLst += k['music']

for i, k in enumerate(X.keys()):
    if i%100 == 0:
        print(i)
    for song in X[k]:
        dfStacked = dfStacked.append(pd.DataFrame(np.array([k,song,1]), index= ['usr','song','like']).T)

#todo a check on names of musics, on nan, on None...

musicLst = list(set(musicLst))
'''
usrLst.remove('1036915825')
usrLst.remove('1066009812')
usrLst.remove('1097633177')
'''

#get df
df = pd.DataFrame(False,index = usrLst, columns=musicLst)
for i,sub in enumerate(usrLst):
    df.loc[sub, X[sub]] = True

musicSum = df.sum()
userSum = df.sum(axis=1)
