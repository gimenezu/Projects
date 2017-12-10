
P = {
    '0':'Ulysse',
    '1': 'Sophie',
    '2': 'Olivier',
    '3': 'Léa',
    '4': 'Philou',
    '5': 'Capucine',
    '6': 'Margot',
    '7': 'Leslie',
    '8': 'Caro',
    '9': 'Fabien'}
inc = {
    '0': '1' ,
    '1': '0' ,
    '3': '4' ,
    '4': '3' ,
    '8': '9' ,
}


def getValue(badlst):
    k = int(10*np.random.random())
    if str(k) not in badlst:
        return k
    else:
        return getValue(badlst)


Cad = {}
lstOut= []
badlst= []
for i in np.arange(10):
    badlst = lstOut
    try:
        badlst = badlst + [inc[str(i)]] + [str(i)]
    except KeyError:
        pass

    k = getValue(badlst)
    Cad[str(i)]= str(k)
    lstOut = lstOut + [str(k)]


for (key, value) in Cad.items():
    print( P[str(key)] + ' fait un cadeau à ' + P[str(value)] )


