#Cargando librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
import warnings
import json
warnings.filterwarnings('ignore')

# Preprocesamiento
from sklearn.preprocessing import StandardScaler

#clasificacion
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier #knn
from sklearn.neural_network import MLPClassifier #MLP
from sklearn.svm import SVC #SVM

# Evaluacion de resultados
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# Leemos los datos clean
pimaClean = pd.read_csv("train_clean.csv")

# Seleccionamos todos los datos de los atributos
X = pimaClean.iloc[:,0:250]

# Seleccionamos todos los datos de las clases
Y = pimaClean.iloc[:,250]

# Estandarizamos
rescaledX = StandardScaler().fit_transform(X)
Xnuevo = pd.DataFrame(data = rescaledX, columns= X.columns)

# Leemos los datos test
pimaTest = pd.read_csv("train_clean.csv")

# Separando variables de Clase
XT = pimaTest.iloc[:, 0:250] # variables
YT = pimaTest.iloc[:,250] # clase

#estandarizamos
rescaledXT = StandardScaler().fit_transform(XT)
XTnuevo = pd.DataFrame(data = rescaledXT, columns=XT.columns)

#split
XT_train, XT_test, YT_train, YT_test = train_test_split(XTnuevo, YT, test_size = 0.16)

X_train = Xnuevo
Y_train = Y
X_test = XT_test
Y_test = YT_test
np.savetxt("pruebas/x_train_2.csv", X_train, delimiter=",")
np.savetxt("pruebas/x_test_2.csv", X_test, delimiter=",")
np.savetxt("pruebas/y_train_2.csv", Y_train, delimiter=",")
np.savetxt("pruebas/y_test_2.csv", Y_test, delimiter=",")

#Data Almacenar
data = {}
data["SVM"]=[]
data["KNN"]=[]
data["MLP"]=[]

models = []

# C: parametro de regularizacion del error
# gamma: coeficiente del kernel para rbf y poli
# degree: grado de la funcion polinomial del kernel
models.append(SVC(kernel = 'poly', degree=1, C = 0.05))
print("SVM")
models.append(KNeighborsClassifier(n_neighbors=3)) #instancia KNN
print("KNN")
models.append(MLPClassifier(solver='adam', hidden_layer_sizes=(3, 15), max_iter=500))  #instancia MLP
print("MLP")
#ejecutando clasificacion
np.set_printoptions(threshold=np.nan)
counter = 0
for mod in models:
    mod.fit(X_train, Y_train)
    a = mod.predict(X_test)
    if counter == 0:
        data["SVM"] = pd.Series(a).to_json(orient='values')
    elif counter == 1:
        data["KNN"] = pd.Series(a).to_json(orient='values')
    else:
        data["MLP"] = pd.Series(a).to_json(orient='values')
    counter = counter + 1
    
with open('data/data_2.json', 'w') as outfile:
    json.dump(data, outfile)


    
    
    









