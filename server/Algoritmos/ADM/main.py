import matplotlib.pyplot as plt
import pandas as pd
import StrategyFile as sf
import sys
import string
import os
import geopandas as gpd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import model_selection
from pandas.plotting import scatter_matrix
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from time import time

pedirParametros = int(sys.argv[2])

#Cargamos los datos de un fichero
file = sys.argv[1]
fichero = os.path.splitext(file)
fichero = fichero[0] + ".csv"
nombreFichero = ""

if file.endswith('.csv'):
    fileSelected = sf.Csv(file, fichero)
    df = fileSelected.collect()
elif file.endswith('.json'):
    fileSelected= sf.Json(file, fichero)
    df = fileSelected.collect()
elif file.endswith('.xlsx'):
    fileSelected= sf.Xlsx(file, fichero)
    df = fileSelected.collect()
else:
    print("Formato no soportado")
    sys.exit()


if(pedirParametros == 1):
    algoritmoSeleccionado = int(input('¿Qué algoritmo quiere ejecutar?: \n\t 1. Regresión Lineal. \n\t 2. Árbol de Regresión. \n\t 3. Regresión árbol Aleatorio. \n\t 4. Red Neuronal.\n  > '))
    columnaSeleccionadaInicial = int(input('¿Qué columna inicial quiere analizar?\n > '))
    columnaSeleccionada = int(input('¿Qué columna final quiere analizar?\n > '))
    valoresPredecir = input('¿Qué valores tiene para predecir?\n > ')
else:
    algoritmoSeleccionado = int(sys.argv[3])
    columnaSeleccionadaInicial = int(sys.argv[4])
    columnaSeleccionada = int(sys.argv[5])
    valoresPredecir = sys.argv[6]
    rutaEscribirJson = sys.argv[7]

array = df.values
X = (array[:,columnaSeleccionadaInicial:columnaSeleccionada])
Y = (array[:,columnaSeleccionada])

if algoritmoSeleccionado == 1:
    model = LinearRegression()
elif algoritmoSeleccionado == 2:
    model = DecisionTreeRegressor()
elif algoritmoSeleccionado == 3:
    model = RandomForestRegressor()
elif algoritmoSeleccionado == 4:
    model = MLPRegressor()
else:
    print("El algoritmo introducido no existe")
    sys.exit()

valorSplit = valoresPredecir.split(",")
valorMap = list(map(float, valorSplit))

valoresPredecir = np.array([valorMap])
# valorPredecir = np.array([[1,1,2010,1.00,12.00,5.00,31.00,56.00,0.4,2.00,3.00,2.00]])
reg = model.fit(X, Y)
result = reg.predict(valoresPredecir)

if(pedirParametros == 1):
    print("Result: \n")
    print(result)
else:
    json = '{"Resultado":'+ str(result[0]) +'}'
    file = open(rutaEscribirJson, "w")
    file.write(json)