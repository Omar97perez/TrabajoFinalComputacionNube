#-*- coding: utf-8-*-

import matplotlib.pyplot as plt
import pandas as pd
import StrategyFile as sf
import StrategyAlgorithm as st
import sys
import string 
import os
import geopandas as gpd
import numpy as np
from sklearn import datasets, linear_model, model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris, make_blobs
from sklearn.linear_model import LinearRegression, RANSACRegressor, LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler

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
    algoritmoSeleccionado = int(input('¿Qué algoritmo quiere ejecutar?: \n\t 1. Clasificación Bayesiana (Aprendizaje supervisado de clasificación). \n\t 2. Decision Tree Regression (Aprendizaje supervisado de Regresión). \n\t 3. Mean Shift (Aprendizaje no supervisado basado en Clustering). \n  > '))
    columnaSeleccionadaInicial = int(input('¿Qué columna inicial quiere analizar?\n > '))
    columnaSeleccionada = int(input('¿Qué columna final quiere analizar?\n > '))
else:
    algoritmoSeleccionado = int(sys.argv[3]) 
    columnaSeleccionadaInicial = int(sys.argv[4])
    columnaSeleccionada = int(sys.argv[5])
    nombreFichero = sys.argv[6]

array = df.values
X = (array[:,columnaSeleccionadaInicial:columnaSeleccionada])
Y = (array[:,columnaSeleccionada])

if algoritmoSeleccionado == 1:
    graficaFinal = st.BR(array, X, Y, pedirParametros, nombreFichero)
    graficaFinal.grafica()
elif algoritmoSeleccionado == 2:
    graficaFinal= st.DecisionTreeRegression(array, X, Y, pedirParametros, nombreFichero)
    graficaFinal.grafica()
elif algoritmoSeleccionado == 3:
    graficaFinal = st.MeanShift(array, X, Y, pedirParametros, nombreFichero)
    graficaFinal.grafica()
elif algoritmoSeleccionado == 4:
  graficaFinal= st.LinearRegresion(array, X, Y, pedirParametros, nombreFichero)
  graficaFinal.grafica()
elif algoritmoSeleccionado == 5:
  graficaFinal= st.RandomForestRegressor(array, X, Y, pedirParametros, nombreFichero)
  graficaFinal.grafica()
elif algoritmoSeleccionado == 6:
  graficaFinal= st.KFold(array,X, Y, pedirParametros, nombreFichero)
  graficaFinal.grafica()
elif algoritmoSeleccionado == 7:
  graficaFinal= st.ComparativaRegresion(array, X, Y, pedirParametros, nombreFichero)
  graficaFinal.grafica()
elif algoritmoSeleccionado == 8:
  graficaFinal= st.ComparativaClasificadores(array, X, Y, pedirParametros, nombreFichero)
  graficaFinal.grafica()
elif algoritmoSeleccionado == 9:
  graficaFinal= st.AgglomerativeClustering(array, X, Y, pedirParametros, nombreFichero)
  graficaFinal.grafica()
elif algoritmoSeleccionado == 10:
  graficaFinal= st.ComparativeClustering(array, X, Y, pedirParametros, nombreFichero)
  graficaFinal.grafica()
elif algoritmoSeleccionado == 11:
  graficaFinal= st.DBSCAN(array, X, Y, pedirParametros, nombreFichero)
  graficaFinal.grafica()
elif algoritmoSeleccionado == 12:
  graficaFinal= st.GaussianProcessClassifier(array, X, Y, pedirParametros, nombreFichero)
  graficaFinal.grafica()
else:
    print("El algoritmo introducido no existe")