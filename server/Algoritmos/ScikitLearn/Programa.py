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
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs


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

if algoritmoSeleccionado == 1:
    X = (array[:,columnaSeleccionadaInicial:columnaSeleccionada])
    Y = (array[:,columnaSeleccionada])
    graficaFinal = st.BR(X, Y, pedirParametros, nombreFichero)
    graficaFinal.grafica()
elif algoritmoSeleccionado == 2:
    X = (array[:,columnaSeleccionadaInicial:columnaSeleccionada])
    Y = (array[:,columnaSeleccionada])
    graficaFinal= st.DecisionTreeRegression(X, Y, pedirParametros, nombreFichero)
    graficaFinal.grafica()
elif algoritmoSeleccionado == 3:
    X = (array[:,columnaSeleccionadaInicial:columnaSeleccionada])
    ms = MeanShift(bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print("number of estimated clusters : %d" % n_clusters_)

    import matplotlib.pyplot as plt
    from itertools import cycle

    plt.figure(1)
    plt.clf()

    colors = cycle('bgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=14)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    if nombreFichero:
      plt.savefig(nombreFichero)
    else:
      plt.show()
    # graficaFinal = st.MeanShift(X, "", pedirParametros)
    # graficaFinal.grafica()
elif algoritmoSeleccionado == 4:
    X = (array[:,columnaSeleccionadaInicial:columnaSeleccionada])
    Y = (array[:,columnaSeleccionada])
    graficaFinal= st.LinearRegresion(X, Y, pedirParametros, nombreFichero)
    graficaFinal.grafica()
elif algoritmoSeleccionado == 5:
    X = (array[:,columnaSeleccionadaInicial:columnaSeleccionada])
    Y = (array[:,columnaSeleccionada])
    validation_size = 0.22
    seed = 123
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,max_features='sqrt', max_leaf_nodes=None)
    kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
    msg = "%s (%f)" % ('Random Forest Regressor', cv_results.mean())

    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)

    fig, ax = plt.subplots()
    fig.suptitle( msg)
    ax.scatter(Y_validation, predictions, edgecolors=(0, 0, 0))
    ax.plot([Y_validation.min(), Y_validation.max()], [Y_validation.min(), Y_validation.max()], 'k--', lw=2)
    ax.set_xlabel('Medido')
    ax.set_ylabel('Predecido')
    if nombreFichero:
      plt.savefig(nombreFichero)
    else:
      plt.show()

    if(pedirParametros == 1):
        fig = plt.figure()
        fig.suptitle('Diagrama de Cajas y Bigotes para Decision Tree Regression')
        ax = fig.add_subplot(111)
        plt.boxplot(cv_results)
        ax.set_xticklabels('BR')
        plt.show()
elif algoritmoSeleccionado == 6:
    X = (array[:,columnaSeleccionadaInicial:columnaSeleccionada])
    Y = (array[:,columnaSeleccionada])
    validation_size = 0.22
    seed = 123
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    model = MLPRegressor()
    kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
    msg = "%s (%f)" % ('Red Neuronal', cv_results.mean())

    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)

    fig, ax = plt.subplots()
    fig.suptitle( msg)
    ax.scatter(Y_validation, predictions, edgecolors=(0, 0, 0))
    ax.plot([Y_validation.min(), Y_validation.max()], [Y_validation.min(), Y_validation.max()], 'k--', lw=2)
    ax.set_xlabel('Medido')
    ax.set_ylabel('Predecido')
    if nombreFichero:
      plt.savefig(nombreFichero)
    else:
      plt.show()

    if(pedirParametros == 1):
        fig = plt.figure()
        fig.suptitle('Diagrama de Cajas y Bigotes para Decision Tree Regression')
        ax = fig.add_subplot(111)
        plt.boxplot(cv_results)
        ax.set_xticklabels('BR')
        plt.show()
else:
    print("El algoritmo introducido no existe")