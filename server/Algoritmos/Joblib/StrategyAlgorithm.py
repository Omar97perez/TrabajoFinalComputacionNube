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
from time import time
from joblib import parallel_backend

n_jobs_parrallel=3

class Algorithm:
    def __init__(self, X,Y,pedirParametros, nombreFichero):
      self.X = X
      self.Y = Y
      self.pedirParametros = pedirParametros
      self.nombreFichero = nombreFichero

class BR(Algorithm):
  def grafica(self):
    validation_size = 0.22
    seed = 123
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(self.X, self.Y, test_size=validation_size, random_state=seed)
    model = linear_model.BayesianRidge()

    start_time = time()
    with parallel_backend('threading', n_jobs=n_jobs_parrallel):
      kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
    elapsed_time = time() - start_time
    elapsed_time = format(elapsed_time, '.8f')
    salida = 'Tiempo ejecución:' + str(elapsed_time) + ' segundos'
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
    msg = 'Clasificador Bayesiano ' + '(' + str(format(cv_results.mean(),'.4f')) + ') \n' +  salida

    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)

    fig, ax = plt.subplots()
    fig.suptitle( msg)
    ax.scatter(Y_validation, predictions, edgecolors=(0, 0, 0))
    ax.plot([Y_validation.min(), Y_validation.max()], [Y_validation.min(), Y_validation.max()], 'k--', lw=2)
    ax.set_xlabel('Medido')
    ax.set_ylabel('Predecido')
    if self.nombreFichero:
      plt.savefig(self.nombreFichero)
    else:
      plt.show()

    if(self.pedirParametros == 1):
        fig = plt.figure() 
        fig.suptitle('Diagrama de Cajas y Bigotes para BR')
        ax = fig.add_subplot(111)
        plt.boxplot(cv_results)
        ax.set_xticklabels('BR')
        plt.show()

class DecisionTreeRegression(Algorithm):
  def grafica(self):
    validation_size = 0.22
    seed = 123
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(self.X, self.Y, test_size=validation_size, random_state=seed)
    model = DecisionTreeRegressor()
    start_time = time()
    with parallel_backend('threading', n_jobs=n_jobs_parrallel):
      kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
    elapsed_time = time() - start_time
    elapsed_time = format(elapsed_time, '.8f')
    salida = 'Tiempo ejecución:' + str(elapsed_time) + ' segundos'
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
    msg = 'Árbol de decisión ' + '(' + str(format(cv_results.mean(),'.4f')) + ') \n' +  salida

    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)

    fig, ax = plt.subplots()
    fig.suptitle( msg)
    ax.scatter(Y_validation, predictions, edgecolors=(0, 0, 0))
    ax.plot([Y_validation.min(), Y_validation.max()], [Y_validation.min(), Y_validation.max()], 'k--', lw=2)
    ax.set_xlabel('Medido')
    ax.set_ylabel('Predecido')
    if self.nombreFichero:
      plt.savefig(self.nombreFichero)
    else:
      plt.show()

    if(self.pedirParametros == 1):
        fig = plt.figure()
        fig.suptitle('Diagrama de Cajas y Bigotes para Decision Tree Regression')
        ax = fig.add_subplot(111)
        plt.boxplot(cv_results)
        ax.set_xticklabels('BR')
        plt.show()
        
class MeanShift(Algorithm):
  def grafica(self):
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
    if self.nombreFichero:
      plt.savefig(self.nombreFichero)
    else:
      plt.show()

class LinearRegresion(Algorithm):
  def grafica(self):
    validation_size = 0.22
    seed = 123
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(self.X, self.Y, test_size=validation_size, random_state=seed)
    model = LinearRegression()
    start_time = time()
    with parallel_backend('threading', n_jobs=n_jobs_parrallel):
      kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
    elapsed_time = time() - start_time
    elapsed_time = format(elapsed_time, '.8f')
    salida = 'Tiempo ejecución:' + str(elapsed_time) + ' segundos'
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
    msg = 'Regresión Lineal ' + '(' + str(format(cv_results.mean(),'.4f')) + ') \n' +  salida

    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)

    fig, ax = plt.subplots()
    fig.suptitle( msg)
    ax.scatter(Y_validation, predictions, edgecolors=(0, 0, 0))
    ax.plot([Y_validation.min(), Y_validation.max()], [Y_validation.min(), Y_validation.max()], 'k--', lw=2)
    ax.set_xlabel('Medido')
    ax.set_ylabel('Predecido')
    if self.nombreFichero:
      plt.savefig(self.nombreFichero)
    else:
      plt.show()

    if(self.pedirParametros == 1):
        fig = plt.figure()
        fig.suptitle('Diagrama de Cajas y Bigotes para Decision Tree Regression')
        ax = fig.add_subplot(111)
        plt.boxplot(cv_results)
        ax.set_xticklabels('BR')
        plt.show()

class RandomForestRegressor(Algorithm):
  def grafica(self):
    validation_size = 0.22
    seed = 123
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(self.X, self.Y, test_size=validation_size, random_state=seed)
    model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,max_features='auto', max_leaf_nodes=None)
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
    if self.nombreFichero:
      plt.savefig(self.nombreFichero)
    else:
      plt.show()

    if(self.pedirParametros == 1):
        fig = plt.figure()
        fig.suptitle('Diagrama de Cajas y Bigotes para Decision Tree Regression')
        ax = fig.add_subplot(111)
        plt.boxplot(cv_results)
        ax.set_xticklabels('BR')
        plt.show()

class MLPRegressor(Algorithm):
  def grafica(self):
    validation_size = 0.22
    seed = 123
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(self.X, self.Y, test_size=validation_size, random_state=seed)
    model = MLPRegressor()
    kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
    msg = "%s (%f)" % ('Linear Regression', cv_results.mean())

    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)

    fig, ax = plt.subplots()
    fig.suptitle( msg)
    ax.scatter(Y_validation, predictions, edgecolors=(0, 0, 0))
    ax.plot([Y_validation.min(), Y_validation.max()], [Y_validation.min(), Y_validation.max()], 'k--', lw=2)
    ax.set_xlabel('Medido')
    ax.set_ylabel('Predecido')
    if self.nombreFichero:
      plt.savefig(self.nombreFichero)
    else:
      plt.show()

    if(self.pedirParametros == 1):
        fig = plt.figure()
        fig.suptitle('Diagrama de Cajas y Bigotes para Decision Tree Regression')
        ax = fig.add_subplot(111)
        plt.boxplot(cv_results)
        ax.set_xticklabels('BR')
        plt.show()