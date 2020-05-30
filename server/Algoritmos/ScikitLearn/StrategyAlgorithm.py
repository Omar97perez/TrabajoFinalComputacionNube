#-*- coding: utf-8-*

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
from itertools import cycle

class Algorithm:
  def __init__(self, array=None, X=None,Y=None,pedirParametros=None, nombreFichero=None):
    self.array = array
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
    kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
    msg = "%s (%f)" % ('Clasificador Bayesiano', cv_results.mean())

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
    kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
    msg = "%s (%f)" % ('Decision Tree Regression', cv_results.mean())

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
    ms.fit(self.X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

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

class GaussianProcessClassifier(Algorithm):
  def grafica(self):
    n_features = self.X.shape[1]
    C = 10
    kernel = 1.0 * RBF([1.0, 1.0])

    # Create different classifiers.
    classifiers = {
      'GPC': GaussianProcessClassifier(kernel)
    }

    # n_classifiers = len(classifiers)

    # plt.figure(figsize=(3 * 2, n_classifiers * 2))
    # plt.subplots_adjust(bottom=.2, top=.95)

    # xx = np.linspace(3, 9, 100)
    # yy = np.linspace(1, 5, 100).T
    # xx, yy = np.meshgrid(xx, yy)
    # Xfull = np.c_[xx.ravel(), yy.ravel()]

    # for index, (name, classifier) in enumerate(classifiers.items()):
    #     classifier.fit(self.X, self.y)

    #     y_pred = classifier.predict(X)
    #     accuracy = accuracy_score(y, y_pred)
    #     print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))

    #     # View probabilities:
    #     probas = classifier.predict_proba(Xfull)
    #     n_classes = np.unique(y_pred).size
    #     for k in range(n_classes):
    #         plt.subplot(n_classifiers, n_classes, index * n_classes + k + 1)
    #         plt.title("Class %d" % k)
    #         if k == 0:
    #             plt.ylabel(name)
    #         imshow_handle = plt.imshow(probas[:, k].reshape((100, 100)),
    #                                   extent=(3, 9, 1, 5), origin='lower')
    #         plt.xticks(())
    #         plt.yticks(())
    #         idx = (y_pred == k)
    #         if idx.any():
    #             plt.scatter(X[idx, 0], X[idx, 1], marker='o', c='w', edgecolor='k')

    # ax = plt.axes([0.15, 0.04, 0.7, 0.05])
    # plt.title("Probability")
    # plt.colorbar(imshow_handle, cax=ax, orientation='horizontal')

    # if nombreFichero:
    #   plt.savefig(nombreFichero)
    # else:
    #   plt.show()

class DBSCAN(Algorithm):
  def grafica(self):
    labels_true = 1

    X = StandardScaler().fit_transform(self.X)

    # #############################################################################
    # Compute DBSCAN
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    # #############################################################################
    # Plot result
    import matplotlib.pyplot as plt

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    if nombreFichero:
      plt.savefig(nombreFichero)
    else:
      plt.show()

class KFold(Algorithm):
  def grafica(self):
    validation_size = 0.22
    seed = 123
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(self.X, self.Y, test_size=validation_size, random_state=seed)
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

class ComparativaRegresion(Algorithm):
  def grafica(self):
    validation_size = 0.22
    seed = 123
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(self.X, self.Y, test_size=validation_size, random_state=seed)
    models = []

    models.append(('LR', LinearRegression()))

    models.append(('DTR', DecisionTreeRegressor()))

    models.append(('RF',RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
            max_features='auto', max_leaf_nodes=None,
            )))

    models.append(('RF(LOG)',RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
            max_features='log2', max_leaf_nodes=None,
            )))

    models.append(('RF(Sqrt)',RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
            max_features='sqrt', max_leaf_nodes=None,
            )))

    models.append(('RF(4)',RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
            max_features=4, max_leaf_nodes=None,
            )))

    models.append(('NN',MLPRegressor()))

    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    fig = plt.figure()
    fig.suptitle('Comparacion de los algoritmos')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    
    if nombreFichero:
      plt.savefig(nombreFichero)
    else:
      plt.show()

class ComparativaClasificadores(Algorithm):
  def grafica(self):
    n_features = X.shape[1]
    C = 10
    kernel = 1.0 * RBF([1.0, 1.0])  # for GPC

    # Create different classifiers.
    classifiers = {
        'L1 logistic': LogisticRegression(C=C, penalty='l1',
                                          solver='saga',
                                          multi_class='multinomial',
                                          max_iter=10000),
        'L2 logistic (Multinomial)': LogisticRegression(C=C, penalty='l2',
                                                        solver='saga',
                                                        multi_class='multinomial',
                                                        max_iter=10000),
        'L2 logistic (OvR)': LogisticRegression(C=C, penalty='l2',
                                                solver='saga',
                                                multi_class='ovr',
                                                max_iter=10000),
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,
                          random_state=0),
        'GPC': GaussianProcessClassifier(kernel)
    }

    n_classifiers = len(classifiers)

    plt.figure(figsize=(3 * 2, n_classifiers * 2))
    plt.subplots_adjust(bottom=.2, top=.95)

    xx = np.linspace(3, 9, 100)
    yy = np.linspace(1, 5, 100).T
    xx, yy = np.meshgrid(xx, yy)
    Xfull = np.c_[xx.ravel(), yy.ravel()]

    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X, y)

        y_pred = classifier.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))

        # View probabilities:
        probas = classifier.predict_proba(Xfull)
        n_classes = np.unique(y_pred).size
        for k in range(n_classes):
            plt.subplot(n_classifiers, n_classes, index * n_classes + k + 1)
            plt.title("Class %d" % k)
            if k == 0:
                plt.ylabel(name)
            imshow_handle = plt.imshow(probas[:, k].reshape((100, 100)),
                                      extent=(3, 9, 1, 5), origin='lower')
            plt.xticks(())
            plt.yticks(())
            idx = (y_pred == k)
            if idx.any():
                plt.scatter(X[idx, 0], X[idx, 1], marker='o', c='w', edgecolor='k')

    ax = plt.axes([0.15, 0.04, 0.7, 0.05])
    plt.title("Probability")
    plt.colorbar(imshow_handle, cax=ax, orientation='horizontal')

    if nombreFichero:
      plt.savefig(nombreFichero)
    else:
      plt.show()

class AgglomerativeClustering(Algorithm):
  def grafica(self):
    def plot_dendrogram(model, **kwargs):
      counts = np.zeros(model.children_.shape[0])
      n_samples = len(model.labels_)
      for i, merge in enumerate(model.children_):
          current_count = 0
          for child_idx in merge:
              if child_idx < n_samples:
                  current_count += 1  
              else:
                  current_count += counts[child_idx - n_samples]
          counts[i] = current_count

      linkage_matrix = np.column_stack([model.children_, model.distances_,
                                        counts]).astype(float)

      dendrogram(linkage_matrix, **kwargs)

    X = self.array
    model = AgglomerativeClustering()
    model.distance_threshold(0)
    model.n_clusters(None)
    model = model.fit(X)
    plt.title('Hierarchical Clustering Dendrogram')
    plot_dendrogram(model, truncate_mode='level', p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    if nombreFichero:
      plt.savefig(nombreFichero)
    else:
      plt.show()

class ComparativeClustering(Algorithm):
  def grafica(self):
    plt.figure(figsize=(9, 3))
    plt.subplot(131)
    
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

    colors = cycle('bgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=14)
    plt.title('Estimated number of clusters: %d' % n_clusters_)

    plt.subplot(132)

    def plot_dendrogram(model, **kwargs):
      # Create linkage matrix and then plot the dendrogram

      # create the counts of samples under each node
      counts = np.zeros(model.children_.shape[0])
      n_samples = len(model.labels_)
      for i, merge in enumerate(model.children_):
          current_count = 0
          for child_idx in merge:
              if child_idx < n_samples:
                  current_count += 1  # leaf node
              else:
                  current_count += counts[child_idx - n_samples]
          counts[i] = current_count

      linkage_matrix = np.column_stack([model.children_, model.distances_,
                                        counts]).astype(float)

      # Plot the corresponding dendrogram
      dendrogram(linkage_matrix, **kwargs)


    iris = array
    X = iris.data

    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

    model = model.fit(X)
    plt.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, truncate_mode='level', p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")

    plt.subplot(133)

    import numpy as np

    from sklearn.cluster import DBSCAN
    from sklearn import metrics
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler
    # #############################################################################
    # Generate sample data
    X = (array[:,columnaSeleccionadaInicial:columnaSeleccionada])

    labels_true = 1

    X = StandardScaler().fit_transform(X)

    # #############################################################################
    # Compute DBSCAN
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    # #############################################################################
    # Plot result
    import matplotlib.pyplot as plt

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)

    if nombreFichero:
      plt.savefig(nombreFichero)
    else:
      plt.show()