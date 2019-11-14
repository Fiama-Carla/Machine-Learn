# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 17:09:23 2019

@author: Fiama Carla
"""
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.svm import NuSVC
from sklearn.ensemble import RandomForestClassifier



dados = load_breast_cancer()

#Divisão da amostra em conj treino e conj teste
train, test, train_labels, test_labels = train_test_split(dados['data'], dados['target'], test_size=0.3, random_state=0)

#Algoritmo 1
gnb = GaussianNB()  
gnb.fit(train, train_labels)
predicted = gnb.predict(test)

print("Relatório de classificação para classificador GaussianNB \n  %s:\n%s\n" % (gnb, metrics.classification_report(test_labels, predicted)))

#Algoritmo 2
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(train, train_labels)
predicted1 = kmeans.predict(test)
print("Relatório de classificação para classificador KMeans \n  %s:\n%s\n" % (KMeans, metrics.classification_report(test_labels, predicted1)))

#Algoritmo 3
clf = NuSVC(gamma='scale')
clf.fit(train, train_labels)
predicted2 = clf.predict(test)
print("Relatório de classificação para classificador NuSVC \n  %s:\n%s\n" % (clf, metrics.classification_report(test_labels, predicted2)))

#Algoritmo 4
classifier = RandomForestClassifier(n_estimators=10)
classifier.fit(train, train_labels)
predicted3 = classifier.predict(test)
print("Relatório de classificação para classificador RandomForest \n  %s:\n%s\n" % (classifier, metrics.classification_report(test_labels, predicted3)))
print(dados['DESCR'])