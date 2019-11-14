# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 18:59:55 2019

@author: Fiama Carla
"""

import matplotlib.pyplot as plt

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

#Parte 1
algoritmo = ['SVM', 'KNeighbors', 'SGDClassifier']
acuracy1 = [97,99,93]
xs = [i + 0.5 for i, _ in enumerate(algoritmo)]
margem_erro = [3, 1, 7]
plt.bar(xs, acuracy1, yerr= margem_erro)

plt.xlabel('Algoritmos utilizados', fontdict=font)
plt.ylabel('Acuracia (%)', fontdict=font)

# Tweak spacing to prevent clipping of ylabel
plt.title('PARTE I', fontdict=font)
plt.xticks([i + 0.5 for i, _ in enumerate(algoritmo)], algoritmo)
plt.subplots_adjust(left=0.15)
plt.show()


#Parte 2
alg = ['GaussianNB', 'KMeans', 'NuSVC', 'RandomForest']
acuracy2 = [92, 84, 88, 96]
x = [j + 0.5 for j, _ in enumerate(alg)]
margem = [8, 16, 12, 4]
plt.bar(x, acuracy2, yerr= margem)

plt.xlabel('Algoritmos utilizados', fontdict=font)
plt.ylabel('Acuracia (%)', fontdict=font)

# Tweak spacing to prevent clipping of ylabel
plt.title('PARTE II', fontdict=font)
plt.xticks([j + 0.5 for j, _ in enumerate(alg)], alg)
plt.subplots_adjust(left=0.15)
plt.show()