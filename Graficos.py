import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

def mostrarGrafico(y_lreg, y_nvbayes, y_adaboost, x_kfolds, y_label):
    figure(num=None, figsize=(8, 6))
    plt.plot(x_kfolds, y_lreg, '.', label='Logistic regression',linestyle='-')  # Recebe dois arrays, um sera do K e outro dos resultados
    plt.plot(x_kfolds, y_nvbayes, '.', label='Multinomial Naive Bayes', linestyle='-')
    plt.plot(x_kfolds, y_adaboost, '.', label='LinearSVC', linestyle='-')
    plt.ylim(0, 1)
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.xlabel("K (Tamanho do 'fold')")
    plt.ylabel(y_label)
    plt.title("K x " + y_label)
    plt.legend()
    plt.savefig('teste-' + y_label + '.png')
    plt.show()