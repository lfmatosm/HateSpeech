import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure

#Cria um gráfico a partir dos parâmetros de abscissas e ordenadas passados como parâmetro e salva num arquivo PNG na raiz do programa.
def mostrarGraficoLinhas(y_linsvc, y_multnvb, y_lreg, x_kfolds, x_label, y_label):
    figure(num=None, figsize=(8, 6))
    plt.plot(x_kfolds, y_lreg, '.', label='Logistic regression',linestyle='-')  # Recebe dois arrays, um sera do K e outro dos resultados
    plt.plot(x_kfolds, y_multnvb, '.', label='Multinomial Naive Bayes', linestyle='-')
    plt.plot(x_kfolds, y_linsvc, '.', label='LinearSVC', linestyle='-')
    plt.ylim(0, 1)
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(x_label + " vs. " + y_label)
    plt.legend()
    plt.savefig('teste-' + y_label + '.png')
    plt.show()

#Cria um gráfico de barras a aprtir dos parâmetros passados.
def mostrarGraficoBarras(classif, accs, x_label, y_label):
    plt.rcdefaults()
    fig, ax = plt.subplots()
    y_pos = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    accs = np.array(accs)
    ax.barh(y_pos, accs, align='center', color='blue', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classif)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel(x_label)
    ax.set_title(x_label + " vs. " + y_label)
    plt.legend()
    plt.savefig('valid-' + x_label + " vs. " + y_label + '.png')
    plt.show()