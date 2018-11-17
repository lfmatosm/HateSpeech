import pandas as pd
import numpy as np
from collections import Counter
import ProcessadorTexto as prtxt
import Graficos as grf
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

TOTAL_KFOLDS = 5
ACCURACY = 0
BALANCED_ACC = 1
ROC_AUC = 2

kfolds = [i+2 for i in range(TOTAL_KFOLDS)]
resultados_metricas = {}

#Métricas de avaliação de erro a serem empregadas para verificar o desempenho de cada um dos classificadores.
metricas = ['accuracy', 'balanced_accuracy', 'roc_auc']

#Criação de todos os classificadores.
nomes = ['Linear Classifier (Logistic Regression)', 'BernoulliNB (Naïve Bayes)',
         'AdaBoost Classifier (?)']
classificadores = [LogisticRegression(random_state=0), BernoulliNB(), AdaBoostClassifier(random_state=0)]

proc = prtxt.ProcessadorTexto()

#Lê o arq. CSV com frases e suas respectivas classificações.
dados = pd.read_csv("base/base_dados.csv", encoding = 'utf-8')
#Estrutura de tabela contendo todas as frases da base CSV lida.
textoOriginal = dados['frase']
marcas = dados['valor']
vetorizacaoDoTexto = proc.processar(textoOriginal)

#Transforma a vetorização do texto e a lista de valorações/anotações da base de dados em arrays numpy.
X = np.array(vetorizacaoDoTexto)
Y = np.array(marcas.tolist())

#Separação das bases de treino e validação. A base de teites será separada a partir da base de treino usando cross-validation
# (validação cruzada - por um número k determinado de vezes, seleciona-se aleatoriamente dentro da base de dados os trechos
# que serão usados para treinamento e aqueles que serão usados para teste, de acordo com as porcentgens definidas anteriormente.
# O desempenho do classificador será dado pela média das k vezes em que o cross-validation foi empregado).
porcentagem_de_treino = 0.8

tamanho_do_treino = int(porcentagem_de_treino * len(Y))
tamanho_da_validacao = int(len(Y) - tamanho_do_treino)

#Separa os dados de treino e as anotações corretas de classificação dos mesmos.
x_treino = X[0:tamanho_do_treino]
y_treino = Y[0:tamanho_do_treino]

#Separa os dados de validação e as anotações corretas de classificação dos mesmos.
x_validacao = X[tamanho_do_treino:]
y_validacao = Y[tamanho_do_treino:]

resultados = []

#Para cada  classificador, realiza treinamento e teste, e avalia o erro/score segundo cada uma das métricas especificadas.
#Realiza o treino e predição de cada classificador, com determinado nome e base de treino com anotações
# de respostas corretas. Utiliza cross-validation. Emprega cada uma das métricas de avaliação de desempenho.
for i in range(len(metricas)):
    res_mat = []
    for j in range(len(classificadores)):
        res_vet_met = []
        for k in range(TOTAL_KFOLDS):
            scores = cross_val_score(classificadores[i], x_treino, y_treino, cv=k+2, scoring=metricas[i])
            taxa = np.mean(scores)
            msg = "Taxa de acerto ({0}) - métrica '{1}': {2} ({3})".format(nomes[j], metricas[i], taxa, np.std(scores))
            print(msg)
            res_vet_met.append(taxa)
        res_mat.append(res_vet_met)
    resultados_metricas[metricas[i]] = res_mat

#Exibe uma imagem de um gráfico representando cada uma das métricas de avaliação empregadas.
for chave, valor in resultados_metricas.items():
    cls1 = valor[0]
    cls2 = valor[1]
    cls3 = valor[2]
    print(valor[0])
    print(valor[1])
    print(valor[2])
    grf.mostrarGrafico(cls1, cls2, cls3, kfolds, chave)

#
# #Determinação do classificador de maior desempenho entre os testados.
# maximo = max(resultados[0])
# pos_vencedor = [i for i, j in enumerate(resultados[0]) if j == maximo][0]
# vencedor = classificadores[pos_vencedor]
#
# #Novo treinamento e predição (predição desta vez realizada com base de validação) do classificador de melhor
# # desempenho nos testes.
# vencedor.fit(x_treino, y_treino)
# resultado = vencedor.predict(x_validacao)
# #Contabilização da qtd. de acertos do classificador e de seu desempenho em porcentagem.
# acertos = (resultado == y_validacao)
# total_de_acertos = sum(acertos)
# total_de_elementos = len(y_validacao)
# taxa_de_acerto = (total_de_acertos / total_de_elementos) * 100
# msg = "\nTaxa de acerto do vencedor ({0}) no mundo real: {1}%".format(nomes[pos_vencedor], taxa_de_acerto)
# print(msg)
#
# #Cálculo de eficiência de um algoritmo que classificaria todas as entradas da base como o valor/predição mais provável apenas.
# #Usado para comparar com o desempenho do classificador escolhido, propriamente dito.
# acerto_base = max(Counter(y_validacao).values())
# taxa_de_acerto_base = 100.0 * acerto_base/len(y_validacao)
# print('Taxa de acerto base: ' + str(taxa_de_acerto_base) + '%')
# print('Total de testes: ' + str(len(x_validacao)))