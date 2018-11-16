import pandas as pd
import numpy as np
from collections import Counter
import ProcessadorTexto as prtxt
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

#Realiza o treino e predição de um classificador do tipo modelo, com determinado nome e base de treino com anotações
# de respostas corretas. Utiliza cross-validation.
def treinarEPrever(nome, modelo, treino_dados, treino_marcacoes):
    k = 10
    scores = cross_val_score(modelo, treino_dados, treino_marcacoes, cv=k)
    taxa_de_acerto = np.mean(scores)*100
    msg = "Taxa de acerto do {0}: {1}%".format(nome, taxa_de_acerto)
    print(msg)
    return taxa_de_acerto

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

#Criação de todos os classificadores e realização de treinamento/predição com os mesmos. Acúmulo dos seus resultados
SGDC = SGDClassifier(loss='log', alpha=0.1, penalty='l2')
OneVsRest = OneVsRestClassifier(LinearSVC(random_state=0))
OneVsOne = OneVsOneClassifier(LinearSVC(random_state=0))
Multinomial = MultinomialNB()
AdaBoost = AdaBoostClassifier(random_state=0)
classificadores = [('Stochastic Gradient Descent', SGDC), ('OneVsRest', OneVsRest), ('OneVsOne', OneVsOne),
                   ('MultinomialNB', Multinomial), ('AdaBoost Classifier', AdaBoost)]
resultados = {}

for tupla in classificadores:
    resultado = treinarEPrever(tupla[0], tupla[1], x_treino, y_treino)
    resultados[resultado] = tupla[1]

#Determinação do classificador de maior desempenho entre os testados.
maximo = max(resultados)
vencedor = resultados[maximo]
#Novo treinamento e predição (predição desta vez realizada com base de validação) do classificador de melhor
# desempenho nos testes.
vencedor.fit(x_treino, y_treino)
resultado = vencedor.predict(x_validacao)
#Contabilização da qtd. de acertos do classificador e de seu desempenho em porcentagem.
acertos = (resultado == y_validacao)
total_de_acertos = sum(acertos)
total_de_elementos = len(y_validacao)
taxa_de_acerto = total_de_acertos / total_de_elementos * 100
msg = "\nTaxa de acerto do vencedor entre os algoritmos no mundo real: {0}%".format(taxa_de_acerto)
print(msg)

#Cálculo de eficiência de um algoritmo que classificaria todas as entradas da base como o valor/predição mais provável apenas.
#Usado para comparar com o desempenho do classificador escolhido, propriamente dito.
acerto_base = max(Counter(y_validacao).values())
taxa_de_acerto_base = 100.0 * acerto_base/len(y_validacao)
print('Taxa de acerto base: ' + str(taxa_de_acerto_base) + '%')
print('Total de Testes: ' + str(len(x_validacao)))