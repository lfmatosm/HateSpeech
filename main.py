import pandas as pd
import numpy as np
import ProcessadorTexto as prtxt
import Graficos as grf
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

TOTAL_KFOLDS = 10

kfolds = [i+1 for i in range(TOTAL_KFOLDS)]

#Métricas de avaliação de erro a serem empregadas para verificar o desempenho de cada um dos classificadores.
metricas = ['accuracy', 'balanced_accuracy', 'roc_auc']
label_metricas = ['Acurácia', 'Acurácia Balanceada', 'Área sobre curva ROC']

#Criação de todos os classificadores.
nomes = ['Logistic Regression', 'Multinomial Naive Bayes',
         'VSM (Linear SVC)']
classificadores = [LogisticRegression(), MultinomialNB(), LinearSVC()]

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

#Separa os dados de treino/validação e as anotações corretas de classificação dos mesmos.
x_treino, x_validacao, y_treino, y_validacao = train_test_split(X, Y, test_size=0.2, random_state=0)

resultados = []

resultados_testes_metricas = {}
resultados_validacao = {}

#Para cada  classificador, realiza treinamento e teste, e avalia o erro/score segundo cada uma das métricas especificadas.
#Realiza o treino e predição de cada classificador, com determinado nome e base de treino com anotações
# de respostas corretas. Utiliza cross-validation. Emprega cada uma das métricas de avaliação de desempenho.
for i in range(len(classificadores)):
    scores = cross_validate(classificadores[i], x_treino, y_treino, cv=TOTAL_KFOLDS, scoring=metricas)
    print("Scores: ")
    print(scores)
    resultados_testes_metricas[nomes[i]] = scores
print(resultados_testes_metricas)

#Realiza a predição com cada um dos classificadores, utilizando uma base de validação com dados que não estão entre aqueles
#utilizados para treino pelos classificadores. Além disso, armazena os valores de acurácia obtidos
#para cada class.
for i in range(len(classificadores)):
    classificadores[i].fit(x_treino, y_treino)
    results = classificadores[i].predict(x_validacao)
    acuracia = accuracy_score(y_validacao, results)
    print("Results.: ")
    print(results)
    resultados_validacao[nomes[i]] = acuracia
print(resultados_validacao)

#Exibe uma imagem de um gráfico representando cada uma das métricas de avaliação empregadas.
for i in range(len(metricas)):
    res_mets = []
    for chave, valor in resultados_testes_metricas.items():
        res_mets.append(valor['test_' + metricas[i]])
    grf.mostrarGraficoLinhas(res_mets[0], res_mets[1], res_mets[2], kfolds, "Número do 'fold'", label_metricas[i])

#Gera o gráfico de acurácia da previsão (predict) de cada classificador.
accs = []
for i in range(len(classificadores)):
    acc = resultados_validacao.get(nomes[i])
    accs.append(acc)
grf.mostrarGraficoBarras(nomes, accs, "Acurácia (na Validação)", "")