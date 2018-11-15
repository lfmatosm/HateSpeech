import pandas as pd
import numpy as np
import nltk
from collections import Counter
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB

#Recebe um texto (frase de radicais) e um tradutor (dicionário mapeando radicais a valores de índice). Gera uma lista
#de ocorrências de cada palavra (radical) do texto passado no próprio texto (acho que em toda a base, na verdade).
def vetorizar_texto(texto, tradutor):
    vetor = [0] * len(tradutor)
    for palavra in texto:
        if len(palavra) > 0:
            raiz = stemmer.stem(palavra)
            if raiz in tradutor:
                posicao = tradutor[raiz]
                vetor[posicao] += 1
    return vetor

nltk.download('stopwords')
nltk.download('rslp')
nltk.download('punkt')

#Stopwords: artigos, preposições, conectivos, etc.
stopwords = nltk.corpus.stopwords.words('portuguese')
print(stopwords)
#Objeto que extrai o radical de palavras.
stemmer = nltk.stem.RSLPStemmer()

#Lê o arq. CSV com frases e suas respectivas classificações.
classificacoes = pd.read_csv("base/base_dados.csv", encoding = 'utf-8')
#Estrutura de tabela contendo todas as frases da base CSV lida.
textosPuros = classificacoes['frase']
#Transforma todas as frases em lower case.
frases = textosPuros.str.lower()
#Cria os tokens para cada uma das frases: tokens serão as palavras de cada frase. Cada sequência de palavras numa mesma
#frase será representada por uma lista que descreve as palavras desta frase (ou linha neste caso).
textosQuebrados = [nltk.tokenize.word_tokenize(frase) for frase in frases]
dicionario = set()

#Para cada linha ou lista de palavras dentro de textosQuebrados (que contém todas as listas ou frases com cada
# uma de suas palavras):
for lista in textosQuebrados:
    #Cria uma lista de palavras válidas. Cada palavra dessa lista 'validas' será um radical (p.ex., 'volunt' é radical de
    # voluntário, pois existem diferentes palavras que terminam com o sufixo 'ário') que não está contido na lista de palavras
    #'stopwords' e têm tamanho maior que 2.
    validas = [stemmer.stem(palavra) for palavra in lista if palavra not in stopwords and len(palavra) > 2]
    dicionario.update(validas)

print(dicionario)

totalDePalavras = len(dicionario)
print(totalDePalavras)
#Cria um iterador relacionando cada palavra do dicionário (conjunto) de palavras a um índice de 0 até totalDePalavras.
tuplas = zip(dicionario, range(totalDePalavras))
#Cria o tradutor. Este será um dicionário mapeando para cada palavra/radical (chave) um índice (valor) associado.
tradutor = {palavra:indice for palavra,indice in tuplas}
#print(tradutor)

#Chama a função que vetoriza o texto original, com base no dict tradutor criado acima. vetoresDeTexto irá conter uma lista
#de listas, sendo que cada lista irá conter a contagem de cada palavra (em ordem) ali existente na frase original (seguindo
#a ordenação original da frase).
vetoresDeTexto = [vetorizar_texto(texto, tradutor) for texto in textosQuebrados]

#print(vetoresDeTexto)
marcas = classificacoes['valor']
#print(marcas)

#Transforma a vetorização do texto e a lista de valorações/anotações da base de dados em arrays numpy.
X = np.array(vetoresDeTexto)
Y = np.array(marcas.tolist())

#Separação das bases de treino e validação. A base de teites será separada a partir da base de treino usando k-folding (?).
porcentagem_de_treino = 0.8

tamanho_do_treino = int(porcentagem_de_treino * len(Y))
tamanho_da_validacao = int(len(Y) - tamanho_do_treino)

#Separa os dados de treino e as anotações corretas de classificação dos mesmos.
treino_dados = X[0:tamanho_do_treino]
treino_marcacoes = Y[0:tamanho_do_treino]

#Separa os dados de validação e as anotações corretas de classificação dos mesmos.
validacao_dados = X[tamanho_do_treino:]
validacao_marcacoes = Y[tamanho_do_treino:]

#Realiza o treino e predição de um classificador do tipo modelo, com determinado nome e base de treino com anotações
# de respostas corretas.
def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes):
    k = 10
    scores = cross_val_score(modelo, treino_dados, treino_marcacoes, cv=k)
    taxa_de_acerto = np.mean(scores)*100
    msg = "Taxa de acerto do {0}: {1}%".format(nome, taxa_de_acerto)
    print(msg)
    return taxa_de_acerto

resultados = {}

#Criação de todos os classificadores e realização de treinamento/predição com os mesmos. Acúmulo dos seus resultados
#num dicionário de resultados.
modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state=0))
resultadoOneVsRest = fit_and_predict("OneVsRest", modeloOneVsRest, treino_dados, treino_marcacoes)
resultados[resultadoOneVsRest] = modeloOneVsRest

modeloOneVsOne = OneVsOneClassifier(LinearSVC(random_state=0))
resultadoOneVsOne = fit_and_predict("OneVsOne", modeloOneVsOne, treino_dados, treino_marcacoes)
resultados[resultadoOneVsOne] = modeloOneVsOne

modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes)
resultados[resultadoMultinomial] = modeloMultinomial

modeloAdaBoost = AdaBoostClassifier(random_state=0)
resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modeloAdaBoost, treino_dados, treino_marcacoes)
resultados[resultadoAdaBoost] = modeloAdaBoost

#Determinação do classificador de maior desempenho entre os testados.
maximo = max(resultados)
vencedor = resultados[maximo]
#Novo treinamento e predição (predição desta vez realizada com base de validação) do classificador de melhor
# desempenho nos testes.
vencedor.fit(treino_dados, treino_marcacoes)
resultado = vencedor.predict(validacao_dados)
#Contabilização da qtd. de acertos do classificador e de seu desempenho em porcentagem.
acertos = (resultado == validacao_marcacoes)
total_de_acertos = sum(acertos)
total_de_elementos = len(validacao_marcacoes)
taxa_de_acerto = total_de_acertos / total_de_elementos * 100
msg = "\nTaxa de acerto do vencedor entre os algoritmos no mundo real: {0}%".format(taxa_de_acerto)
print(msg)

#Cálculo de eficiência de um algoritmo que classificaria todas as entradas da base como o valor/predição mais provável apenas.
#Usado para comparar com o desempenho do classificador escolhido, propriamente dito.
acerto_base = max(Counter(validacao_marcacoes).values())
taxa_de_acerto_base = 100.0 * acerto_base/len(validacao_marcacoes)
print('Taxa de acerto base: ' + str(taxa_de_acerto_base) + '%')
print('Total de Testes: ' + str(len(validacao_dados)))