import nltk

#Classe que tem como função processar as entradas em linguagem natural dadas na forma de frases, separando palavras,
#radicais e etc.
class ProcessadorTexto():
    def __init__(self):
        # O Natural Language Toolkit necessita de bases de dados/bibliotecas que possam categorizar informações em pt-BR.
        nltk.download('stopwords')
        nltk.download('rslp')
        nltk.download('punkt')
        # Stopwords: artigos, preposições, conectivos, etc.
        self.stopwords = nltk.corpus.stopwords.words('portuguese')
        # Objeto que extrai o radical de palavras.
        self.stemmer = nltk.stem.RSLPStemmer()
        #print(self.stopwords)

    # Recebe um texto (frase de radicais) e um tradutor (dicionário mapeando radicais a valores de índice). Gera uma lista
    # de ocorrências de cada palavra (radical) do texto passado no próprio texto (acho que em toda a base, na verdade).
    def vetorizarTexto(self, texto, tradutor):
        vetor = [0] * len(tradutor)
        for palavra in texto:
            if len(palavra) > 0:
                raiz = self.stemmer.stem(palavra)
                if raiz in tradutor:
                    posicao = tradutor[raiz]
                    vetor[posicao] += 1
        return vetor

    def processar(self, textoOriginal):
        # Transforma todas as frases em lower case.
        frases = textoOriginal.str.lower()
        # Cria os tokens para cada uma das frases: tokens serão as palavras de cada frase. Cada sequência de palavras numa mesma
        # frase será representada por uma lista que descreve as palavras desta frase (ou linha neste caso).
        textosQuebrados = [nltk.tokenize.word_tokenize(frase) for frase in frases]
        dicionario = set()
        # Para cada linha ou lista de palavras dentro de textosQuebrados (que contém todas as listas ou frases com cada
        # uma de suas palavras):
        for lista in textosQuebrados:
            # Cria uma lista de palavras válidas. Cada palavra dessa lista 'validas' será um radical (p.ex., 'volunt' é radical de
            # voluntário, pois existem diferentes palavras que terminam com o sufixo 'ário') que não está contido na lista de palavras
            # 'stopwords' e têm tamanho maior que 2.
            validas = [self.stemmer.stem(palavra) for palavra in lista if palavra not in self.stopwords and len(palavra) > 2]
            dicionario.update(validas)
        #print(dicionario)
        totalDePalavras = len(dicionario)
        #print(totalDePalavras)
        # Cria um iterador relacionando cada palavra do dicionário (conjunto) de palavras a um índice de 0 até totalDePalavras.
        tuplas = zip(dicionario, range(totalDePalavras))
        # Cria o tradutor. Este será um dicionário mapeando para cada palavra/radical (chave) um índice (valor) associado.
        tradutor = {palavra: indice for palavra, indice in tuplas}
        # print(tradutor)
        return [self.vetorizarTexto(texto, tradutor) for texto in textosQuebrados]