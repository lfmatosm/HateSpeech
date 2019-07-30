# HateSpeech
Detecção de discurso de ódio utilizando técnicas de aprendizado de máquina.

## Procedimentos

Os procedimentos a seguir descrevem os passos necessários para execução do programa "HateSpeech", um detector de discurso de ódio utilizando aprendizado de máquina.

### Pré-requisitos

O programa foi testado nos sistemas operacionais Windows e Ubuntu 16.04 sem diferenças de execução entre ambos.
Para executar o programa é necessário possuir uma versão do Python instalada além das seguintes bibliotecas. Recomenda-se:

* [Python >= 3.5.2](https://www.python.org/downloads/) - O programa foi implementado e testado na versão [3.5.2](https://www.python.org/downloads/release/python-352/).
* [numpy >= 1.15.4](https://www.scipy.org/install.html) - Testado na versão 1.15.4. Biblioteca numérica eficiente para Python.
* [scikit-learn >= 0.20.0](https://scikit-learn.org/stable/install.html) - Testado na versão 0.20.0. Biblioteca para machine learning.
* [matplotlib >= 3.0.2](https://www.scipy.org/install.html) - Testado na versão 3.0.2. Biblioteca para plotting de gráficos.
* [pandas >= 0.23.4](https://www.scipy.org/install.html) - Testado na versão 0.23.4.
* [nltk >= 3.3](https://www.nltk.org/install.html) - Testado na versão 3.3. Biblioteca de processamento de linguagem natural. Possui pacote para processamento da língua portuguesa (usado neste programa).

### Executando

O arquivo a ser executado para iniciar o programa encontra-se na raiz do projeto e chama-se 'main.py'.

Logo ao ser iniciado, o programa iniciará o treinamento dos classificadores Logistic Regression (Classif. linear), MultinomialNB (Naive Bayes) e Linear SVC (Support Vector Machine). Por exemplo, após o treino de cada classificador e de sua avaliação segundo cada uma das métricas estabelecidas, o desempenho do mesmo será exibido no console Python na forma

```
...
Scores: 
{'test_roc_auc': array([0.74821747, 0.77361854, 0.80392157, ...]),
'test_balanced_accuracy': array([0.71212121, 0.66889483, ...]),
'train_roc_auc': array([0.9978093 , 0.99802221, ...]),
'train_balanced_accuracy': array([0.97921359, 0.98086922, ...]),
...
}
...
...
```

Após o treinamento de cada classificador e após realizadas previsões usando a base de validação dos modelos, os resultados acima serão exibidos na forma de gráficos do tipo "K vs. Taxa de desempenho da métrica empregada", sendo K o número do fold atual usado para realizar cross-validation em cada classificador e a taxa de desempenho sendo relativa a cada uma das métricas utilizadas. Arquivos PNG serão criados na raiz do projeto contendo estes gráficos.

## Autores

* **Luiz Felipe de Melo** - *Implementação, documentação.* - [lffloyd](https://github.com/lffloyd)
* **Vítor Costa** - *Implementação, documentação.* - [vitorhardoim](https://github.com/vitorhardoim)
* **Renato Bastos** - *Implementação, documentação.* - [RenatoBastos33](https://github.com/RenatoBastos33)

Veja a lista de [contribuidores](https://github.com/lffloyd/HateSpeech/contributors) participantes no projeto.

## Licença

Projeto licenciado sob a licença MIT - leia [LICENSE.md](https://github.com/lffloyd/HateSpeech/blob/master/LICENSE) para maiores detalhes.
