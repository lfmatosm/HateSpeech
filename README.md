# HateSpeech
Detecção de discurso de ódio utilizando técnicas de aprendizado de máquina.

## Procedimentos

Os procedimentos a seguir descrevem os passos necessários para execução do programa "HateSpeech", um detector de discurso de ódio utilizando aprendizado de máquina.

### Pré-requisitos

O programa foi testado nos sistemas operacionais Windows e Ubuntu 16.04 sem diferenças de execução entre ambos.
Para executar o programa é necessário possuir uma versão do Python instalada além das seguintes bibliotecas. Recomenda-se:

* [Python >= 3.5.2](https://www.python.org/downloads/) - O programa foi implementado e testado na versão [3.5.2](https://www.python.org/downloads/release/python-352/).
* [numpy >= 1.15.4](https://www.scipy.org/install.html) - Testado na versão 1.15.4.
* [scikit-learn >= 0.20.0](https://scikit-learn.org/stable/install.html) - Testado na versão 0.20.0.
* [matplotlib >= 3.0.2](https://www.scipy.org/install.html) - Testado na versão 3.0.2.
* [pandas >= 0.23.4](https://www.scipy.org/install.html) - Testado na versão 0.23.4.

### Executando

O arquivo a ser executado para iniciar o programa encontra-se na raiz do projeto e chama-se 'main.py'.

Logo ao ser iniciado, o programa iniciará o treinamento dos classificadores Logistic Regression, BernoulliNB e AdaBoost Classifier. Por exemplo, após o treino de cada classificador e de sua avaliação segundo cada uma das métricas estabelecidas, o desempenho do mesmo será exibido no console Python na forma

```
...
Taxa de acerto (Linear Classifier (Logistic Regression)) - métrica 'roc_auc': 0.7100984311821367 (0.10211872868554253)
Taxa de acerto (BernoulliNB (Naïve Bayes)) - métrica 'roc_auc': 0.5525923312009816 (0.05753986863423663)
Taxa de acerto (AdaBoost Classifier) - métrica 'roc_auc': 0.624365111139245 (0.0010222903767552194)
...
```

Após o treinamento de cada classificador e após realizadas previsões usando a base de validação dos modelos, os resultados acima serão exibidos na forma de gráficos do tipo "K vs. Taxa de desempenho da métrica empregada", sendo K o número de 'folds' criados para realizar cross-validation em cada classificador e a taxa de desempenho sendo relativa a cada uma das métricas utilizadas. Arquivos .PNG serão criados na raiz do projeto contendo estes gráficos.

## Autores

* **Luiz Felipe de Melo** - *Implementação, documentação.* - [lffloyd](https://github.com/lffloyd)
* **Vítor Costa** - *Implementação, documentação.* - [vitorhardoim](https://github.com/vitorhardoim)
* **Renato Bastos** - *Implementação, documentação.* - [RenatoBastos33](https://github.com/RenatoBastos33)
* **Erick Guimarães** - *Implementação, documentação.* - [ErickGuimaraes](https://github.com/ErickGuimaraes)

Veja a lista de [contribuidores](https://github.com/lffloyd/HateSpeech/contributors) participantes no projeto.

## Licença

Projeto licenciado sob a licença MIT - leia [LICENSE.md](https://github.com/lffloyd/HateSpeech/blob/master/LICENSE) para maiores detalhes.