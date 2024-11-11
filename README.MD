# Projeto de Previsão de Inadimplência com Modelos Fuzzy

Este projeto tem como objetivo prever a inadimplência de clientes de um banco financeiro utilizando um modelo fuzzy. A previsão é baseada em entradas relacionadas ao histórico de pagamentos e faturas dos clientes. O modelo utiliza funções de pertinência gaussianas e sigmoid e uma arquitetura inspirada em redes neurais para classificação binária (inadimplência ou não).

## Estrutura do Projeto

O projeto é composto por duas etapas principais:

1. **Treinamento do Modelo** (`TreinamentoModelo.py`): 
    - Realiza o treinamento de um modelo fuzzy utilizando um conjunto de dados históricos (faturas e pagamentos).
    - O modelo é treinado para prever a inadimplência de clientes, considerando um conjunto de  5 entradas referentes a fatura e 5 entradas refrentes a pagamentos e a variável alvo binária (inadimplente ou não).
    - Após ser feito o treinamento, o moledo em salvo no diretório /modelos.

2. **Teste do Modelo** (`TestarModelo.py`): 
    - Carrega um modelo previamente treinado e realiza previsões sobre um novo conjunto de dados (dados de teste).
    - Exibe as predições e compara com os valores reais de inadimplência para avaliar a acurácia do modelo.

## Requisitos

Antes de rodar o projeto, é necessário instalar as dependências necessárias:

- `numpy`
- `pandas`
- `matplotlib`

Você pode instalar essas bibliotecas utilizando o seguinte comando:

```bash
pip install numpy pandas matplotlib
```

## Video Explicativo: https://drive.google.com/file/d/1vT-tceRM-NPrtklnkZ8KXfJHjUl-jkkH/view?usp=drive_link 