import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Função de pertinência gaussiana
def gaussmf(x, mean, sigma):
    return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

# Função sigmoide para modelar uma saída entre 0 e 1 (para classificação binária)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Função para carregar e preparar dados
def import_data(dataPath, inicialRow, SampleSize):
    # Carregar dados a partir do arquivo Excel
    df = pd.read_excel(dataPath, skiprows=2 + inicialRow)
   
    # Seleciona as colunas de 13 até 23 (de faturas e pagamentos) e as 250 primeiras linhas
    entradas = df.iloc[:SampleSize, 13:23].values
    y = df.iloc[:SampleSize, 24].values  # Variável binária alvo
    return entradas, y

# Classe para treinamento do modelo fuzzy
class TreinamentoFuzzy:
    
    def __init__(self, n_inputs, n_rules, entradas):
        self.n_inputs = n_inputs
        self.n_rules = n_rules

        # Inicializar parâmetros fuzzy (mean e sigma para gaussianas)
        # Calcula a média e desvio padrão das colunas das entradas no dataset
        self.input_means = np.mean(entradas, axis=0)  # médias das 10 variáveis de entrada
        self.input_stds = np.std(entradas, axis=0)    # desvios padrão das 10 variáveis de entrada

        self.input_means = np.tile(self.input_means, (self.n_rules, 1))  # Cria um array (n_rules, n_inputs)
        self.input_stds = np.tile(self.input_stds, (self.n_rules, 1))    # Cria um array (n_rules, n_inputs)

        
        self.weights = np.random.uniform(-1, 1, n_rules)  # Pesos das regras
        self.learning_rate = 0.01  # Taxa de aprendizado

    def forward(self, X):
        # Inicializa as ativações para as entradas
        ativacoes_p_entrada = np.zeros((X.shape[0], self.n_rules))

        for i in range(self.n_rules):
            # Inicializa o grau de pertinência para cada regra como 1
            mu = 1
            # Calcula o grau de pertinência para cada entrada e para a regra i
            for j in range(self.n_inputs):  # Aqui j vai de 0 a 9 (10 entradas)
                mu *= gaussmf(X[:, j], self.input_means[i, j], self.input_stds[i, j])
            # Armazena o grau de pertinência para a regra i
            ativacoes_p_entrada[:, i] = mu

        # Calcula a saída final multiplicando as ativações pelas forças de disparo (pesos)
        output = np.dot(ativacoes_p_entrada, self.weights)
        
        # Aplica a função sigmoide para obter uma probabilidade entre 0 e 1
        probabilidade = (sigmoid(output) >= 0.5).astype(int)  # 0 ou 1 baseado na probabilidade
        return probabilidade, ativacoes_p_entrada

    # Função de treinamento usando backpropagation
    def train(self, entradas, y, epochs=500, save_weights_path="weights.npy"):
        # Loop de treinamento por número de épocas
        for epoch in range(epochs):
            # Passo forward: calcula a saída e as forças de ativação
            probabilidade, ativacoes_p_entrada = self.forward(entradas)

            # Calcula o erro usando a função de custo logarítmica (log-loss)
            erro = y - probabilidade

            # Atualiza os pesos das regras com base no erro calculado
            self.weights += self.learning_rate * np.dot(ativacoes_p_entrada.T, erro)

            # Backpropagation para ajustar os parâmetros das funções de pertinência (mean e sigma)
            for i in range(self.n_rules):
                for j in range(self.n_inputs):
                    # Gradiente para a média (mean) de cada função de pertinência
                    grad_mean = np.sum(erro * ativacoes_p_entrada[:, i] * (entradas[:, j] - self.input_means[i, j]) / (self.input_stds[i, j] ** 2))
                    # Gradiente para o desvio padrão (sigma) de cada função de pertinência
                    grad_sigma = np.sum(erro * ativacoes_p_entrada[:, i] * ((entradas[:, j] - self.input_means[i, j]) ** 2) / (self.input_stds[i, j] ** 3))

                    # Atualiza as médias e os desvios padrão com base nos gradientes
                    self.input_means[i, j] += self.learning_rate * grad_mean
                    self.input_stds[i, j] += self.learning_rate * grad_sigma

            # Exibe o erro a cada 10 épocas
            if (epoch + 1) % 10 == 0:
                mse = np.mean(erro ** 2)  # Calcula o erro quadrático médio (MSE)
                print(f"Epoch {epoch + 1}, Erro MSE: {mse}")

        np.save(save_weights_path, self.weights)
        print(f"Pesos salvos em: {save_weights_path}")

# Carregar dados
entradas, y = import_data('default_credit_card_clients.xls', 0, 250)  # Carregar entradas (faturas e pagamentos)

# Treinamento do modelo
modelo_fuzzy = TreinamentoFuzzy(n_inputs=10, n_rules=3, entradas=entradas)
modelo_fuzzy.train(entradas, y, epochs=100, save_weights_path="modelos/modelo")

# Realizar previsão
y_pred, _ = modelo_fuzzy.forward(entradas)

# Mostrar resultados
for i in range(len(entradas)):
    print(f"inadimplência esperada: {y[i]}, inadimplência predita: {y_pred[i]}")
    print("---")

# Gráfico da comparação entre inadimplência esperada e predita
plt.plot(y, label='inadimplência esperada')
plt.plot(y_pred, label='inadimplência predita', linestyle='--')
plt.xlabel('Amostras')
plt.ylabel('inadimplência')
plt.legend()
plt.show()
