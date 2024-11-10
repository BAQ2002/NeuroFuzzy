import numpy as np
from TreinamentoModelo import TreinamentoFuzzy
from TreinamentoModelo import import_data

# Carregar dados de teste
entradas_teste, _ = import_data('default_credit_card_clients_test.xls', 250, 250)

# Função para carregar os pesos do modelo
def carregar_pesos(modelo, weights_path="modelos/modelo.npy"):
    modelo.weights = np.load(weights_path)
    print(f"Pesos carregados de: {weights_path}")

# Função para realizar a predição com o modelo treinado
def realizar_predicao(modelo, entradas_teste):
    probabilidade = modelo.forward(entradas_teste)
    y_pred = (probabilidade >= 0.5).astype(int)  # Converte probabilidade para 0 ou 1
    return y_pred