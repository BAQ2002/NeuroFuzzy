import numpy as np
import matplotlib.pyplot as plt
from TreinamentoModelo import TreinamentoFuzzy
from TreinamentoModelo import import_data

# Carregar dados de teste
entradas_teste, _ = import_data('default_credit_card_clients.xls', 250, 100)

# Função para carregar os pesos do modelo
def carregar_pesos(modelo, weights_path="modelos/modelo.npy"):
    modelo.weights = np.load(weights_path)
    print(f"Pesos carregados de: {weights_path}")

# Função para realizar a predição com o modelo treinado
def realizar_predicao(modelo, entradas_teste):
    probabilidade, _ = modelo.forward(entradas_teste)  # Obtém as predições e as ativações
    y_pred = (probabilidade >= 0.5).astype(int)  # Converte probabilidade para 0 ou 1
    return y_pred

# Inicializar o modelo fuzzy com o número de entradas e regras
modelo_fuzzy = TreinamentoFuzzy(n_inputs=10, n_rules=3, entradas=entradas_teste)

# Carregar os pesos treinados
carregar_pesos(modelo_fuzzy, weights_path="modelos/modelo.npy")

# Realizar a predição nos dados de teste
y_pred = realizar_predicao(modelo_fuzzy, entradas_teste)

# Exibir os resultados da predição
for i in range(len(entradas_teste)):
    print(f"Predição {i + 1}: {y_pred[i]}, valor real {i + 1}: {_[i]}")

# Gráfico da comparação entre inadimplência esperada e predita
plt.plot(_, label='inadimplência esperada')
plt.plot(y_pred, label='inadimplência predita', linestyle='--')
plt.xlabel('Amostras')
plt.ylabel('inadimplência')
plt.legend()
plt.show()