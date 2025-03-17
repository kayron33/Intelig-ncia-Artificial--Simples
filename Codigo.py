#Nesse exemplo eu vou criar uma Inteligência Artificial capaz de aprender com dados artificiais, sobre a média salarial de pessoas de acordo coma  sua idade. 

import pandas as pd 
import numpy as np

np.random.seed(42)


dados = pd.read_csv("dados_ia_salario.csv")

# Agora vou separar os dados que a IA deve estudar em X e Y:
X = dados.iloc[:,:-1].values # Esse codigo faz com que a Inteligência Artificial consiga estudar com todas as colunas menos a ultima coluna que seria a nossa resposta no caso.
Y = dados.iloc[:,-1].values.reshape(-1,1) # Essa parte faz com que os dados de Y seja somente os da ultima coluna

# Vamos normalizar os dados para uma melhor otimização da nossa IA

# Para começar vou normalizar os dados com o calculo de normalização que seria- X = (X - X_min) / (X_max - X_min)
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X = (X - X_min) / (X_max - X_min)
# Agora vou fazer a mesma coisa com os dados de Y:

Y_min = Y.min(axis=0)
Y_max = Y.max(axis=0)
Y = (Y - Y_min) / (Y_max - Y_min)

# Agora vamos construir os pesos, que irão conter os neurônios de nossa IA ou seja 1 só ultilizando o metodo Xavier_init:
def xavier_init(shape):
    return np.random.randn(*shape)*np.sqrt(2/shape[0])

pesos_ocultos = xavier_init((X.shape[1], 10))
pesos_saida = xavier_init((10,1))

# Fazendo os calculos de sigmoid:
def sigmoid(x):
    return 1 / (1+ np.exp(-x))

def sigmoid_derivada(x):
    return x * (1 - x)

# Adicionando os dados para treino da IA:
taxa_aprendizado = 0.1
epocas = 50000
lambda_l2 = 0.01
batch_size = 4

# Iniciando o treinamento da IA 
for epoca in range(epocas):
    indices = np.random.permutation(X.shape[0])
    X_embaralhado = X[indices]
    Y_embaralhado = Y[indices]
    for i in range(0,X.shape[0],batch_size):
        X_batch = X_embaralhado[i: i + batch_size]
        Y_batch = Y_embaralhado[i: i + batch_size]

        # Construindo as camadas da nossa rede neural para treina-la
        camada_oculta = sigmoid(np.dot(X_batch,pesos_ocultos))
        saida_final = sigmoid(np.dot(camada_oculta,pesos_saida))

        # Fazendo o calculo para que a IA saiba quanto esta errando:
        erro = Y_batch - saida_final

        # Fazendo o ajuste do treino da nossa rede neural:
        ajuste_saida = erro * sigmoid_derivada(saida_final)
        ajuste_oculta = np.dot(ajuste_saida,pesos_saida.T)* sigmoid_derivada(camada_oculta)

        # Fazendo a  atualização dos pesos:
        pesos_saida -= taxa_aprendizado *(np.dot(camada_oculta.T,ajuste_saida)+ lambda_l2 * pesos_saida)
        pesos_ocultos -= taxa_aprendizado * (np.dot(X_batch.T,ajuste_oculta)+ lambda_l2*pesos_ocultos)

        # Informando ao leitor o aprendizado dessa rede Neural:
        
        if epoca % 10000 == 0:
            print(f"Epoca: {epoca}-Erro Medio: {np.mean(np.abs(erro)):.4f}")

print("IA treinada com sucesso!!!!!")
