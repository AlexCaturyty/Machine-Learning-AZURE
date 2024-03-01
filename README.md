# Modelo de Previsão de Preços de Casas

Neste projeto, construímos um modelo de previsão simples para estimar o preço de casas com base em suas características. Utilizamos Python e a biblioteca scikit-learn para criar um modelo de regressão linear.

## Passo a Passo

### Passo 1: Definição do Problema

O objetivo deste projeto é desenvolver um modelo que possa prever o preço de casas com base em características como tamanho, número de quartos, número de banheiros e ano de construção.

### Passo 2: Coleta de Dados

Geramos dados de exemplo para este projeto. Criamos um conjunto de dados fictício contendo informações sobre o tamanho das casas, o número de quartos, o número de banheiros e o ano de construção, juntamente com os preços das casas.

### Passo 3: Pré-processamento de Dados

Não foi necessário realizar um pré-processamento extenso dos dados, pois geramos dados de exemplo aleatoriamente e não encontramos valores ausentes ou inconsistentes.

### Passo 4: Divisão dos Dados

Dividimos os dados em conjuntos de treino e teste usando a função `train_test_split` da biblioteca scikit-learn. Isso nos permitiu avaliar o desempenho do modelo em dados não vistos durante o treinamento.

### Passo 5: Construção do Modelo

Utilizamos um modelo de regressão linear da biblioteca scikit-learn para construir o modelo de previsão. Treinamos o modelo com os dados de treino e fizemos previsões no conjunto de teste.

### Passo 6: Avaliação do Modelo

Avaliamos o desempenho do modelo usando o erro quadrático médio (MSE). Quanto menor o valor do MSE, melhor o desempenho do modelo.

## Código

Aqui está o código Python utilizado para construir e avaliar o modelo:

```python
# Importar as bibliotecas necessárias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Gerar dados de exemplo (características das casas e seus preços)
np.random.seed(0)
num_samples = 1000
house_size = np.random.randint(800, 5000, num_samples)
num_rooms = np.random.randint(2, 6, num_samples)
num_bathrooms = np.random.randint(1, 4, num_samples)
year_built = np.random.randint(1900, 2020, num_samples)
price = 1000 * house_size + 5000 * num_rooms + 2000 * num_bathrooms + 100 * (2024 - year_built) + np.random.normal(0, 50000, num_samples)

# Criar um DataFrame pandas com os dados
data = pd.DataFrame({
    'Size': house_size,
    'Rooms': num_rooms,
    'Bathrooms': num_bathrooms,
    'YearBuilt': year_built,
    'Price': price
})

# Dividir os dados em conjuntos de treino e teste
X = data.drop('Price', axis=1)
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o desempenho do modelo usando o erro quadrático médio (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Erro Quadrático Médio (MSE):", mse)
