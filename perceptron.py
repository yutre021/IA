import numpy as np

# Função de ativação: degrau
def step_function(x):
    return 1 if x >= 0 else 0

# Classe Perceptron
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=10):
        self.weights = np.zeros(input_size + 1)  # Inclui o peso do bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, x):
        try:
            # Calcula a saída com base nos pesos e no bias
            weighted_sum = np.dot(x, self.weights[1:]) + self.weights[0]  # w0 é o bias
            return step_function(weighted_sum)
        except Exception as e:
            print(f"Erro durante a previsão: {e}")
            return None

    def train(self, X, y):
        try:
            for _ in range(self.epochs):
                for xi, target in zip(X, y):
                    prediction = self.predict(xi)
                    if prediction is not None:
                        error = target - prediction
                        # Atualiza os pesos e o bias
                        self.weights[1:] += self.learning_rate * error * xi
                        self.weights[0] += self.learning_rate * error
        except Exception as e:
            print(f"Erro durante o treinamento: {e}")

# Exemplo de uso
if __name__ == "__main__":
    try:
        # Entradas (X) e saídas desejadas (y)
        # Exemplo: Porta lógica AND
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 0, 1])  # Saída da porta AND

        # Criação e treinamento do Perceptron
        perceptron = Perceptron(input_size=2)
        perceptron.train(X, y)

        # Teste do Perceptron
        print("Resultados:")
        for xi in X:
            result = perceptron.predict(xi)
            if result is not None:
                print(f"Entrada: {xi} -> Saída: {result}")
            else:
                print(f"Erro ao prever saída para a entrada {xi}")

    except Exception as e:
        print(f"Erro geral no programa: {e}")
