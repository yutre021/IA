#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>

using namespace std;

// Definições
#define EPOCHS 10000    // Número máximo de iterações
#define LEARNING_RATE 0.5 // Taxa de aprendizado

// Função de ativação sigmoide
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivada da sigmoide
float sigmoid_derivative(float x) {
    return x * (1.0 - x);
}

int main() {
    srand(time(0)); // Semente para números aleatórios

    // Conjunto de treinamento XOR
    float e1[4] = {0, 0, 1, 1};
    float e2[4] = {0, 1, 0, 1};
    float tg[4] = {0, 1, 1, 0};

    // Inicialização dos pesos aleatórios
    float w1[3][2]; // Pesos entre a entrada e a camada oculta (2 neurônios)
    float w2[3][1]; // Pesos entre a camada oculta e a saída

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            w1[i][j] = ((rand() / (float)RAND_MAX) * 2.0) - 1.0; // [-1,1]
        }
        w2[i][0] = ((rand() / (float)RAND_MAX) * 2.0) - 1.0; // [-1,1]
    }

    // Treinamento
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float totalError = 0.0;

        for (int cs = 0; cs < 4; cs++) {
            // Entrada com bias
            float i1 = e1[cs], i2 = e2[cs], bias = 1.0;

            // Forward Pass - Cálculo da saída da camada oculta
            float h1 = sigmoid(i1 * w1[0][0] + i2 * w1[1][0] + bias * w1[2][0]);
            float h2 = sigmoid(i1 * w1[0][1] + i2 * w1[1][1] + bias * w1[2][1]);

            // Forward Pass - Cálculo da saída final
            float o = sigmoid(h1 * w2[0][0] + h2 * w2[1][0] + bias * w2[2][0]);

            // Erro
            float erro = tg[cs] - o;
            totalError += erro * erro;

            // Backpropagation - Ajuste dos pesos
            float delta_o = erro * sigmoid_derivative(o);
            float delta_h1 = delta_o * w2[0][0] * sigmoid_derivative(h1);
            float delta_h2 = delta_o * w2[1][0] * sigmoid_derivative(h2);

            // Atualização dos pesos da saída
            w2[0][0] += LEARNING_RATE * delta_o * h1;
            w2[1][0] += LEARNING_RATE * delta_o * h2;
            w2[2][0] += LEARNING_RATE * delta_o * bias;

            // Atualização dos pesos da camada oculta
            w1[0][0] += LEARNING_RATE * delta_h1 * i1;
            w1[1][0] += LEARNING_RATE * delta_h1 * i2;
            w1[2][0] += LEARNING_RATE * delta_h1 * bias;
            
            w1[0][1] += LEARNING_RATE * delta_h2 * i1;
            w1[1][1] += LEARNING_RATE * delta_h2 * i2;
            w1[2][1] += LEARNING_RATE * delta_h2 * bias;
        }

        totalError /= 4.0;
        if (epoch % 1000 == 0) {
            cout << "Epoch " << epoch << " - Erro: " << totalError << endl;
        }

        if (totalError < 0.0001) break; // Critério de parada
    }

    // Exibir pesos treinados
    cout << "\nPesos finais:\n";
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            cout << "w1[" << i << "][" << j << "] = " << w1[i][j] << endl;
        }
    }
    for (int i = 0; i < 3; i++) {
        cout << "w2[" << i << "][0] = " << w2[i][0] << endl;
    }

    return 0;
}
