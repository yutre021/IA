import torch

try:
    # Definindo uma variável com gradiente habilitado
    x = torch.tensor(5.0, requires_grad=True)  # Valor inicial de x

    # Definindo a função: f(x) = x^2 + 3x + 2
    y = x**2 + 3*x + 2

    # Calculando o gradiente automaticamente
    y.backward()  # Calcula a derivada de y em relação a x

    # Obtendo o gradiente
    print(f"O valor da função em x = {x.item()} é {y.item()}")
    print(f"A derivada de f(x) em x = {x.item()} é {x.grad.item()}")

except Exception as e:
    print(f"Ocorreu um erro: {e}")
