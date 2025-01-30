import tensorflow as tf

try:
    # Definindo uma função simples: f(x) = x^2 + 3x + 2
    def func(x):
        return x**2 + 3*x + 2

    # Inicializando uma variável x
    x = tf.Variable(5.0)  # Valor inicial de x

    # Usando autodiff para calcular a derivada de f(x) em relação a x
    with tf.GradientTape() as tape:
        y = func(x)  # Calcula a função

    # Obtém o gradiente de y em relação a x
    grad = tape.gradient(y, x)
    print(f"O valor da função em x = {x.numpy()} é {y.numpy()}")
    print(f"A derivada de f(x) em x = {x.numpy()} é {grad.numpy()}")

except Exception as e:
    print(f"Ocorreu um erro: {e}")
