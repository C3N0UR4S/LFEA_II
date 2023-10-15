import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

dados = pd.read_excel('dados.xlsx', sheet_name='Botao-Osciloscopio')

botao = dados['Botao [ns]']
oscilo = dados['Osciloscopio [ns]']
err_oscilo = dados['incerteza [ns]']

def linear_model(x, a, b):
    return a * x + b

# Realiza o ajuste ponderado
params, covariance = curve_fit(linear_model, botao, oscilo, sigma=err_oscilo)

# Parâmetros do ajuste (a e b)
m, b = params

# Erros padrão dos parâmetros
m_err, b_err = np.sqrt(np.diag(covariance))

print(f"Ajuste linear: a = {m:.5f} +/- {m_err:.5f}, b = {b:.5f} +/- {b_err:.5f}")

# Calcule o coeficiente de determinação (R²)
residuals = oscilo - linear_model(botao, m, b)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((oscilo - np.mean(oscilo))**2)
r_squared = 1 - (ss_res / ss_tot)

print(f"Coeficiente de Determinação (R²): {r_squared:.4f}")

# Crie uma faixa de valores de x para traçar a função de ajuste
x_fit = np.linspace(min(botao), max(botao), 100)
y_fit = linear_model(x_fit, m, b)

plt.errorbar(botao, oscilo, yerr=err_oscilo, fmt='.', capsize=2, label='pontos experimentais')
plt.plot(x_fit, y_fit, label=f'Ajuste - $y = ({m:.4f} \pm {m_err:.4f} )*x + ({b:.4f} \pm {b_err:.4f})$', color='red')
plt.xlabel('Botão [ns]')
plt.ylabel('Osciloscopio [ns]')
plt.grid(True)
plt.legend()
plt.annotate(f'R² = {r_squared:.4f}', xy=(0.7, 0.1), xycoords='axes fraction', fontsize=12)
plt.savefig('botao_oscilo.pdf')


# Adicione o valor do R² ao gráfico

plt.show()

