import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Define a fitting function for the sigmoid curve
def sigmoid(x, A, B, C, D):
    return A / (1 + np.exp(B * (x - C))) + D

dados = pd.read_excel('dados.xlsx', sheet_name='Janela')

oscilo = dados['Osciloscopio [ns]']
err_oscilo = dados['incerteza [ns]']
Rc = dados['Rc [1/s]']
err_Rc = dados['incerteza Rc [1/s]']

# Define custom weights based on the uncertainties in both variables
weights = 1 / (err_Rc * err_Rc + err_oscilo * err_oscilo)

parametros_iniciais = [-100, 0.35, 132, 100]

params_sigmoid, params_covariance_sigmoid = curve_fit(sigmoid, oscilo, Rc, p0=parametros_iniciais, sigma=weights, maxfev=2000)

parametros_erros_sigmoid = np.sqrt(np.diag(params_covariance_sigmoid))

x_fit_sigmoid = np.linspace(min(oscilo), max(oscilo), 1000)
y_fit_sigmoid = sigmoid(x_fit_sigmoid, *params_sigmoid)

r2_sigmoid = r2_score(Rc, sigmoid(oscilo, *params_sigmoid))

chi_squared_sigmoid = np.sum(weights * (Rc - sigmoid(oscilo, *params_sigmoid)) ** 2)

ndf_sigmoid = len(oscilo) - len(params_sigmoid)

plt.errorbar(oscilo, Rc, xerr=err_oscilo, yerr=err_Rc, fmt='.', ecolor='red', elinewidth=2, capsize=2, label='Pontos experimentais', color='black')
plt.plot(x_fit_sigmoid, y_fit_sigmoid, label=f'Ajuste: y=a/(1+exp(b(x-c)))+d', color='blue')
plt.xlabel(f'$\\tau$ [ns]')
plt.ylabel(f'$R_c$ [1/s]')
plt.grid(True)
plt.legend()
#plt.savefig('sigmoid_fit.pdf')
plt.show()

print('y=a/(1+exp(b(x-c)))+d')
print('Parâmetros ajustados (sigmoid):', params_sigmoid)
print('Erros dos parâmetros (sigmoid):', parametros_erros_sigmoid)
print(f'Chi-squared ($\chi^2$) (sigmoid): {chi_squared_sigmoid:.4f}')
print('Degrees of Freedom (ndf) (sigmoid):', ndf_sigmoid)
print(f'Coefficient of Determination ($R^2$) (sigmoid): {r2_sigmoid:.4f}')
