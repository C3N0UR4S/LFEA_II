import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

# Função gaussiana para o ajuste
def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

# Carregando os dados
dados = pd.read_excel('dados.xlsx', sheet_name='corr_angular')
theta = dados['theta']
r_corr = dados['Rc corr']
e_r_corr = dados['eRc corr']

# Ajuste gaussiano
param_ini = [1.0, 0.0, 1.0]  # Valores iniciais dos parâmetros (A, mu, sigma)
param_otim, cov = curve_fit(gauss, theta, r_corr, p0=param_ini, sigma=e_r_corr)

# Extrair os valores dos parâmetros de ajuste e seus erros
A, mu, sigma = param_otim
err_A, err_mu, err_sigma = np.sqrt(np.diag(cov))

# Calcular o qui-quadrado e o grau de liberdade
chi2 = np.sum(((r_corr - gauss(theta, A, mu, sigma)) / r_corr) ** 2)
ndf = len(theta) - len(param_ini)  # Grau de liberdade

# Calcular o coeficiente de determinação (R²)
residuals = r_corr - gauss(theta, A, mu, sigma)
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((r_corr - np.mean(r_corr)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

# Imprimir os resultados
print(f'Parâmetros de ajuste:')
print(f'A = {A}, Erro A = {err_A}')
print(f'mu = {mu}, Erro mu = {err_mu}')
print(f'sigma = {sigma}, Erro sigma = {err_sigma}')
print(f'Qui-quadrado = {chi2}')
print(f'Grau de liberdade = {ndf}')
print(f'R² = {r_squared}')

# Traçar o ajuste
x_fit = np.linspace(min(theta), max(theta), 1000)
y_fit = gauss(x_fit, A, mu, sigma)


plt.errorbar(theta, r_corr, yerr=e_r_corr, fmt='.', ecolor='red', capsize=1, label='Pontos experimentais')
plt.plot(x_fit, y_fit, 'r-', label='Ajuste gaussiano', color = 'green')
plt.axvline(x = mu, color='pink', linestyle='--')
plt.xlabel(f'$\\theta [^\circ]$')
plt.ylabel(f'$R_c [1/s]$')
plt.grid(True)
plt.legend()
plt.show()



