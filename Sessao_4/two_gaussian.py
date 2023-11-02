import numpy as np
import matplotlib.pyplot as plt
from pandas import read_excel
from scipy.optimize import curve_fit
import pandas as pd
from sklearn.metrics import r2_score

import math

# Ler os dados do Excel
df = read_excel("C:/Users/samhe/Desktop/3ºAno-S1/LFEA II/dados_s4.xlsx", sheet_name='Folha3', skiprows=1)

# Função que é a soma de duas funções gaussianas
def two_gaussians(x, A1, mu1, sigma1, A2, mu2, sigma2, c):
    gaussian1 = A1 * np.exp(-(x - mu1) ** 2 / (2 * sigma1 ** 2))
    gaussian2 = A2 * np.exp(-(x - mu2) ** 2 / (2 * sigma2 ** 2))
    return gaussian1 + gaussian2 + c

#Função para calcular o coeficiente de determinação (R^2)
def r_squared(y, y_fit):
    residuals = y - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - (ss_res / ss_tot)

# Dividir os dados em X e Y
x = df['phi'].values
x_err = 2

# Três conjuntos de dados Y, um para cada conjunto
y1 = df['RccCorrigido1'].values
y2 = df['RccCorrigido2'].values
y3 = df['RccCorrigido3'].values
y1_err = df['ErroCorrigido1'].values
y2_err = df['ErroCorrigido2'].values
y3_err = df['ErroCorrigido3'].values

# Crie uma grade de pontos mais densa para o ajuste
x_fit = np.linspace(0, 350, 1000)


# Ajustes iniciais para as três gaussianas
params1 = [189, 80, 35, 197, 270, 40, 1500]  # Valores iniciais para A, mu e sigma da primeira gaussiana
params2 = [178, 90, 30, 188, 270, 30, 30]  # Valores iniciais para A, mu e sigma da segunda gaussiana
params3 = [165, 90, 20, 169, 270, 30, 20]  # Valores iniciais para A, mu e sigma da terceira gaussiana

# Matriz de covariância que leva em consideração erros verticais e horizontais
cov_matrix1 = np.diag(y1_err**2)
cov_matrix2 = np.diag(y2_err**2)
cov_matrix3 = np.diag(y3_err**2)


# Realizar os ajustes de curva
popt1, pcov1 = curve_fit(two_gaussians, x, y1, p0=params1)
popt2, pcov2 = curve_fit(two_gaussians, x, y2, p0=params2)
popt3, pcov3 = curve_fit(two_gaussians, x, y3, p0=params3)

# Calcular o erro padrão (desvio padrão) dos parâmetros
perr1 = np.sqrt(np.diag(pcov1))
perr2 = np.sqrt(np.diag(pcov2))
perr3 = np.sqrt(np.diag(pcov3))

# Erros dos parâmetros
error1_A1, error1_mu1, error1_sigma1, error1_A2, error1_mu2, error1_sigma2, error1_c = perr1
error2_A1, error2_mu1, error2_sigma1, error2_A2, error2_mu2, error2_sigma2, error2_c = perr2
error3_A1, error3_mu1, error3_sigma1, error3_A2, error3_mu2, error3_sigma2, error3_c = perr3

# Gerar os valores ajustados
fit1 = two_gaussians(x, *popt1)
fit2 = two_gaussians(x, *popt2)
fit3 = two_gaussians(x, *popt3)


# Calcular o coeficiente de determinação (R^2) considerando as barras de erro
r2_1 = r_squared(y1, fit1)
r2_2 = r_squared(y2, fit2)
r2_3 = r_squared(y3, fit3)

# Gerar os valores ajustados para a grade densa
fit_curve1 = two_gaussians(x_fit, *popt1)
fit_curve2 = two_gaussians(x_fit, *popt2)
fit_curve3 = two_gaussians(x_fit, *popt3)

#Calcule o qui-quadrado (chi-squared) e o número de graus de liberdade (ndf)
chi_squared1 = np.sum(abs(y1 - two_gaussians(x, *popt1))**2 / two_gaussians(x, *popt1))
ndf1 = len(x) - len(popt1)
chi_squared2 = np.sum(abs(y2 - two_gaussians(x, *popt2))**2 / two_gaussians(x, *popt2))
ndf2 = len(x) - len(popt2)
chi_squared3 = np.sum(abs(y3 - two_gaussians(x, *popt3))**2 / abs(two_gaussians(x, *popt3)))
ndf3 = len(x) - len(popt3)

# Plotar os dados e os ajustes
plt.figure(figsize=(10, 8))
plt.errorbar(x, y1, xerr= x_err, yerr= y1_err, fmt='.', ecolor='red', color='black', capsize= 1, markersize=0.5)
plt.errorbar(x, y2, xerr= x_err, yerr= y2_err, fmt='.', ecolor='green', color='black', capsize= 1, markersize=0.5)
plt.errorbar(x, y3, xerr= x_err, yerr= y3_err, fmt='.', ecolor='blue', color='black', capsize= 1, markersize=0.5)

plt.scatter(x, y1, label='R=0.5 in', s=10, color='red')
plt.scatter(x, y2, label='R=1.0 in', s=10, color='green')
plt.scatter(x, y3, label='R=1.5 in', s=10, color='blue')

plt.plot(x_fit, fit_curve1, 'r--', linewidth=1)  # Linha de ajuste
plt.plot(x_fit, fit_curve2, 'g--', linewidth=1)  # Linha de ajuste
plt.plot(x_fit, fit_curve3, 'b--', linewidth=1)  # Linha de ajuste


plt.xlabel(f'$\phi$ [º]')
plt.ylabel(f'$R_C$ corrigido')
plt.legend()
plt.title('Fit coincidências em função do ângulo de rotação phi')
plt.grid(True)


# Configurações estéticas dos eixos
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.2', color='black')
plt.grid(which='minor', linestyle=':', linewidth='0.2', color='gray')

# Criar uma tabela com 4 colunas e 8 linhas
parametros = {
    'Parâmetro': ['Amplitude 1', 'Erro', 'Média 1', 'Erro', 'Desvio padrão 1', 'Erro', 'Amplitude 2', 'Erro', 'Média 2', 'Erro', 'Desvio padrão 2', 'Erro', 'Termo de fundo (c)', 'Erro', 'X^2', f'Ndf', 'R^2'],
    'R=0.5 in': [round(popt1[0], 3), error1_A1, round(popt1[1], 3), error1_mu1, round(popt1[2], 3), error1_sigma1,
                 round(popt1[3], 3), error1_A2, round(popt1[4], 3), error1_mu2, round(popt1[5], 3), error1_sigma2,
                 round(popt1[6], 3), error1_c, chi_squared1, ndf1, r2_1],
    'R=1.0 in': [round(popt2[0], 3), error2_A1, round(popt2[1], 3), error2_mu1, round(popt2[2], 3), error2_sigma1,
                 round(popt2[3], 3), error2_A2, round(popt2[4], 3), error2_mu2, round(popt2[5], 3), error2_sigma2,
                 round(popt2[6], 3), error2_c, chi_squared2, ndf2, r2_2],
    'R=1.5 in': [round(popt3[0], 3), error3_A1, round(popt3[1], 3), error3_mu1, round(popt3[2], 3), error3_sigma1,
                 round(popt3[3], 3), error3_A2, round(popt3[4], 3), error3_mu2, round(popt3[5], 3), error3_sigma2,
                 round(popt3[6], 3), error3_c, chi_squared3, ndf3, r2_3]
}

# Converter para DataFrame do Pandas
df_parametros = pd.DataFrame(parametros)

# Exibir a tabela
print(df_parametros)
plt.show()