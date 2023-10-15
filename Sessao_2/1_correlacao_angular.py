import openpyxl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

#ler o ficheiro


# Defina a função gaussiana que você deseja ajustar
#def gaussian(x, A, mu, sigma):
#    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))
#
## Realize o ajuste usando curve_fit
#params, cov = curve_fit(gaussian, ang_dg, Rc, sigma=err_Rc)
#
## Extraia os parâmetros ajustados e seus erros
#A, mu, sigma = params
#A_err, mu_err, sigma_err = np.sqrt(np.diag(cov))
#
## Calcule o qui-quadrado (chi-squared) e o número de graus de liberdade (ndf)
#chi_squared = np.sum(((Rc - gaussian(ang_dg, A, mu, sigma)) / err_Rc)**2)
#ndf = len(ang_dg) - len(params)
#
## Crie um novo conjunto de pontos para o ajuste gaussiano
#x_fit = np.linspace(min(ang_dg), max(ang_dg), 1000)
#y_fit = gaussian(x_fit, A, mu, sigma)
#
#
#plt.errorbar(ang_dg, Rc, err_Rc, fmt='.', capsize=2, label='Dados', markersize=2, ecolor= 'red', color = 'black')
#plt.plot(x_fit, y_fit, label='Ajuste Gaussiano', color='green')
#plt.xlabel(r'$\theta$ [°]')
#plt.ylabel(r'$R_c [cnt/s]$')
#
## Imprima os valores dos parâmetros e seus erros
#print(f'A = {A:.2f} +/- {A_err:.2f}')
#print(f'mu = {mu:.2f} +/- {mu_err:.2f}')
#print(f'sigma = {sigma:.2f} +/- {sigma_err:.2f}')
#
## Imprima o qui-quadrado e o número de graus de liberdade
#print(f'Qui-quadrado = {chi_squared:.2f}')
#print(f'NDF = {ndf}')
#print(f'Qui-quadrado/Ndf = {(chi_squared/ndf):.2f}')
#
#
#plt.grid(True)
#plt.legend()
#plt.show()

