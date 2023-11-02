import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

dados = pd.read_excel('dados.xlsx', sheet_name= 'xx')

x = dados['x [cm]']
Rc = dados['Rc corr']
eRc = dados['eRc corr']

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

params, covariance = curve_fit(gaussian, x,  Rc, sigma=eRc, p0=(100, 0, 1))
x_values = np.linspace(min(x), max(x), 1000)

plt.errorbar(x,  Rc, yerr = eRc, color = 'black', capsize = 2,fmt='.', ecolor='red', markersize=0.5 )
plt.plot(x_values, gaussian(x_values, *params), label='ajuste', color='green')
plt.axvline(params[1], color='blue', linestyle='--', label=f'x = {params[1]:.3f}', linewidth=1)
plt.xlabel(f'$\\theta $ [$\degree$]')
plt.xlabel(f'$\\theta $ [$\degree$]')
plt.ylabel(f'$R_c$ [1/s]')
plt.title('Variação em xx com $\\theta = 0^\circ$')
plt.legend()
plt.grid(True)
plt.show()

def r_squared(data, fit):
    ss_res = np.sum((data - fit)**2)
    ss_tot = np.sum((data - np.mean(data))**2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

r2 = r_squared(x, gaussian(x, *params))

def chi_squared(data, fit, errors, num_params):
    residuals = data - fit
    chi2 = np.sum((residuals / errors) ** 2)
    ndf = len(data) - num_params  # Graus de liberdade
    return chi2, ndf

chi2, ndf = chi_squared(Rc, gaussian(x, *params), eRc, len(params))

print("Parameters:", params)
print("Covariance matrix:", covariance)
print("R-squared :", r2)
print("Chi-squared and ndf :", chi2, ndf)

