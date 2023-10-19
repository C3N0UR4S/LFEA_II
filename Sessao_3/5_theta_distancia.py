import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np


dados = pd.read_excel('dados.xlsx', sheet_name= 'theta distancia')

theta, etheta, d = dados['theta'], dados['etheta'], dados['d']

def linear(x, m, b):
    return m*x + b

params, covariance = curve_fit(linear, d, theta, sigma=etheta)

m, b = params

theta_fit = linear(d, m, b)

residuals = theta - theta_fit

chi_squared = np.sum((residuals / theta_fit) ** 2)
ndf = len(d) - 2  # Number of data points minus number of parameters

ss_tot = np.sum((theta - np.mean(theta)) ** 2)
ss_res = np.sum(residuals ** 2)
r_squared = 1 - (ss_res / ss_tot)

m_error, b_error = np.sqrt(np.diag(covariance))

print(f'Slope (m): {m} ± {m_error}')
print(f'Intercept (b): {b} ± {b_error}')
print(f'Chi-squared: {chi_squared}')
print(f'Degrees of Freedom: {ndf}')
print(f'R-squared: {r_squared}')

plt.errorbar(d, theta, yerr=etheta, capsize=2, ecolor='red', color='black', fmt='.', markersize=0.5)
plt.plot(d, theta_fit, label=f'Ajuste: $y=mx+b$', color='green', linewidth = 1)
plt.grid(True)
plt.xlabel('d [in]')
plt.ylabel(f'$\\theta_0$ $[\\degree]$')
plt.legend()
plt.show()








