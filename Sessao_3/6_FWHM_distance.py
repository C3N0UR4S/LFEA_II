import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

# Read your data
dados = pd.read_excel('dados.xlsx', sheet_name='theta distancia')
fwhm, efwhm, d = dados['FWHM'], dados['eFWHM'], dados['d']

# Define the linear function
def linear(x, m, b):
    return m * x + b

# Define the constant function
def constant(x, c):
    return c

# Fit the data to the linear model
params_linear, covariance_linear = curve_fit(linear, d, fwhm, sigma=efwhm)

# Fit the data to the constant model
params_constant, covariance_constant = curve_fit(constant, d, fwhm, sigma=efwhm)

# Get the optimized parameters
m_linear, b_linear = params_linear
c_constant = params_constant[0]

# Calculate residuals
residuals_linear = fwhm - linear(d, m_linear, b_linear)
residuals_constant = fwhm - constant(d, c_constant)

linear_fit = linear(d, m_linear, b_linear)
const_fit = constant(d, c_constant)

# Calculate the chi-squared and degrees of freedom for both models
chi_squared_linear = np.sum((residuals_linear / linear_fit) ** 2)
chi_squared_constant = np.sum((residuals_constant / const_fit) ** 2)
ndf = len(d) - 2  # Number of data points minus number of parameters



# Calculate the R-squared for both models
ss_tot = np.sum((fwhm - np.mean(fwhm)) ** 2)
ss_res_linear = np.sum(residuals_linear ** 2)
r_squared_linear = 1 - (ss_res_linear / ss_tot)

# Print the optimized parameters, chi-squared, ndf, and R-squared for both models
print('Linear Fit Parameters:')
print(f'Slope (m): {m_linear} ± {np.sqrt(covariance_linear[0, 0])}')
print(f'Intercept (b): {b_linear} ± {np.sqrt(covariance_linear[1, 1])}')
print(f'Chi-squared: {chi_squared_linear}')
print(f'Degrees of Freedom: {ndf}')
print(f'R-squared: {r_squared_linear}')

print('\nConstant Fit Parameters:')
print(f'Constant (c): {c_constant} ± {np.sqrt(covariance_constant[0, 0])}')
print(f'Chi-squared: {chi_squared_constant}')
print(f'Degrees of Freedom: {ndf}')

d_v=np.linspace(min(d), max(d))
linear_fit = linear(d_v, m_linear, b_linear)
const_fit = linear(d_v, 0, c_constant)

# Plot the data, linear fit, and constant fit
plt.errorbar(d, fwhm, yerr=efwhm, capsize=2, ecolor='red', color='black', fmt='.', markersize=0.5)
plt.plot(d_v, linear_fit, label=f'$y = mx+b$')
plt.plot(d_v, const_fit, label=f'$y = b$')
plt.ylim(12, 15)
plt.grid(True)
plt.xlabel('d [in]')
plt.ylabel(f'$FWHM$ $[\\degree]$')
plt.legend()
plt.show()

