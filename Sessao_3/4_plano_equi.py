import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

dados = pd.read_excel('dados.xlsx', sheet_name= 'Plano equidistante')

_3_x_theta, _3_x_Rc, _3_x_Rc_err = dados['theta [graus]'][0:12], dados['Rc corr'][0:12], dados['eRc corr'][0:12]
_2_x_theta, _2_x_Rc, _2_x_Rc_err = dados['theta [graus]'][12:22], dados['Rc corr'][12:22], dados['eRc corr'][12:22]
_1_x_theta, _1_x_Rc, _1_x_Rc_err = dados['theta [graus]'][22:32], dados['Rc corr'][22:32], dados['eRc corr'][22:32]
x_1_theta, x_1_Rc, x_1_Rc_err = dados['theta [graus]'][32:42], dados['Rc corr'][32:42], dados['eRc corr'][32:42]
x_2_theta, x_2_Rc, x_2_Rc_err = dados['theta [graus]'][42:52], dados['Rc corr'][42:52], dados['eRc corr'][42:52]
x_3_theta, x_3_Rc, x_3_Rc_err = dados['theta [graus]'][52:], dados['Rc corr'][52:], dados['eRc corr'][52:]

theta_err = 0.52

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

params_3, covariance_3 = curve_fit(gaussian, _3_x_theta, _3_x_Rc, sigma=_3_x_Rc_err, p0=(100, -25, 1))
params_2, covariance_2 = curve_fit(gaussian, _2_x_theta, _2_x_Rc, sigma=_2_x_Rc_err, p0=(100, -15, 1))
params_1, covariance_1 = curve_fit(gaussian, _1_x_theta, _1_x_Rc, sigma=_1_x_Rc_err, p0=(100, -10, 1))
params_x1, covariance_x1 = curve_fit(gaussian, x_1_theta, x_1_Rc, sigma=x_1_Rc_err, p0=(100, 5, 1))
params_x2, covariance_x2 = curve_fit(gaussian, x_2_theta, x_2_Rc, sigma=x_2_Rc_err, p0=(100, 15, 1))
params_x3, covariance_x3 = curve_fit(gaussian, x_3_theta, x_3_Rc, sigma=x_3_Rc_err, p0=(100, 20, 1))

x_values = np.linspace(min(_3_x_theta), max(x_3_theta), 1000)

plt.errorbar(_3_x_theta, _3_x_Rc, xerr=0.52, yerr = _3_x_Rc_err, color = 'black', capsize = 2,fmt='.', ecolor='black', markersize=0.5 )
plt.errorbar(_2_x_theta, _2_x_Rc, xerr=0.52, yerr = _2_x_Rc_err,  color = 'red', capsize = 2,fmt='.', ecolor='red', markersize=0.5 )
plt.errorbar(_1_x_theta, _1_x_Rc, xerr=0.52, yerr = _1_x_Rc_err,  color = 'green', capsize = 2,fmt='.', ecolor='green', markersize=0.5 )
plt.errorbar(x_1_theta, x_1_Rc, xerr=0.52, yerr = x_1_Rc_err,  color = 'blue', capsize = 2,fmt='.', ecolor='blue', markersize=0.5 )
plt.errorbar(x_2_theta, x_2_Rc, xerr=0.52, yerr = x_2_Rc_err,  color = 'brown', capsize = 2,fmt='.', ecolor='brown', markersize=0.5 )
plt.errorbar(x_3_theta, x_3_Rc, xerr=0.52, yerr = x_3_Rc_err, color = 'pink', capsize = 2,fmt='.', ecolor='pink', markersize=0.5 )
plt.plot(x_values, gaussian(x_values, *params_3), label='x = -1.5', color='black')
plt.plot(x_values, gaussian(x_values, *params_2), label='x = -1.0', color='red')
plt.plot(x_values, gaussian(x_values, *params_1), label='x = -0.5', color='green')
plt.plot(x_values, gaussian(x_values, *params_x1), label='x = 0.5', color='blue')
plt.plot(x_values, gaussian(x_values, *params_x2), label='x = 1.0', color='brown')
plt.plot(x_values, gaussian(x_values, *params_x3), label='x = 1.5', color='pink')
plt.xlabel(f'$\\theta $ [$\degree$]')
plt.ylabel(f'$R_c$ [1/s]')
plt.legend()
plt.grid(True)
plt.show()

def r_squared(data, fit):
    ss_res = np.sum((data - fit)**2)
    ss_tot = np.sum((data - np.mean(data))**2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

r2_3 = r_squared(_3_x_Rc, gaussian(_3_x_theta, *params_3))
r2_2 = r_squared(_2_x_Rc, gaussian(_2_x_theta, *params_2))
r2_1 = r_squared(_1_x_Rc, gaussian(_1_x_theta, *params_1))
r2_x1 = r_squared(x_1_Rc, gaussian(x_1_theta, *params_x1))
r2_x2 = r_squared(x_2_Rc, gaussian(x_2_theta, *params_x2))
r2_x3 = r_squared(x_3_Rc, gaussian(x_3_theta, *params_x3))

def chi_squared(data, fit, errors, num_params):
    residuals = data - fit
    chi2 = np.sum((residuals / data) ** 2)
    ndf = len(data) - num_params  # Graus de liberdade
    return chi2, ndf

chi2_3, ndf_3 = chi_squared(_3_x_Rc, gaussian(_3_x_theta, *params_3), _3_x_Rc_err, len(params_3))
chi2_2, ndf_2 = chi_squared(_2_x_Rc, gaussian(_2_x_theta, *params_2), _2_x_Rc_err, len(params_2))
chi2_1, ndf_1 = chi_squared(_1_x_Rc, gaussian(_1_x_theta, *params_1), _1_x_Rc_err, len(params_1))
chi2_x1, ndf_x1 = chi_squared(x_1_Rc, gaussian(x_1_theta, *params_x1), x_1_Rc_err, len(params_x1))
chi2_x2, ndf_x2 = chi_squared(x_2_Rc, gaussian(x_2_theta, *params_x2), x_2_Rc_err, len(params_x2))
chi2_x3, ndf_x3 = chi_squared(x_3_Rc, gaussian(x_3_theta, *params_x3), x_3_Rc_err, len(params_x3))

print("Parameters for x = -1.5 in:", params_3)
print("Covariance matrix for x = -1.5 in:", covariance_3)
print("R-squared for x = -1.5 in:", r2_3)
print("Chi-squared and ndf for x = -1.5 in:", chi2_3, ndf_3)

print("Parameters for x = -1.0 in:", params_2)
print("Covariance matrix for x = -1.0 in:", covariance_2)
print("R-squared for x = -1.0 in:", r2_2)
print("Chi-squared and ndf for x = -1.0 in:", chi2_2, ndf_2)

print("Parameters for x = -0.5 in:", params_1)
print("Covariance matrix for x = -0.5 in:", covariance_1)
print("R-squared for x = -0.5 in:", r2_1)
print("Chi-squared and ndf for x = -0.5 in:", chi2_1, ndf_1)

print("Parameters for x = 0.5 in:", params_x1)
print("Covariance matrix for x = 0.5 in:", covariance_x1)
print("R-squared for x = 0.5 in:", r2_x1)
print("Chi-squared and ndf for x = 0.5 in:", chi2_x1, ndf_x1)

print("Parameters for x = 1.0 in:", params_x2)
print("Covariance matrix for x = 1.0 in:", covariance_x2)
print("R-squared for x = 1.0 in:", r2_x2)
print("Chi-squared and ndf for x = 1.0 in:", chi2_x2, ndf_x2)

print("Parameters for x = 1.5 in:", params_x3)
print("Covariance matrix for x = 1.5 in:", covariance_x3)
print("R-squared for x = 1.5 in:", r2_x3)
print("Chi-squared and ndf for x = 1.5 in:", chi2_x3, ndf_x3)                                             