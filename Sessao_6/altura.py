import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gaussian(x, a, b, c):
    return a * np.exp(-((x - b)**2) / (2 * c**2))


dados_10 = pd.read_excel('Sessao_5/dados.xlsx', sheet_name='10')
dados_5 = pd.read_excel('Sessao_5/dados.xlsx', sheet_name='5')

y_10 = dados_10['h [mm]']
y_5 = dados_5['h [mm]']
Rc_10 = dados_10['Rc corr']
eRc_10 = dados_10['eRc corr']
Rc_5 = dados_5['Rc corr']
eRc_5 = dados_5['eRc corr']

# Ajuste da gaussiana para Rc_10 com erros
params_10, covariance_10 = curve_fit(gaussian, y_10, Rc_10, p0=[1, 10, 1], sigma=eRc_10)

# Ajuste da gaussiana para Rc_5 com erros
params_5, covariance_5 = curve_fit(gaussian, y_5, Rc_5, p0=[1, 10, 1], sigma=eRc_5)

a_10, b_10, c_10 = params_10
a_5, b_5, c_5 = params_5

# Qui-quadrado e número de graus de liberdade (ndf) para Rc_10
chi_squared_10 = np.sum(((Rc_10 - gaussian(y_10, *params_10)) / eRc_10)**2)
ndf_10 = len(y_10) - len(params_10)

# Qui-quadrado e número de graus de liberdade (ndf) para Rc_5
chi_squared_5 = np.sum(((Rc_5 - gaussian(y_5, *params_5)) / eRc_5)**2)
ndf_5 = len(y_5) - len(params_5)

# Plot dos dados, curva ajustada e erros para Rc_10
plt.errorbar(y_10, Rc_10, yerr=eRc_10, label=f'10 $\mu Ci$ em cima', capsize=1, markersize=2, ecolor='red', color='red', fmt='.')
x_range_10 = np.linspace(min(y_10), max(y_10), 100)
plt.plot(x_range_10, gaussian(x_range_10, a_10, b_10, c_10), label='Ajuste 10 $\mu Ci$ em cima', color='blue')

# Plot dos dados, curva ajustada e erros para Rc_5
plt.errorbar(y_5, Rc_5, yerr=eRc_5, label=f'5 $\mu Ci$ em cima', capsize=1, markersize=2, ecolor='green', color='green', fmt='.')
x_range_5 = np.linspace(min(y_5), max(y_5), 100)
plt.plot(x_range_5, gaussian(x_range_5, a_5, b_5, c_5), label='Ajuste 5 $\mu Ci$ em cima', color='brown')

plt.grid(True)
plt.legend()
plt.xlabel('z [mm]')
plt.ylabel(f'$R_c$ [1/s]')
plt.show()

# Imprimir os resultados
print("Valores dos parâmetros para 10 µCi em cima:")
print(f"a_10: {a_10}")
print(f"b_10: {b_10}")
print(f"c_10: {c_10}")
print("\nValores dos parâmetros para 5 µCi em cima:")
print(f"a_5: {a_5}")
print(f"b_5: {b_5}")
print(f"c_5: {c_5}")
print("\nQui-quadrado e ndf para 10 µCi em cima:")
print(f"Qui-quadrado_10: {chi_squared_10}")
print(f"ndf_10: {ndf_10}")
print("\nQui-quadrado e ndf para 5 µCi em cima:")
print(f"Qui-quadrado_5: {chi_squared_5}")
print(f"ndf_5: {ndf_5}")


