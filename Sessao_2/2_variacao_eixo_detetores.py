import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

# Definindo as funções de ajuste
def ajuste_A(y, A, d0_A):
    return A / (d0_A - y) ** 2

def ajuste_B(y, B, d0_B):
    return B / (d0_B + y) ** 2

def ajuste_C(y, C, d0_c):
    return C / (d0_c - np.abs(y)) ** 2 


# Lendo os dados do arquivo Excel
dados = pd.read_excel('dados.xlsx', sheet_name='eixo_detetores')

# Separando os dados em variáveis
y, Ra, eRa, Rb, eRb, Rc, eRc = dados['y [cm]'],\
    dados['Ra corr'], dados['eRa corr'], dados['Rb corr'], dados['eRb corr'], dados['Rc corr'], dados['eRc corr']

# Ajustando os dados a função ajuste_A com consideração aos erros verticais
params_A, covariance_A = curve_fit(ajuste_A, y, Ra, p0=[300, 16], sigma=eRa)
y_fit_A = np.linspace(min(y), max(y), 100)
Ra_fit_A = ajuste_A(y_fit_A, *params_A)

# Ajustando os dados a função ajuste_B com consideração aos erros verticais
params_B, covariance_B = curve_fit(ajuste_B, y, Rb, p0=[300, -16], sigma=eRb)
y_fit_B = np.linspace(min(y), max(y), 100)
Rb_fit_B = ajuste_B(y_fit_B, *params_B)

# Ajustando os dados a função ajuste_C com consideração aos erros verticais
params_C, covariance_C = curve_fit(ajuste_C, y, Rc, p0=[1000, 1], sigma=eRc)
y_fit_C = np.linspace(min(y), max(y), 100)
Rc_fit_C = ajuste_C(y_fit_C, *params_C)


#interseção dos ajustes
inter = (params_A[1]+params_B[1])/(np.sqrt(params_A[0]/params_B[0])+1)-params_B[1]


# Plotando os dados experimentais e os ajustes
plt.errorbar(y, Ra, yerr=eRa, label=f'$R_a$', capsize=2, ecolor='red', fmt='.', markersize=0.5)
plt.plot(y_fit_A, Ra_fit_A, 'r-', label=f'Ajuste $R_a$')
plt.errorbar(y, Rb, yerr=eRb, label=f'$R_b$', capsize=2, ecolor='green', fmt='.', markersize=0.5)
plt.plot(y_fit_B, Rb_fit_B, 'g-', label=f'Ajuste $R_b$')
plt.axvline(x = inter, color='pink', linestyle='--')
plt.xlabel('y [cm]')
plt.ylabel(f'$R_i$ [1/s]')
plt.legend()
plt.grid(True)
plt.show()

# Obtendo os erros dos parâmetros ajustados
errors_A = np.sqrt(np.diag(covariance_A))

# Calculando o coeficiente de determinação (R²)
residuals_A = Ra - ajuste_A(y, *params_A)
ss_res_A = np.sum(residuals_A**2)
ss_tot_A = np.sum((Ra - np.mean(Ra))**2)
r_squared_A = 1 - (ss_res_A / ss_tot_A)

# Calculando o qui-quadrado (χ²)
ndf_A = len(y) - len(params_A)  # Número de graus de liberdade
chi_squared_A = np.sum((residuals_A / eRa)**2)

# Obtendo os erros dos parâmetros ajustados
errors_B = np.sqrt(np.diag(covariance_B))

# Calculando o coeficiente de determinação (R²)
residuals_B = Rb - ajuste_B(y, *params_B)
ss_res_B = np.sum(residuals_B**2)
ss_tot_B = np.sum((Rb - np.mean(Rb))**2)
r_squared_B = 1 - (ss_res_B / ss_tot_B)

# Calculando o qui-quadrado (χ²)
ndf_B = len(y) - len(params_B)  # Número de graus de liberdade
chi_squared_B = np.sum((residuals_B / eRb)**2)



# Exibindo os resultados
print("Ajuste A:")
print(f"Parâmetros A: {params_A}")
print(f"Erros dos parâmetros A: {errors_A}")
print(f"R² A: {r_squared_A}")
print(f"Qui-quadrado A: {chi_squared_A}")
print(f"Número de graus de liberdade A: {ndf_A}")

print("\nAjuste B:")
print(f"Parâmetros B: {params_B}")
print(f"Erros dos parâmetros B: {errors_B}")
print(f"R² B: {r_squared_B}")
print(f"Qui-quadrado B: {chi_squared_B}")
print(f"Número de graus de liberdade B: {ndf_B}")

print(f'Interseção dos ajustes: x = {inter} ')

inter = (params_A[1]+params_B[1])/(np.sqrt(params_A[0]/params_B[0])+1)-params_B[1]


plt.errorbar(y, Rc, yerr= eRc,  label=f'$R_c$', capsize=2, ecolor='orange', fmt='.', markersize=0.5)
plt.plot(y_fit_C, Rc_fit_C, 'g-', label=f'Ajuste $R_c$')
plt.xlabel('y [cm]')
plt.ylabel(f'$R_i$ [1/s]')
plt.legend()
plt.grid(True)
plt.show()

# Obtendo os erros dos parâmetros ajustados
errors_C = np.sqrt(np.diag(covariance_C))

# Calculando o coeficiente de determinação (R²)
residuals_C = Rc - ajuste_C(y, *params_C)
ss_res_C = np.sum(residuals_C**2)
ss_tot_C = np.sum((Rc - np.mean(Rc))**2)
r_squared_C = 1 - (ss_res_C / ss_tot_C)

# Calculando o qui-quadrado (χ²)
ndf_C = len(y) - len(params_C)  # Número de graus de liberdade
chi_squared_C = np.sum((residuals_C / eRc)**2)

print("\nAjuste C:")
print(f"Parâmetros B: {params_C}")
print(f"Erros dos parâmetros B: {errors_C}")
print(f"R² B: {r_squared_C}")
print(f"Qui-quadrado B: {chi_squared_C}")
print(f"Número de graus de liberdade B: {ndf_C}")