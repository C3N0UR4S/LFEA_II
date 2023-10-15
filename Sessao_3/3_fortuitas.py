import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

dados = pd.read_excel('dados.xlsx', sheet_name='Fortuitas')

oscilo = dados['Osciloscopio']
err_oscilo = dados['incerteza']
teo_fort = dados['Rf teo']
err_teo_fort = dados['eRf teo']
exp_fort = dados['Rf exp']
err_exp_fort = dados['eRf exp']

weights = 1 / (err_oscilo * err_oscilo)

# Ajuste linear teórico
X = sm.add_constant(oscilo)  # Adicionar uma coluna de uns para o termo constante
model_teo = sm.WLS(teo_fort, X, weights=1 / (err_teo_fort * err_teo_fort))
result_teo = model_teo.fit()

# Ajuste linear experimental
model_exp = sm.WLS(exp_fort, X, weights=1 / (err_exp_fort * err_exp_fort))
result_exp = model_exp.fit()

# Obter os erros dos parâmetros de ajuste e o R²
params_teo = result_teo.params
stderr_teo = result_teo.bse
rsquared_teo = result_teo.rsquared

params_exp = result_exp.params
stderr_exp = result_exp.bse
rsquared_exp = result_exp.rsquared

# Imprimir os resultados
print("Ajuste Teórico:")
print("Parâmetros:", params_teo)
print("Erros dos Parâmetros:", stderr_teo)
print("R²:", rsquared_teo)

print("\nAjuste Experimental:")
print("Parâmetros:", params_exp)
print("Erros dos Parâmetros:", stderr_exp)
print("R²:", rsquared_exp)

plt.errorbar(oscilo, teo_fort, xerr= err_oscilo, yerr= err_teo_fort ,fmt='.', ecolor='blue', color='black', capsize= 1,label=f'$R_f$ teóricas', markersize=0.5)
plt.errorbar(oscilo, exp_fort, xerr= err_oscilo, yerr= err_exp_fort, fmt='.', ecolor='red', color='black', capsize= 1, label=f'$R_f$ experimentais', markersize=0.5)
plt.plot(oscilo, result_teo.fittedvalues, label='Ajuste aos pontos teóricos', color='brown')
plt.plot(oscilo, result_exp.fittedvalues, label='Ajuste aos pontos experimentais', color='green')

plt.xlabel(f'$\\tau$ [ns]')
plt.ylabel(f'$R_f$ [1/s]')
plt.title(f'Coincidências fortuitas, $R_f$')
plt.grid(True)
plt.legend()
plt.savefig('3_fortuitas.pdf')
plt.show()