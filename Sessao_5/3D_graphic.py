import numpy as np
import matplotlib.pyplot as plt
from pandas import read_excel
import math
from mpl_toolkits.mplot3d import Axes3D

# Ler os dados do Excel
df = read_excel("C:/Users/samhe/Desktop/3ºAno-S1/LFEA II/dados_s4.xlsx", sheet_name='Folha3', skiprows=1)

# Dividir os dados em X e Y
x = df['phi'].values
y1 = df['RccCorrigido1'].values
y2 = df['RccCorrigido2'].values
y3 = df['RccCorrigido3'].values
r_1 = 0.5
r_2 = 1
r_3 = 1.5

# Converter ângulos para radianos
x_rad = [math.radians(angulo) for angulo in x]

# Calcular as coordenadas x, y e z
x_coords1 = [r_1 * np.cos(rad) for rad in x_rad]
y_coords1 = [r_1 * np.sin(rad) for rad in x_rad]
x_coords2 = [r_2 * np.cos(rad) for rad in x_rad]
y_coords2 = [r_2 * np.sin(rad) for rad in x_rad]
x_coords3 = [r_3 * np.cos(rad) for rad in x_rad]
y_coords3 = [r_3 * np.sin(rad) for rad in x_rad]

# Crie uma figura 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Crie um gráfico 3D de dispersão com os dados de y1
ax.scatter(x_coords1, y_coords1, y1, label='y1', c='r', marker='o')

# Adicione os dados de y2 e y3
ax.scatter(x_coords2, y_coords2, y2, label='y2', c='g', marker='^')
ax.scatter(x_coords3, y_coords3, y3, label='y3', c='b', marker='s')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Gráfico 3D com eixos x e y transformados')

# Exiba a legenda
ax.legend()

plt.show()