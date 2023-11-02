import numpy as np
import matplotlib.pyplot as plt
from pandas import read_excel
import math

# Ler os dados do Excel
df = read_excel("C:/Users/samhe/Desktop/3ºAno-S1/LFEA II/dados_s4.xlsx", sheet_name='Folha3', skiprows=1)

# Sigmas obtidos
sigma_1 = 35.4
sigma_2 = 24.8
sigma_3 = 16.3
media_1_1 = 84.153
media_1_2 = 278
media_2_1 = 87.464
media_2_2 = 273.467
media_3_1 = 87.268
media_3_2 = 271.189

# Dividir os dados em X e Y
x = df['phi'].values
y1 = df['RccCorrigido1'].values
y2 = df['RccCorrigido2'].values
y3 = df['RccCorrigido3'].values
r_1 = 0.5
r_2 = 1
r_3 = 1.5

#Converter para radianos
x_rad = [math.radians(angulo) for angulo in x]

# Lista para armazenar as coordenadas dos pontos com y > 10
coordenadas_marcadores1 = []
coordenadas_marcadores2 = []
coordenadas_marcadores3 = []


# Loop pelos pontos
for i in range(len(x)):
    if ((media_1_1 - (3 * sigma_1)) < x[i] < (media_1_1 + (3 * sigma_1))) or ((media_1_2 - (3 * sigma_1)) < x[i] < (media_1_2 + (3 * sigma_1))):
        coordenadas_marcadores1.append((np.cos(x_rad[i]) * r_1, np.sin(x_rad[i]) * r_1))

for i in range(len(x)):
    if ((media_2_1 - (3 * sigma_2)) < x[i] < (media_2_1 + (3 * sigma_2))) or ((media_2_2 - (3 * sigma_2)) < x[i] < (media_2_2 + (3 * sigma_2))):
        coordenadas_marcadores2.append((np.cos(x_rad[i]) * r_2, np.sin(x_rad[i]) * r_2))

for i in range(len(x)):
    if ((media_3_1 - (3 * sigma_3)) < x[i] < (media_3_1 + (3 * sigma_3))) or ((media_3_2 - (3 * sigma_3)) < x[i] < (media_3_2 + (3 * sigma_3))):
        coordenadas_marcadores3.append((np.cos(x_rad[i]) * r_3, np.sin(x_rad[i]) * r_3))

# Converta a lista em uma matriz NumPy
coordenadas_marcadores1 = np.array(coordenadas_marcadores1)
coordenadas_marcadores2 = np.array(coordenadas_marcadores2)
coordenadas_marcadores3 = np.array(coordenadas_marcadores3)


# Plotar os dados e os ajustes
plt.figure(figsize=(10, 10))
# Se houver coordenadas de marcadores, plote-as
if coordenadas_marcadores1.size > 0:
    plt.scatter(coordenadas_marcadores1[:, 0], coordenadas_marcadores1[:, 1], label='R=0.5 in', s=10, color='red')

if coordenadas_marcadores2.size > 0:
    plt.scatter(coordenadas_marcadores2[:, 0], coordenadas_marcadores2[:, 1], label='R=1.0 in', s=10, color='green')

if coordenadas_marcadores3.size > 0:
    plt.scatter(coordenadas_marcadores3[:, 0], coordenadas_marcadores3[:, 1], label='R=1.5 in', s=10, color='blue')

#plt.scatter(coordenadas_marcadores1[0], coordenadas_marcadores1[1], label='R=0.5 in', s=10, color='red')
#plt.scatter(coordenadas_marcadores2[0], coordenadas_marcadores2[1], label='R=0.5 in', s=10, color='green')
#plt.scatter(coordenadas_marcadores3[0], coordenadas_marcadores3[1], label='R=1.5 in', s=10, color='blue')

# Ajuste os limites dos eixos
#plt.xlim(x.min(), x.max())  # Limites automáticos com base nos dados
#plt.ylim(y.min(), y.max())  # Limites automáticos com base nos dados
plt.axis('equal')
plt.xlabel(f'x [in]')
plt.ylabel(f'y [in]')
plt.legend()
plt.title('Área de deteção')
plt.grid(True)

# Configurações estéticas dos eixos
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.2', color='black')
plt.grid(which='minor', linestyle=':', linewidth='0.2', color='gray')

#Print time
print((media_1_1 - (3 * sigma_1)))
print((media_1_1 + (3 * sigma_1)))
print((media_1_2 - (3 * sigma_1)))
print((media_1_2 + (3 * sigma_1)))
print((media_2_1 - (3 * sigma_2)))
print((media_2_1 + (3 * sigma_2)))
print((media_2_2 - (3 * sigma_2)))
print((media_2_2 + (3 * sigma_2)))
print((media_3_1 - (3 * sigma_3)))
print((media_3_1 + (3 * sigma_3)))
print((media_3_2 - (3 * sigma_3)))
print((media_3_2 + (3 * sigma_3)))
print(coordenadas_marcadores2)
print("-----------")
print(coordenadas_marcadores3)


plt.show()