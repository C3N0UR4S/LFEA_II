import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Ler os dados do Excel
df = pd.read_excel("C:/Users/samhe/Desktop/3ºAno-S1/LFEA II/Análise/Dados reconstrução tomográfica.xlsx", sheet_name='Folha1')

# Extrair os valores das colunas
z = df['Z'].values
x = df['Valores de X'].values
y = df['Valores de Y'].values
print(z)

# Definir o tamanho da matriz 3D (ajuste conforme necessário)
matrix_size = 100
z_max = z.max()
z_min = z.min()

# Interpolação dos pontos
xi, yi = np.linspace(x.min(), x.max(), matrix_size), np.linspace(y.min(), y.max(), matrix_size)
zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='cubic')


# Criação do gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Inverter a direção do eixo X no gráfico
ax.invert_xaxis()

# Cria as coordenadas 3D para o gráfico
x3d, y3d = np.meshgrid(xi, yi)

# Plota a superfície 3D com preenchimento e cores personalizadas
norm = Normalize(vmin=z.min(), vmax=z.max())
colors = plt.cm.viridis(norm(zi))
surf = ax.plot_surface(x3d, y3d, zi, facecolors=colors, rstride=1, cstride=1, linewidth=0, antialiased=False)

# Plota a superfície 3D com preenchimento
ax.plot_surface(x3d, y3d, zi, cmap='viridis')

# Configurações estéticas
ax.set_xlabel('X [in]')
ax.set_ylabel('Y [in]')
ax.set_zlabel('Counts [/s]')
ax.set_title('Reconstrução Tomográfica 3D')

# Adiciona uma barra de cores personalizada
sm = ScalarMappable(cmap=plt.cm.viridis, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.05, shrink=0.5)
cbar.set_label('Counts [/s]')

plt.show()
