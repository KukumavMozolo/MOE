import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




def get_4dfunction(x,y):
    res = np.exp(-0.5 *(x*x + y*y))
    return res



x = np.linspace(-3, 3., 100)
y = np.linspace(-3., 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.empty_like(X)
for xidx, xs in enumerate(x):
    for yidx, ys in enumerate(y):
        Z[xidx,yidx] = get_4dfunction(xs,ys)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_zlim(-1.01, 1.01)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

print(np.mean(Z))


