import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




def func( a, b):
    def x(a,b):
        return  np.sin(a*5)

    def y(a,b):
        if(a >0 and b >0):
            return np.sin(b*5) +0.5*np.exp(-(b-np.pi/10.0)**2)
        else:
            return np.sin(b*5)
    return x(a,b)*y(a,b)



x = np.linspace(-0.5, 0.6)
y = np.linspace(-0.5, 0.6)
X, Y = np.meshgrid(x, y)
Z = np.empty_like(X)

for xidx, xs in enumerate(x):
    for yidx, ys in enumerate(y):
        Z[xidx,yidx] = func(xs,ys)



fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_zlim(-1.01, 1.01)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

ax.set_xlabel('X')
ax.set_ylabel('T')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()