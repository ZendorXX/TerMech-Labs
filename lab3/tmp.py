import matplotlib.pyplot as p
import numpy as n
from scipy.integrate import odeint

# y' = sin(t) -> y(t) = -cos(t) + C1
def dy(y, t):
    dy = n.sin(t)
    return dy

y0 = 3
T = n.linspace(0, 10, 100)

Y = odeint(dy, y0, T)

yt = -n.cos(T) + 4

fgr = p.figure()
plt = fgr.add_subplot(2, 1, 1)
plt.plot(T, Y)
plt = fgr.add_subplot(2, 1, 2)
plt.plot(T, yt)

fgr.show()