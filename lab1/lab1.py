import math
import sympy as s
import matplotlib.pyplot as plot
import numpy as n
from matplotlib.animation import FuncAnimation

t = s.Symbol('t')

x = s.sin(t)
y = s.sin(2 * t)

Vx = s.diff(x)
Vy = s.diff(y)

step = 1000
T = n.linspace(0, 10, step)
X = n.zeros_like(T)
Y = n.zeros_like(T)
VX = n.zeros_like(T)
VY = n.zeros_like(T)
for i in n.arange(len(T)):
    X[i] = s.Subs(x, t, T[i])
    Y[i] = s.Subs(y, t, T[i])
    VX[i] = s.Subs(Vx, t, T[i])
    VY[i] = s.Subs(Vy, t, T[i])

fgr = plot.figure()
grf = fgr.add_subplot(1, 1, 1)
grf.axis('equal')
grf.set(xlim=[-2, 2], ylim=[-2, 2])
grf.plot(X, Y)

Pnt = grf.plot(X[0], Y[0], marker='o')[0]
Vpl = grf.plot([X[0], X[0] + VX[0]], [Y[0], Y[0] + VY[0]], 'r')[0]

def Vect_arrow(VecX, VecY, X, Y):
    a = 0.3
    b = 0.2
    Arrx = n.array([-a, 0, -a])
    Arry = n.array([b, 0, -b])

    phi = math.atan2(VecY, VecX)

    RotX = Arrx * n.cos(phi) - Arry * n.sin(phi)
    RotY = Arrx * n.sin(phi) + Arry * n.cos(phi)

    Arrx = RotX + X + VecX
    Arry = RotY + Y + VecY

    return Arrx, Arry

ArVX, ArVY = Vect_arrow(VX[0], VY[0], X[0], Y[0])
Varr = grf.plot(ArVX, ArVY, 'r')[0]

def anim(i):
    Pnt.set_data(X[i], Y[i])
    Vpl.set_data([X[i], X[i] + VX[i]], [Y[i], Y[i]+VY[i]])
    Varr.set_data(Vect_arrow(VX[i], VY[i], X[i], Y[i]))
    return

an = FuncAnimation(fgr, anim, frames=step, interval=1)

fgr.show()
