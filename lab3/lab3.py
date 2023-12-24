import numpy as np    
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

def SystDiffEq(y, t, P, l, c, mu, g):
    dy = np.zeros_like(y)
    dy[0] = y[2]
    dy[1] = y[3]

    s = y[0]
    phi = y[1]
    ds = y[2]
    dphi = y[3]

    # a11 * s'' + a12 * phi'' = b1
    # a21 * s'' + a22 * phi'' = b2

    a11 = 1
    a12 = 0
    a21 = 0
    a22 = (l + s)

    b1 = g * np.cos(phi) - ((mu * g) / P) * ds  + ((c * g) / P) * s - (l + s) * (dphi) ** 2
    b2 = -ds * dphi - g * (l + s) * np.sin(phi)

    detA = a11 * a22 - a12 * a21
    detA1 = b1 * a22 - a12 * b2
    detA2 = a11 * b2 - b1 * a21

    dy[2] = detA1 / detA
    dy[3] = detA2 / detA

    return dy

def spring(start, end, nodes, width):
    nodes = max(int(nodes), 1)

    start, end = np.array(start).reshape((2,)), np.array(end).reshape((2,))

    if (start == end).all():
        return start[0], start[1]

    length = np.linalg.norm(np.subtract(end, start))

    u_t = np.subtract(end, start) / length
    u_n = np.array([[0, -1], [1, 0]]).dot(u_t)

    spring_coords = np.zeros((2, nodes + 2))
    spring_coords[:,0], spring_coords[:,-1] = start, end

    normal_dist = np.sqrt(max(0, width**2 - (length**2 / nodes**2))) / 2

    for i in range(1, nodes + 1):
        spring_coords[:,i] = (
            start
            + ((length * (2 * i - 1) * u_t) / (2 * nodes))
            + (normal_dist * (-1)**i * u_n))

    return spring_coords[0,:], spring_coords[1,:]

def anim(i):
    pM.set_data([M_x[i], M_y[i]])
    OE.set_data([O_x, E_x[i]], [O_y, E_y[i]])
    Spr.set_data(*spring((O_x, O_y), (M_x[i], M_y[i]), 10, 0.3))
    return

steps = 1000
t = np.linspace(-np.pi, 0, steps)

y0 = [0, np.pi / 10, 0, 0.3]

P = 10
l = 0.5
c = 20
g = 9.8
mu = 0

Y = odeint(SystDiffEq, y0, t, (P, l, c, mu, g))

s = Y[:, 0]
phi = Y[:, 1]
ds = Y[:, 2]
dphi = Y[:, 3]

Stt = np.zeros_like(t)
Phitt = np.zeros_like(t)
for i in range(len(t)):
    Stt[i] = SystDiffEq(Y[i], t[i], P, l, c, mu, g)[2]
    Phitt[i] = SystDiffEq(Y[i], t[i], P, l, c, mu, g)[3]

diff_solve = plt.figure()
s_plt = diff_solve.add_subplot(2, 1, 1)
s_plt.plot(t, s)
phi_plt = diff_solve.add_subplot(2, 1, 2)
phi_plt.plot(t, phi)

x0 = 2
y0 = 3
L = 1.5

O_x = x0
O_y = y0

E_x = x0 + L * np.cos(t)
E_y = y0 + L * np.sin(t)

L_sping_max = l 
L_sping_curr = L_sping_max * np.sin(t)
M_x = x0 - L_sping_curr * np.cos(t)
M_y = y0 - L_sping_curr * np.sin(t)

fgr = plt.figure()
gr = fgr.add_subplot(1, 1, 1)
gr.axis('equal')
gr.set(xlim=[0, 5], ylim=[-1, 5])

gr.plot([0, 0, 4], [4, 0, 0], linewidth=3)
gr.plot([1.75, 2.25], [3, 3], 'black', linewidth=3)

pO = gr.plot(O_x, O_y)[0]
pE = gr.plot(E_x[0], E_y[0])[0]
pM = gr.plot(M_x[0], M_y[0], 'yellow', marker='o')[0]
OE = gr.plot([O_x, E_x[0]], [O_y, E_y[0]], 'grey')[0]
Spr = gr.plot(*spring((O_x, O_y), (M_x[0], M_y[0]), 10, 0.3), 'red')[0]

an = FuncAnimation(fgr, anim, frames=steps, interval=1)

plt.show()