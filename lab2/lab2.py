import numpy as np    
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

fgr = plt.figure()
gr = fgr.add_subplot(1, 1, 1)
gr.axis('equal')
gr.set(xlim=[0, 5], ylim=[-1, 5])

gr.plot([0, 0, 4], [4, 0, 0], linewidth=3)
gr.plot([1.75, 2.25], [3, 3], 'black', linewidth=3)

t = np.linspace(0, 10, steps)

phi_t = np.cos(t) - np.pi / 2
s_t = np.sin(t)

x0 = 2
y0 = 3
L = 1.5

O_x = x0
O_y = y0

E_x = x0 + L * np.cos(phi_t)
E_y = y0 + L * np.sin(phi_t)

L_sping_max = 1.2 
L_sping_curr = L_sping_max * np.sin(s_t)
M_x = x0 + L_sping_curr * np.cos(phi_t)
M_y = y0 + L_sping_curr * np.sin(phi_t)

pO = gr.plot(O_x, O_y)[0]
pE = gr.plot(E_x[0], E_y[0])[0]
pM = gr.plot(M_x[0], M_y[0], 'yellow', marker='o')[0]
OE = gr.plot([O_x, E_x[0]], [O_y, E_y[0]], 'grey')[0]
Spr = gr.plot(*spring((O_x, O_y), (M_x[0], M_y[0]), 10, 0.3), 'red')[0]

an = FuncAnimation(fgr, anim, frames=steps, interval=1)

plt.show()
