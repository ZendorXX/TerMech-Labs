import numpy as np    
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

steps = 1000

fgr = plt.figure()
gr = fgr.add_subplot(1, 1, 1)
gr.axis('equal')

gr.plot([0, 0, 4], [4, 0, 0], linewidth=3)
gr.plot([0.4, 3.6], [3, 3], 'black', linewidth=3)

x0 = 2
y0 = 3
L = 1.5

O_x = x0
O_y = y0

pO = gr.plot(O_x, O_y)[0]

#a = np.linspace(0, 10, steps)
#t = np.sin(a)

t = np.linspace(-np.pi, 0, steps)

E_x = x0 + L * np.cos(t)
E_y = y0 + L * np.sin(t)

pE = gr.plot(E_x[0], E_y[0])[0]

k = 1
l = 0.5
t_pr = np.linspace(0, l, steps)
L_pruzh = np.cos(np.sqrt(k) * t_pr)

Ring_x = x0 + (L - L_pruzh) * np.cos(t)
Ring_y = y0 + (L - L_pruzh) * np.sin(t)

pRing = gr.plot(Ring_x[0], Ring_y[0], 'red', marker='o')[0]

OE = gr.plot([O_x, E_x[0]], [O_y, E_y[0]], 'black')[0]

def anim(i):
    #pE.set_data(E_x[i], E_y[i])
    pRing.set_data(Ring_x[i], Ring_y[i])
    OE.set_data([O_x, E_x[i]], [O_y, E_y[i]])
    return

an = FuncAnimation(fgr, anim, frames=steps, interval=1)

plt.show()
