import numpy as n
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

step = 1000
t = n.linspace(0, 10, step)
x = n.sin(t)
phi = n.sin(2 * t)

fgr = plt.figure()
gr = fgr.add_subplot(1, 1, 1)
gr.axis('equal')

gr.plot([0, 0, 5], [2, 0, 0], linewidth=3)

W = 0.8
H = 0.4
r = 0.1
x0 = 1.5 
L = 0.5 # AB

Xa = x0 + W/2 + x
Ya = 2 * r + H/2

Xb = Xa + L * n.sin(phi)
Yb = Ya + L * n.cos(phi)

pA = gr.plot(Xa[0], Ya, marker='o')[0]
pB = gr.plot(Xb[0], Yb[0], marker='o')[0]

Telega = gr.plot(n.array([-W/2, W/2, W/2, -W/2, -W/2]) + Xa[0], n.array([-H/2, -H/2, H/2, H/2, -H/2]) + Ya)[0]
AB = gr.plot([Xa[0], Xb[0]], [Ya, Yb[0]])[0]

Alp = n.linspace(0, 2 * n.pi, 100)
Xc = r * n.cos(Alp)
Yc = r * n.sin(Alp) + r

Wheel1 = gr.plot(Xc + x0 + Xa[0] - W/4, Yc, 'black')[0]
Wheel2 = gr.plot(Xc + x0 + Xa[0] + W/4, Yc, 'black')[0]

# Шаблон пружины
# /\  /\
#   \/  \/
Np = 20
Xp = n.linspace(0, 1, 2 * Np + 1)
Yp = 0.06 * n.sin(n.pi / 2 * n.arange(2 * Np + 1))

Pruzh = gr.plot((x0 + x[0]) * Xp, Yp + 2 * r + H/2)[0]

# Шаблон спиральной пружины
Ns = 3
r1 = 0.1
r2 = 0.3
numpnts = n.linspace(0, 1, 5 * Ns + 1)
Betas = numpnts * (Ns * 2 * n.pi - phi[0])
Xs = (r1 + (r2 - r1) * numpnts) * n.cos(Betas + n.pi/2)
Ys = (r1 + (r2 - r1) * numpnts) * n.sin(Betas + n.pi/2)

SpPruzh = gr.plot(Xs + Xa[0], Ys + Ya)[0]

def run(i):
    pA.set_data(Xa[i], Ya)
    pB.set_data(Xb[i], Yb[i])
    Telega.set_data(n.array([-W/2, W/2, W/2, -W/2, -W/2]) + Xa[i], n.array([-H/2, -H/2, H/2, H/2, -H/2]) + Ya)
    AB.set_data([Xa[i], Xb[i]], [Ya, Yb[i]])
    Wheel1.set_data(Xc + Xa[i] - W/4, Yc)
    Wheel2.set_data(Xc + Xa[i] + W/4, Yc)
    Pruzh.set_data((x0 + x[i]) * Xp, Yp + 2 * r + H/2)
    
    Betas = numpnts * (Ns * 2 * n.pi - phi[i])
    Xs = (r1 + (r2 - r1) * numpnts) * n.cos(Betas + n.pi/2)
    Ys = (r1 + (r2 - r1) * numpnts) * n.sin(Betas + n.pi/2)

    SpPruzh.set_data(Xs + Xa[i], Ys + Ya)[0]

    return
 
anim = FuncAnimation(fgr, run, frames=step, interval=1) 

fgr.show()



