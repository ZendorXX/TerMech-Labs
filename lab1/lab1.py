import math
import sympy as s
import matplotlib.pyplot as plot
import numpy as np
from matplotlib.animation import FuncAnimation

def rotation2D(x, y, angle):
    Rot_x = x * np.cos(angle) - y * np.sin(angle)
    Rot_y = x * np.sin(angle) + y * np.cos(angle)
    return Rot_x, Rot_y

def Vect_arrow(VecX, VecY, X, Y):
    a = 0.3
    b = 0.2
    arrow_x = np.array([-a, 0, -a])
    arrow_y = np.array([b, 0, -b])

    phi = math.atan2(VecY, VecX)

    RotX, RotY = rotation2D(arrow_x, arrow_y, phi)

    arrow_x = RotX + X + VecX
    arrow_y = RotY + Y + VecY

    return arrow_x, arrow_y

def anim(i):
    Pnt.set_data(X[i], Y[i])

    Radius_vector.set_data([0, X[i]], [0, Y[i]])
    Radius_vector_arrow.set_data(Vect_arrow(X[i], Y[i], 0, 0))

    Velocity_vector.set_data([X[i], X[i] + X_velocity[i]], [Y[i], Y[i]+Y_velocity[i]])
    Velocity_arrow.set_data(Vect_arrow(X_velocity[i], Y_velocity[i], X[i], Y[i]))

    Acceleration_vector.set_data([X[i], X[i] + X_acceleration[i]], [Y[i], Y[i] + Y_acceleration[i]])
    Acceleration_arrow.set_data(Vect_arrow(X_acceleration[i], Y_acceleration[i], X[i], Y[i]))
    
    return

t = s.Symbol('t')

r = 2 + s.cos(6 * t)
phi = t + 1.2 * s.cos(6 * t)

x = r * s.cos(phi)
y = r * s.sin(phi)

x_velocity = s.diff(x)
y_velocity = s.diff(y)

x_acceleration = s.diff(x_velocity)
y_acceleration = s.diff(y_velocity)

step = 1000

T = np.linspace(0, 10, step)

X = np.zeros_like(T)
Y = np.zeros_like(T)

X_velocity = np.zeros_like(T)
Y_velocity = np.zeros_like(T)

X_acceleration = np.zeros_like(T)
Y_acceleration = np.zeros_like(T)

for i in np.arange(len(T)):
    X[i] = s.Subs(x, t, T[i])
    Y[i] = s.Subs(y, t, T[i])

    X_velocity[i] = s.Subs(x_velocity, t, T[i])
    Y_velocity[i] = s.Subs(y_velocity, t, T[i])

    X_acceleration[i] = s.Subs(x_acceleration, t, T[i])
    Y_acceleration[i] = s.Subs(y_acceleration, t, T[i])

fgr = plot.figure()

grf = fgr.add_subplot(1, 1, 1)
grf.axis('equal')
grf.set(xlim=[-10, 10], ylim=[-10, 10])
grf.plot(X, Y)

Pnt = grf.plot(X[0], Y[0], marker='o')[0]

Radius_vector = grf.plot([0, X[0]], [0, Y[0]], 'black')[0]
X_radius_vector_arrow, Y_radius_vector_arrow = Vect_arrow(X[0], Y[0], 0, 0)
Radius_vector_arrow = grf.plot(X_radius_vector_arrow, Y_radius_vector_arrow, 'black')[0]

Velocity_vector = grf.plot([X[0], X[0] + X_velocity[0]], [Y[0], Y[0] + Y_velocity[0]], 'r')[0]
X_velocity_arrow, Y_velocity_arrow = Vect_arrow(X_velocity[0], Y_velocity[0], X[0], Y[0])
Velocity_arrow = grf.plot(X_velocity_arrow, Y_velocity_arrow, 'r')[0]

Acceleration_vector = grf.plot([X[0], X[0] + X_acceleration[0]], [Y[0], Y[0] + Y_acceleration[0]], 'g')[0]
X_acceleration_arrow, Y_acceleration_arrow = Vect_arrow(X_acceleration[0], Y_acceleration[0], X[0], Y[0])
Acceleration_arrow = grf.plot(X_acceleration_arrow, Y_acceleration_arrow, 'g')[0]

an = FuncAnimation(fgr, anim, frames=step, interval=10)

fgr.show()
