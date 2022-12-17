import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

omega_f = 5 # generator angular frequency
A = 5 # generator amplitude
gamma = 0.00001 # vibration damping coefficient
omega_2 = 1 # omega_2 = k / M

M = np.eye(12) * (-gamma)

for i in range(1, 12, 2):
    M[i][i - 1] = 1

for i in range(2, 10, 2):
    M[i][i - 1] = omega_2
    M[i][i + 1] = -2 * omega_2
    M[i][i + 3] = omega_2

M[0][1] = -2 * omega_2
M[0][3] = omega_2
M[10][9] = omega_2
M[10][11] = -2 * omega_2


def deriv_u(u, t):
    return M @ u + np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, A * np.sin(omega_f * t), 0])


# array of initial conditions (even - u_i, odd - u_i derivative)
u0 = np.zeros(12)

t = np.linspace(0, 80)
u = odeint(deriv_u, u0, t)


u_ = [[u[i][k] for i in range(len(u))] for k in range(1, 12, 2)]

# here the plot for first body
plt.plot(t, u_[1])
plt.xlabel('Time')
plt.ylabel('U(t)')
plt.show()