import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

mu = 1.0          
D_drag = 0.02     
r0 = np.array([1.0, 0.0])
v0 = np.array([0.0, 1.0])
y0 = np.concatenate((r0, v0))

t0 = 0.0
T = 40.0         
dt = 1e-3         



# ODE system with gravity + constant-magnitude drag
def rhs(t, y):
    x, y_pos, vx, vy = y

    r_vec = np.array([x, y_pos])
    v_vec = np.array([vx, vy])

    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    if r == 0.0:
        a_grav = np.zeros(2)
    else:
        a_grav = -mu * r_vec / r**3

    if v == 0.0:
        a_drag = np.zeros(2)
    else:
        v_hat = v_vec / v
        a_drag = -D_drag * v_hat

    a_total = a_grav + a_drag

    return np.array([vx, vy, a_total[0], a_total[1]])


# RK4 integrator
def rk4_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + 0.5*h, y + 0.5*h*k1)
    k3 = f(t + 0.5*h, y + 0.5*h*k2)
    k4 = f(t + h,     y + h*k3)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def integrate_orbit(f, y0, t0, T, dt):
    N = int(np.round((T - t0) / dt)) + 1
    t_vals = np.linspace(t0, T, N)
    y_vals = np.zeros((N, 4))
    y_vals[0] = y0

    for n in range(N - 1):
        y_vals[n+1] = rk4_step(f, t_vals[n], y_vals[n], dt)

    return t_vals, y_vals



# Energies and angular momentum 

def compute_energy_and_angular_momentum(y_vals):
    x = y_vals[:, 0]
    y = y_vals[:, 1]
    vx = y_vals[:, 2]
    vy = y_vals[:, 3]

    r = np.sqrt(x**2 + y**2)
    v2 = vx**2 + vy**2

    E = 0.5 * v2 - mu / r

    Lz = x * vy - y * vx

    return E, Lz



# Plotting utilities
def add_image(ax, img_path, xy, zoom=0.15):
    try:
        img = plt.imread(img_path)
    except FileNotFoundError:
        return None

    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, xy, frameon=False)
    ax.add_artist(ab)
    return ab


def plot_orbit_with_images(t_vals, y_vals):
    x = y_vals[:, 0]
    y = y_vals[:, 1]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, y, linestyle='-', linewidth=1.0, label='Orbit with drag')

    earth_added = add_image(ax, 'Earth.png', (0.0, 0.0), zoom=0.15)
    if earth_added is None:
        ax.scatter(0.0, 0.0, s=200, label='Earth')

    x_sat, y_sat = x[-1], y[-1]
    sat_added = add_image(ax, 'Sat.png', (x_sat, y_sat), zoom=0.05)
    if sat_added is None:
        ax.scatter(x_sat, y_sat, s=50, label='Satellite (final)')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal', 'box')
    ax.set_title('Orbital Trajectory with Constant Drag')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()


def plot_energy_and_ang_momentum(t_vals, E, Lz):
    E0 = E[0]
    L0 = Lz[0]
    dE = E - E0
    dL = Lz - L0

    fig, axs = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    axs[0].plot(t_vals, dE)
    axs[0].set_ylabel(r'$\Delta E(t) = E(t) - E(0)$')
    axs[0].set_title('Change in Energy')
    axs[0].grid(True)

    axs[1].plot(t_vals, dL)
    axs[1].set_xlabel('t')
    axs[1].set_ylabel(r'$\Delta L_z(t) = L_z(t) - L_z(0)$')
    axs[1].set_title('Change in Angular Momentum')
    axs[1].grid(True)

    plt.tight_layout()


# Main driver
def main():
    t_vals, y_vals = integrate_orbit(rhs, y0, t0, T, dt)

    E, Lz = compute_energy_and_angular_momentum(y_vals)

    plot_orbit_with_images(t_vals, y_vals)
    plot_energy_and_ang_momentum(t_vals, E, Lz)

    plt.show()


if __name__ == "__main__":
    main()
