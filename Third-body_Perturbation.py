import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


# PARAMETERS
mu = 1.0              
mu3 = 0.0123          
moon_radius = 3.5      
moon_omega  = 0.35     

# Satellite initial conditions 
r0 = np.array([1.0, 0.0])
v0 = np.array([0.0, 1.0])
y = np.hstack([r0, v0])
dt = 0.001
T  = 40
N  = int(T/dt)

# Data storage
xs = np.zeros(N)
ys = np.zeros(N)
moon_x = np.zeros(N)
moon_y = np.zeros(N)
Es = np.zeros(N)
Ls = np.zeros(N)      # <--- store angular momentum


# TIME-DEPENDENT MOON ORBIT FUNCTION
def moon_position(t):
    x = moon_radius * np.cos(moon_omega * t)
    y = moon_radius * np.sin(moon_omega * t)
    return np.array([x, y])

# ACCELERATION MODEL
def acceleration(r, v, t):
    acc_earth = -(mu * r) / (np.linalg.norm(r)**3)
    R3 = moon_position(t)
    dr3 = R3 - r
    acc_3body = mu3 * (dr3/np.linalg.norm(dr3)**3 - R3/np.linalg.norm(R3)**3)
    return acc_earth + acc_3body

# RUNGE-KUTTA 4
def rk4_step(y, t):
    r = y[:2]
    v = y[2:]

    def f(state, time):
        r = state[:2]
        v = state[2:]
        a = acceleration(r, v, time)
        return np.hstack([v, a])

    k1 = f(y, t)
    k2 = f(y + 0.5*dt*k1, t + 0.5*dt)
    k3 = f(y + 0.5*dt*k2, t + 0.5*dt)
    k4 = f(y + dt*k3, t + dt)

    return y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

# SIMULATION LOOP
t = 0
for i in range(N):
    r = y[:2]
    v = y[2:]

    xs[i] = r[0]
    ys[i] = r[1]

    # Moon path
    R3 = moon_position(t)
    moon_x[i] = R3[0]
    moon_y[i] = R3[1]

    # Mechanical energy
    E = 0.5*np.dot(v, v) - mu/np.linalg.norm(r)
    E += -mu3 * (1/np.linalg.norm(r-R3) - 1/np.linalg.norm(R3))
    Es[i] = E

    # Angular momentum Lz = x*v_y - y*v_x
    Ls[i] = r[0]*v[1] - r[1]*v[0]

    y = rk4_step(y, t)
    t += dt


# HELPER FUNCTION TO ADD IMAGES
def add_image(ax, img_path, xy, zoom=0.15):
    img = plt.imread(img_path)
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, xy, frameon=False)
    ax.add_artist(ab)

# ORBIT PLOT WITH IMAGES
fig, ax = plt.subplots(figsize=(7,7))
ax.plot(xs, ys, label="Satellite orbit", lw=1)
ax.plot(moon_x, moon_y, 'k--', lw=0.8, label="Moon path")
add_image(ax, "Earth.png", (0, 0), zoom=0.06)
add_image(ax, "Sat.png", (xs[-1], ys[-1]), zoom=0.04)
add_image(ax, "Moon.png", (moon_x[-1], moon_y[-1]), zoom=0.04)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Satellite Orbit with Third-Body Perturbation")
ax.set_aspect("equal")
ax.grid(True)
ax.legend()
plt.show()


t_vals = np.linspace(0, T, N)
dE = Es - Es[0]     # ΔE(t)
dL = Ls - Ls[0]     # ΔL(t)

fig, axs = plt.subplots(2, 1, figsize=(7,7), sharex=True)

# --- Energy subplot ---
axs[0].plot(t_vals, dE, lw=1.5)
axs[0].set_ylabel(r"$\Delta E(t) = E(t) - E(0)$")
axs[0].set_title("Change in Energy")
axs[0].grid(True)

# --- Angular momentum subplot ---
axs[1].plot(t_vals, dL, lw=1.5)
axs[1].set_xlabel("t")
axs[1].set_ylabel(r"$\Delta L_z(t) = L_z(t) - L_z(0)$")
axs[1].set_title("Change in Angular Momentum")
axs[1].grid(True)

plt.tight_layout()
plt.show()




