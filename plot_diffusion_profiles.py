# filename: plot_diffusion_profiles.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# ──────────────── USER‐DEFINED PARAMETERS ────────────────
a = 1.0           # half‐width scale
D = 1.0           # diffusion coefficient
# dimensionless times τ = D·t/a²
tau_list = [0.01, 0.1, 1.0, 10.0]  
# spatial range in units of a
z_over_a = np.linspace(-5, 5, 1000)
# ───────────────────────────────────────────────────────────

def c_over_c0(eta, tau):
    """
    Dimensionless concentration c/c0 as a function
    of eta = z/a and tau = D t / a^2.
    """
    denom = 2.0 * np.sqrt(tau)
    arg1 = (eta - 1.0) / denom
    arg2 = (eta - 2.0) / denom
    arg3 = (eta + 2.0) / denom
    arg4 = (eta + 1.0) / denom
    return 0.5 * (erf(arg1) - erf(arg2) + erf(arg3) - erf(arg4))

# ──────────────── PLOTTING ────────────────
plt.figure(figsize=(8,6))
for tau in tau_list:
    plt.plot(z_over_a,
             c_over_c0(z_over_a, tau),
             lw=2,
             label=rf'$\tau = {tau}$')

plt.xlabel(r'$z / a$')
plt.ylabel(r'$c / c_{0}$')
plt.title('Diffusion profile for multiple dimensionless times\n' +
          r'$\tau = D\,t/a^{2}$')
plt.legend(title='Dimensionless time')
plt.grid(True)
plt.tight_layout()
plt.show()
