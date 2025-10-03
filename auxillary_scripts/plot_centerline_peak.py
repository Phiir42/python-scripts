# filename: plot_centerline_peak.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# ─── USER PARAMETERS ────────────────────────────────────────
a = 1.0       # half-width (set your a [L] here)
D = 1.0       # diffusion coefficient (set your D [L²/T] here)

# initial wide time grid to locate the maximum
t_full = np.linspace(1e-4, 20.0, 20000)  # physical time [T]

# ─── DEFINE THE NORMALIZED CENTERLINE CONCENTRATION ────────
def c0_norm(t):
    """
    Returns c(0,t)/c0 = erf(2a/sqrt(4 D t)) - erf(a/sqrt(4 D t))
    """
    arg1 = 2*a / np.sqrt(4*D*t)
    arg2 =   a / np.sqrt(4*D*t)
    return erf(arg1) - erf(arg2)

# ─── FIND PEAK ──────────────────────────────────────────────
y_full = c0_norm(t_full)
idx_peak = np.argmax(y_full)
t_peak = t_full[idx_peak]
y_peak = y_full[idx_peak]

print(f"Peak occurs at t_peak = {t_peak:.4f} (in units of your time), "
      f"with c(0,t)/c0 = {y_peak:.4f}")

# ─── FOCUS REGION: from t≈0 to 5×t_peak ────────────────────
t_max = 5 * t_peak
mask = t_full <= t_max
t = t_full[mask]
y = y_full[mask]

# ─── PLOTTING ──────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(t, y, lw=2, label=r'$c(0,t)/c_0$')
plt.scatter([t_peak], [y_peak], s=50, marker='o',
            label=f'peak at t={t_peak:.3f}')

plt.xlabel(r'$t$')
plt.ylabel(r'$c(0,t)/c_{0}$')
plt.title('Centerline concentration vs. time\n'
          f'(showing 0 ≤ t ≤ {t_max:.2f})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()