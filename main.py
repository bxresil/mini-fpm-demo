import numpy as np
import matplotlib.pyplot as plt
from src.fpm_sim import make_sample, make_pupil, k_shifts, simulate_measurements
from src.fpm_recon import recon_epie

np.random.seed(0)
N = 256

obj, amp, phase = make_sample(N)
pupil = make_pupil(N, na_obj=0.18, na_max=0.5)
shifts = k_shifts(N, na_led=0.5, na_step=0.08, grid=3)
ints, low = simulate_measurements(obj, pupil, shifts)

plt.figure(); plt.imshow(amp, cmap='gray'); plt.axis('off'); plt.title('Sample amplitude'); plt.savefig('figs/sample_amp.png', dpi=150)
plt.figure(); plt.imshow(phase, cmap='twilight'); plt.axis('off'); plt.title('Sample phase'); plt.colorbar(); plt.savefig('figs/sample_phase.png', dpi=150)
plt.figure(); plt.imshow(low, cmap='gray'); plt.axis('off'); plt.title('Low NA image'); plt.savefig('figs/low_NA_image.png', dpi=150)

x0 = np.sqrt(low) * np.exp(1j * 0.0)
recon, hist = recon_epie(ints, pupil, shifts, iters=80, beta=0.6, obj_init=x0)

plt.figure(); plt.plot(hist); plt.xlabel('iter'); plt.ylabel('MSE proxy'); plt.grid(True); plt.title('Convergence'); plt.savefig('figs/mse_curve.png', dpi=150)
plt.figure(); plt.imshow(np.abs(recon), cmap='gray'); plt.axis('off'); plt.title('Reconstruction amplitude'); plt.savefig('figs/recon_amp.png', dpi=150)
plt.figure(); plt.imshow(np.angle(recon), cmap='twilight'); plt.axis('off'); plt.title('Reconstruction phase'); plt.colorbar(); plt.savefig('figs/recon_phase.png', dpi=150)

print("Done. See figs/ for outputs.")
