import numpy as np
from .utils import fft2c, ifft2c, circ_aperture

def make_sample(N=256, amp_levels=(0.4, 0.7, 1.0), phase_scale=1.2):
    y, x = np.indices((N, N)) - N // 2
    r = np.sqrt(x**2 + y**2)
    th = np.arctan2(y, x)

    amp = np.ones((N, N), np.float32) * amp_levels[1]
    amp[r < 0.25 * N] = amp_levels[2]
    ann = (r > 0.35 * N) & (r < 0.45 * N)
    amp[ann] = amp_levels[0]
    amp *= (0.85 + 0.15 * np.cos(6 * th))

    phase = phase_scale * (np.sin(2 * np.pi * x / N * 3) + np.cos(2 * np.pi * y / N * 2))
    return amp * np.exp(1j * phase), amp, phase

def make_pupil(N, na_obj=0.18, na_max=0.5):
    r = int((na_obj / na_max) * (N / 2))
    return circ_aperture(N, r)

def k_shifts(N, na_led=0.5, na_step=0.08, grid=3):
    offs = (np.arange(grid) - (grid // 2)) * na_step
    pairs = [(kx, ky) for kx in offs for ky in offs]
    scale = (N / 2) / na_led
    return [(int(kx * scale), int(ky * scale)) for kx, ky in pairs]

def shift_spec(S, kx, ky):
    return np.roll(np.roll(S, ky, 0), kx, 1)

def simulate_measurements(obj, pupil, shifts):
    O = fft2c(obj)
    ims = []
    for kx, ky in shifts:
        sub = shift_spec(O, -kx, -ky) * pupil
        field = ifft2c(sub)
        ims.append(np.abs(field)**2)
    low = np.abs(ifft2c(O * pupil))**2
    return ims, low
