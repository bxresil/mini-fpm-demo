import numpy as np
from .utils import fft2c, ifft2c

def shift_spec(S, kx, ky):
    return np.roll(np.roll(S, ky, 0), kx, 1)

def recon_epie(ints, pupil, shifts, iters=80, beta=0.6, obj_init=None):
    N = ints[0].shape[0]
    obj = np.ones((N, N), np.complex64) if obj_init is None else obj_init.astype(np.complex64)
    O = fft2c(obj)
    hist = []
    for _ in range(iters):
        err = 0.0
        for I, (kx, ky) in zip(ints, shifts):
            sub = shift_spec(O, -kx, -ky) * pupil
            u = ifft2c(sub)
            amp = np.sqrt(np.maximum(I, 0))
            u_new = amp * np.exp(1j * np.angle(u + 1e-12))
            sub_new = fft2c(u_new)
            resid = (sub_new - sub) * pupil
            sub_upd = sub + beta * resid
            O = shift_spec(sub_upd, kx, ky)
            err += np.mean((np.abs(u)**2 - I)**2)
        hist.append(err / len(ints))
    x = ifft2c(O)
    return x, np.array(hist)
