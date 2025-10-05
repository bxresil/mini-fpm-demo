import numpy as np

def fft2c(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

def ifft2c(X):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(X)))

def mse(a, b):
    return float(np.mean((np.abs(a) - np.abs(b))**2))

def circ_aperture(N, r):
    yy, xx = np.indices((N, N)) - N // 2
    rr = np.sqrt(xx**2 + yy**2)
    return (rr <= r).astype(np.float32)
