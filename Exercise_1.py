import numpy as np
import matplotlib.pyplot as plt

COLS, ROWS = np.meshgrid(np.arange(8), np.arange(8))

u0 = 2.0
v0 = 2.0

I1 = 0.5 * np.exp(1j * 2 * np.pi / 8 * (u0 * COLS + v0 * ROWS))

I1R = np.real(I1)
I1I = np.imag(I1)

# Display real and imaginary parts as images
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(I1R, cmap='gray')
plt.axis('off')
plt.title('Re[I1]')

plt.subplot(1, 2, 2)
plt.imshow(I1I, cmap='gray')
plt.axis('off')
plt.title('Im[I1]')

Itilde1 = np.fft.fftshift(np.fft.fft2(I1))

print("Re[DFT(I1)]:")
print(np.round(np.real(Itilde1) * 10 ** 4) * 10 ** (-4))

print("Im[DFT(I1)]:")
print(np.round(np.imag(Itilde1) * 10 ** 4) * 10 ** (-4))

plt.show()
