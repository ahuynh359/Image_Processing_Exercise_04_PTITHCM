import numpy as np
import matplotlib.pyplot as plt

COLS, ROWS = np.meshgrid(np.arange(8), np.arange(8))

u0 = 2
v0 = 2
I2 = 0.5 * np.exp(-1j * 2 * np.pi / 8 * (u0 * COLS + v0 * ROWS))

I2R = np.real(I2)
I2I = np.imag(I2)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(I2R, cmap='gray')
plt.axis('off')
plt.title('Re[I2]')

plt.subplot(1, 2, 2)
plt.imshow(I2I, cmap='gray')
plt.axis('off')
plt.title('Im[I2]')

Itilde2 = np.fft.fftshift(np.fft.fft2(I2))

print("Re[DFT(I2)]:")
print(np.round(np.real(Itilde2) * 10 ** 4) * 10 ** (-4))

print("Im[DFT(I2)]:")
print(np.round(np.imag(Itilde2) * 10 ** 4) * 10 ** (-4))

plt.show()
