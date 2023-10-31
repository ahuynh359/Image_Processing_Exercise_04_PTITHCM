import numpy as np
import matplotlib.pyplot as plt

COLS, ROWS = np.meshgrid(np.arange(8), np.arange(8))
u0 = 2
v0 = 2
I5 = np.cos(2 * np.pi / 8 * (u0 * COLS + v0 * ROWS))

I5R = np.real(I5)
I5I = np.imag(I5)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(I5R, cmap='gray')
plt.axis('off')
plt.title('Re[I5]')

plt.subplot(1, 2, 2)
plt.imshow(I5I, cmap='gray')
plt.axis('off')
plt.title('Im[I5]')

Itilde5 = np.fft.fftshift(np.fft.fft2(I5))
print("Re[DFT(I5)]:")
print(np.round(np.real(Itilde5) * 10 ** 4) * 10 ** (-4))

print("Im[DFT(I5)]:")
print(np.round(np.imag(Itilde5) * 10 ** 4) * 10 ** (-4))

plt.show()
