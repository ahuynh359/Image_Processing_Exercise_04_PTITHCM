import numpy as np
import matplotlib.pyplot as plt

COLS, ROWS = np.meshgrid(np.arange(8), np.arange(8))
u0 = 2
v0 = 2
I4 = np.sin(2 * np.pi / 8 * (u0 * COLS + v0 * ROWS))

I4R = np.real(I4)
I4I = np.imag(I4)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(I4R, cmap='gray')
plt.axis('off')
plt.title('Re[I4]')

plt.subplot(1, 2, 2)
plt.imshow(I4I, cmap='gray')
plt.axis('off')
plt.title('Im[I4]')

Itilde4 = np.fft.fftshift(np.fft.fft2(I4))
print("Re[DFT(I4)]:")
print(np.round(np.real(Itilde4) * 10 ** 4) * 10 ** (-4))





print("Im[DFT(I4)]:")
print(np.round(np.imag(Itilde4) * 10 ** 4) * 10 ** (-4))

plt.show()
