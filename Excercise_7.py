import numpy as np
import matplotlib.pyplot as plt


# Function to read binary image data
def read_bin(filename, shape):
    return np.fromfile(filename, dtype=np.uint8).reshape(shape)


# Function to perform full-scale contrast stretch on an image
def stretch(x):
    x_max = np.max(x)
    x_min = np.min(x)
    scale_factor = 255.0 / (x_max - x_min)
    return np.round((x - x_min) * scale_factor).astype(np.uint8)


# Load 'camera.bin' image
x = read_bin('camerabin.sec', (256, 256))

# Calculate J1tilde
xtilde = np.fft.fft2(x)
j1tilde = np.abs(xtilde)

# Calculate J1 and J1prime
j1 = np.real(np.fft.ifft2(j1tilde))
j1prime = stretch(np.log(1 + j1))

# Display J1prime
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(j1prime, cmap='gray')
plt.axis('off')
plt.title('J1')

# Calculate J2tilde
j2tilde = np.exp(1j * np.angle(xtilde))

# Calculate J2
j2 = stretch(np.real(np.fft.ifft2(j2tilde)))

# Display J2
plt.subplot(1, 2, 2)
plt.imshow(j2, cmap='gray')
plt.axis('off')
plt.title('J2')

plt.tight_layout()
plt.show()
