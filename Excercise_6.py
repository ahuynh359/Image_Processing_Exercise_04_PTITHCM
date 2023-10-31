import numpy as np
import matplotlib.pyplot as plt


# Function to read binary image data
def read_bin(filename, shape):
    return np.fromfile(filename, dtype=np.uint8).reshape(shape)


# Function to compute the DFT, stretch, and create visualizations
def compute_dft_and_visualize(image, title, index, total_images):
    Xtilde = np.fft.fftshift(np.fft.fft2(image))
    XtildeR = np.real(Xtilde)
    XtildeI = np.imag(Xtilde)
    XtildeMag = np.log(1 + np.abs(Xtilde))
    XtildePhase = np.angle(Xtilde)

    # Create subplots for the images and their DFT-related visualizations
    plt.subplot(total_images, 5, (index - 1) * 5 + 1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title(title)

    plt.subplot(total_images, 5, (index - 1) * 5 + 2)
    plt.imshow(XtildeR, cmap='gray')
    plt.axis('off')
    plt.title(f'Re[DFT({title})]')

    plt.subplot(total_images, 5, (index - 1) * 5 + 3)
    plt.imshow(XtildeI, cmap='gray')
    plt.axis('off')
    plt.title(f'Im[DFT({title})]')

    plt.subplot(total_images, 5, (index - 1) * 5 + 4)
    plt.imshow(XtildeMag, cmap='gray', vmin=0, vmax=np.max(XtildeMag))
    plt.axis('off')
    plt.title(f'{title} log magnitude spectrum')

    plt.subplot(total_images, 5, (index - 1) * 5 + 5)
    plt.imshow(XtildePhase, cmap='gray', vmin=np.min(XtildePhase), vmax=np.max(XtildePhase))
    plt.axis('off')
    plt.title(f'arg[DFT({title})]')


image_filenames = ['camerabin.sec', 'salesmanbin.sec', 'headbin.sec', 'eyeRbin.sec']
image_titles = ['camera', 'salesman', 'head', 'eyeR']

# Initialize a total number of images
total_images = len(image_filenames)

# Create a single figure with subplots for all images
plt.figure(figsize=(15, 4 * total_images))

# Load and process each image


for index, (filename, title) in enumerate(zip(image_filenames, image_titles), 1):
    image_data = read_bin(filename, (256, 256))
    compute_dft_and_visualize(image_data, title, index, total_images)

plt.tight_layout()
plt.show()
