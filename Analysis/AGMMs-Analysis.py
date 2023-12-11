import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate Gaussian distribution
def calculate_gaussian(data):
    mean = np.mean(data)
    variance = np.var(data)
    x = np.linspace(0, 255, 256)
    gaussian = (1 / (np.sqrt(2 * np.pi * variance))) * np.exp(-((x - mean) ** 2) / (2 * variance))
    return x, gaussian

image_directory = r'Media\WavingTrees'

# List all files in the directory
import os
image_files = sorted([f for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])

if len(image_files) < 2:
    print("There are not enough images in the directory.")
    exit()

# Initialize variables for histograms
hist_first = np.zeros(256, dtype=np.uint32)
hist_last = np.zeros(256, dtype=np.uint32)
hist_500th = np.zeros(256, dtype=np.uint32)

# Load the first, last, and 130th images
first_image = cv2.imread(os.path.join(image_directory, image_files[0]), cv2.IMREAD_GRAYSCALE)
last_image = cv2.imread(os.path.join(image_directory, image_files[286]), cv2.IMREAD_GRAYSCALE)
image_500 = cv2.imread(os.path.join(image_directory, image_files[130]), cv2.IMREAD_GRAYSCALE)

# Perform background subtraction using GMM for the first, last, and 500th images
fgbg = cv2.createBackgroundSubtractorMOG2()
first_fgmask = fgbg.apply(first_image)
last_fgmask = fgbg.apply(last_image)
image_500_fgmask = fgbg.apply(image_500)

# Apply the foreground masks to the images
first_image_altered = cv2.bitwise_and(first_image, first_image, mask=first_fgmask)
last_image_altered = cv2.bitwise_and(last_image, last_image, mask=last_fgmask)
image_500_altered = cv2.bitwise_and(image_500, image_500, mask=image_500_fgmask)

# Calculate histograms for the original images
hist_first = cv2.calcHist([first_image], [0], None, [256], [0, 256])
hist_last = cv2.calcHist([last_image], [0], None, [256], [0, 256])
hist_500th = cv2.calcHist([image_500], [0], None, [256], [0, 256])

# Create subplots for histograms and images
plt.figure(figsize=(12, 9))

# Subplot for the first image histogram
plt.subplot(3, 2, 1)
plt.title('Histogram (First Image)')
plt.plot(hist_first, color='green')

# Subplot for the first image after background subtraction
plt.subplot(3, 2, 2)
plt.title('After BGS (First Image)')
plt.imshow(first_image_altered, cmap='gray')

# Subplot for the 130th image histogram
plt.subplot(3, 2, 5)
plt.title('Histogram (500th Image)')
plt.plot(hist_500th, color='red')

# Subplot for the 130th image after background subtraction
plt.subplot(3, 2, 6)
plt.title('After BGS (500th Image)')
plt.imshow(image_500_altered, cmap='gray')

# Subplot for the last image histogram
plt.subplot(3, 2, 3)
plt.title('Histogram (Last Image)')
plt.plot(hist_last, color='blue')

# Subplot for the last image after background subtraction
plt.subplot(3, 2, 4)
plt.title('After BGS (Last Image)')
plt.imshow(last_image_altered, cmap='gray')



plt.tight_layout()
plt.show()
