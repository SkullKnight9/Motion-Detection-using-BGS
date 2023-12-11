import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread(r'Media\WavingTrees\b00000.bmp')  

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate the histogram of the grayscale image
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# Normalize the histogram
hist /= hist.sum()

# Calculate mean and standard deviation for the Gaussian distribution
mean = np.mean(gray)
std_dev = np.std(gray)

# Create a range of values for the x-axis
x = np.arange(0, 256)

# Calculate the Gaussian distribution using the mean and standard deviation
gaussian = 1 / (std_dev * np.sqrt(2 * np.pi)) * np.exp(-(x - mean) ** 2 / (2 * std_dev ** 2))

# Plot the histogram and Gaussian distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(hist, color='black')
plt.title('Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.xlim([0, 256])

plt.subplot(1, 2, 2)
plt.plot(x, gaussian, color='red')
plt.title('Gaussian Distribution')
plt.xlabel('Pixel Value')
plt.ylabel('Probability Density')

plt.show()
