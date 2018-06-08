
import cv2
import numpy as np
from matplotlib import pyplot as plt

# plots an image using matplotlib
def plot_img(plotimg, color=True):
    plt.figure(num=None, figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')
    if color == True:
        img = cv2.cvtColor(plotimg, cv2.COLOR_BGR2RGB)
        plt.imshow(img, interpolation = 'bicubic')
    else:
        plt.imshow(plotimg, interpolation = 'bicubic', cmap='gray')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


test = cv2.imread("latte.jpg")
plot_img(test)

test2 = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
plot_img(test2, False)
