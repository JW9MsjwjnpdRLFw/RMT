import cv2
import numpy as np 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

img = mpimg.imread('D:\\PSPNet-Keras-tensorflow-master\\output\\1479425441182877835_seg_read.jpg')
# img = mpimg.imread('D:\\pix2pixHD-master\\datasets\\cityscapes\\train_label\\aachen_000000_000019_gtFine_labelIds.png')
# img2 = mpimg.imread('D:\\pix2pixHD-master\\datasets\\cityscapes\\frankfurt_000000_000294_gtFine_labelIds_ori.png')
img = img / 255.
plt.imshow(img)
plt.show()

# mpimg.imsave('123',img[:,:,0])
# img = mpimg.imread('123')
# plt.imshow(img)
# plt.show()