import torch
from torchvision import datasets
import cv2

data_path = '../dlwpt-code/data/p1ch2/data-unversioned/p1ch7'
image_path = data_path + '/horse.jpg'
img = imageio.read(image_path)
print(img)
