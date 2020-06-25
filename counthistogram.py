import matplotlib.pyplot as plt
import cv2

im = cv2.imread('image.jpg')
# calculate mean value from RGB channels and flatten to 1D array
vals = im.mean(axis=2).flatten()
