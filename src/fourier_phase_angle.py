import cv2
import numpy as np 
from scipy import ndimage

image = cv2.imread('../images/dip/DIP3E_CH04_Original_Images/DIP3E_Original_Images_CH04/Fig0424(a)(rectangle).tif', 0).astype(np.float)
F = np.fft.fft2(image)
F_shifted = np.fft.fftshift(F)
PA = np.arctan2(np.real(F_shifted), np.imag(F_shifted))

img_shifted = ndimage.interpolation.shift(image, (-200.,200.))
F_2 = np.fft.fft2(img_shifted)
F_2 = np.fft.fftshift(F_2)
PA_2 = np.arctan2(np.real(F_2), np.imag(F_2))

img_rot = ndimage.rotate(image, -45, reshape=False).astype(np.float)
F_3 = np.fft.fft2(img_rot)
F_3 = np.fft.fftshift(F_3)
PA_3 = np.arctan2(np.real(F_3), np.imag(F_3))

cv2.imshow('11', PA)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('11', f_2)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('11', PA_3)
cv2.waitKey(0)
cv2.destroyAllWindows()

f = np.fft.ifft2(F)
print(f)

print(f.dtype)
cv2.imshow('11', np.abs(f))
cv2.waitKey(0)
cv2.destroyAllWindows()
