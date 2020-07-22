import cv2
import numpy as np 
image = cv2.imread('../images/dip/DIP3E_CH04_Original_Images/DIP3E_Original_Images_CH04/Fig0424(a)(rectangle).tif', 0).astype(np.float)
image = cv2.resize(image, (512, 512))
F = np.fft.fft2(image)
S_ = np.abs(F)
PA_ = np.arctan2(np.imag(F), np.real(F))

image = cv2.imread('../images/dip/DIP3E_CH04_Original_Images/DIP3E_Original_Images_CH04/Fig0427(a)(woman).tif', 0).astype(np.float)

F = np.fft.fft2(image)
F = np.fft.fftshift(F)
PA = np.arctan2(np.imag(F), np.real(F))
img_res_1 = np.abs(np.fft.ifft2(1*np.exp(1j*PA)))

img_res_2 = np.abs(np.fft.ifft2(np.abs(F)))
img_res_2 = img_res_2**(0.6)
img_res_3 = np.abs(np.fft.ifft2(S_*np.exp(1j*PA)))
img_res_4 = np.abs(np.fft.ifft2(np.abs(F)*np.exp(1j*PA_)))
cv2.imshow('11', img_res_3)
cv2.waitKey(0)
cv2.destroyAllWindows()
