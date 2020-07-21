import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
# image = cv2.imread('../images/dip/DIP3E_CH04_Original_Images/DIP3E_Original_Images_CH04/Fig0424(a)(rectangle).tif', 0)
image = np.zeros((64,64), dtype=np.float32)
image[25:40,32:34] = 255

# DFT
def discrete_fourier_transform(img, shift=False, phase_angle=False):
    
    M, N = img.shape
    F = np.zeros((M, N), dtype=np.complex)
    
    x, y = np.arange(0, N, 1), np.arange(0, M, 1)
    x, y = np.meshgrid(x,y)
    
    if shift:
        img = img * (-1)**(x+y)
    
    for v in range(N-1):
        for u in range(M-1):           
            F[u, v] = np.sum(img * np.exp(- 1j * 2*np.pi * (u * x / M + v * y / N)))
    
    if phase_angle:
        PA = np.arctan2(np.real(F), np.imag(F))
        return PA
    return F

def inverse_discrete_fourier_transform(F):
    
    M, N = F.shape
    f = np.zeros((M, N), dtype=np.float)
    
    u, v = np.arange(0, N, 1), np.arange(0, M, 1)
    u, v = np.meshgrid(u, v)
    
    
    for y in range(N-1):
        for x in range(M-1):           
            f[x, y] = np.sum(F * np.exp(1j * 2*np.pi * (u * x / M + v * y / N)))
        
#     f = np.clip(f, 0, 255)#.astype(np.uint8)
    
    return f
def scale(img, K=255):

    img = img - img.min()
    img = K * (img / img.max())
    
    return img#.astype(np.uint8)

F1 = discrete_fourier_transform(image, shift=True)
# ps = (np.abs(F) / np.abs(F).max() * 255).astype(np.uint8)
F = np.fft.fft2(image)
F = np.fft.fftshift(F)
F = np.log(np.abs(F))
# print(F)
# F_shifted = discrete_fourier_transform(image, shift=True)
r_img = inverse_discrete_fourier_transform(F1)
# fig, ax = plt.subplots(1,2, figsize=(15,15))
plt.subplot(131)
plt.imshow(F, 'gray')
plt.subplot(132)
plt.imshow(F, 'gray')
plt.subplot(133)
print(r_img)
plt.imshow(r_img, 'gray')
plt.show()

F = np.clip(F, 0, 255)
F = scale(F)
print(F.min(), F.max())
cv2.imshow("result", image)
cv2.waitKey(0)

cv2.imshow("result", F)
cv2.waitKey(0)

# Save result
cv2.imshow("result", r_img)
cv2.waitKey(0)
