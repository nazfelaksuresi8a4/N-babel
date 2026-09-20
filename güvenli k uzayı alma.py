import nibabel as nib
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np 

nii_file = 'BraTS20_Training_002_t1.nii'

loaded = nib.load(nii_file)
matlike = loaded.get_fdata()

h,w,c = matlike.shape

matlike = matlike[:,:,c//2]

x_start,y_start = (w // 2) - 50 ,(h // 2) - 50 
x_stop,y_stop = x_start + 100,y_start + 100

new_matlike = matlike[x_start:x_stop,y_start:y_stop]

fig,ax = plt.subplots(2,2)

ax[0,0].imshow(matlike,cmap='gray')
ax[0,1].imshow(new_matlike,cmap='gray')

fft2_mat = np.log(1 + np.abs(np.fft.fftshift(np.fft.fft2(matlike))).astype(np.uint16))
fft2_newmat = np.log(1 + np.abs(np.fft.fftshift(np.fft.fft2(new_matlike))).astype(np.uint16))

ax[1,0].imshow(fft2_mat,cmap='gray')
ax[1,1].imshow(fft2_newmat,cmap='gray')

plt.show()
