import numpy as np 
import nibabel as nib 
import matplotlib.pyplot as plt

nii_path = r"C:\Users\alper\Desktop\testing_zone\32-Flair.nii"

nii_obj = nib.load(nii_path)
fdata = nii_obj.get_fdata()
h,w,d = fdata.shape

print(f'Yükseklik: {h} Genislik: {w} Slice sayisi: {d}')

sqrt = int(np.sqrt(d))

if sqrt %2 == 0 and int(sqrt) + int(sqrt) == int(d):
    sqrt = int(sqrt)

fig,ax = plt.subplots(sqrt,sqrt,figsize=(8,8))
[axX.axis(False) for axY in ax for axX in axY]
dX = 0
for i,j in enumerate(ax):
    for iX,iJ in enumerate(j):
        it_data = fdata[:,:,dX]
        
        fft2 = np.fft.fft2(it_data)
        shifted = np.fft.fftshift(fft2)
        log1p = np.abs(shifted)

        iJ.imshow(log1p)
        iJ.set_title(f'(Dilim {i,iX})')

        dX += 1

plt.show()
