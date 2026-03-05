import numpy as np 
import nibabel as nib 
import matplotlib.pyplot as plt


nii_path = r"C:\Users\alper\Desktop\testing_zone\32-Flair.nii"
d = nib.load(nii_path).get_fdata().shape[2]

def I(rowX,path):
    return nib.load(path).get_fdata()[:,:,rowX]

def K(kT,s):
    E = lambda kX : np.abs(np.log(kX))**2
    eX = lambda xK : np.abs(np.log1p(xK))

    if s == 0:
        return E(np.fft.fft2(kT))
    else:
        return eX(np.fft.fftshift(np.fft.fft2(kT)))
    
fig,ax = plt.subplots(1,3,figsize=(4,4))


for rowX in range(0,d-1):
    sT = K(I(rowX,nii_path),0)

    kx,ky = np.indices(sT.shape)
    center = np.array(sT.shape) // 2

    r = np.sqrt(((kx - center[0])**2) + ((ky - center[1])**2))
    Nr = np.bincount(r.ravel().astype(int),weights=sT.ravel().astype(int))

    nNorm = Nr / np.bincount(r.ravel().astype(int))

    ax[1].set_title(f'{rowX}. dilimin görüntüsü')
    ax[0].set_title(f'{rowX}. dilimin k-uzayı')
    ax[0].imshow(K(I(rowX,nii_path),0xf2))
    ax[1].imshow(I(rowX,nii_path),cmap='gray')
    ax[2].clear()
    ax[2].plot(nNorm)
    ax[2].set_title(f'{rowX}. dilimin k-uzayının enerji grafiği')
    plt.pause(0.5)


plt.tight_layout()
plt.show()
