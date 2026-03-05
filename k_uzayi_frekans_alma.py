import numpy as np 
import nibabel as nib 
import matplotlib.pyplot as plt

# NIfTI dosyasını yükle
nii_path = r"C:\Users\alper\Desktop\testing_zone\32-Flair.nii"
row = 0
I = nib.load(nii_path).get_fdata()[:,:,row]

# Fourier dönüşümü fonksiyonu
def K(x):
    return np.abs(np.log(np.fft.fftshift(np.fft.fft2(x))))**2

# Fourier dönüşümünü uygula
sT = K(I)

# İndeksleri al
kx, ky = np.indices(sT.shape)
center = np.array(sT.shape) // 2

# Radial mesafe hesapla
r = np.sqrt((kx - center[0])**2 + (ky - center[1])**2)

# Mesafeyi tam sayıya yuvarlayıp, her mesafe için enerji toplamı
r_rounded = np.round(r).astype(int)

# Enerji toplama
Nr = np.bincount(r_rounded.ravel(), weights=sT.ravel())

# Piksel sayısı (normalizasyon için)
pixel_count = np.bincount(r_rounded.ravel())

# Normalize et (ortalama enerji)
Nr_norm = Nr / pixel_count

# Grafiği çiz
plt.plot(Nr_norm)
plt.title("Normalized Radial Energy Signal")
plt.xlabel("Radius (k)")
plt.ylabel("Normalized Energy")
plt.grid(True)
plt.show()
