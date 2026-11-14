import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

img = plt.imread("photo.jpg")
print(img.shape, img.dtype)

pixels = img.reshape(-1, 3)    # the grid of pixels → one long list, 3 columns (R, G, B)
#print(pixels.shape)
#print(pixels[:5])

km = KMeans(n_clusters=5, n_init=10, random_state=0)
km.fit(pixels)                                   # ← the learning happens here

palette = km.cluster_centers_.astype(np.uint8)   # the 5 centres = 5 colours
print(palette)

share = np.bincount(km.labels_) / len(pixels)    # how much of the photo each colour covers
print(np.round(share * 100), "%")

fig, (left, right) = plt.subplots(1, 2, figsize=(10, 5))
left.imshow(img)
left.set_title("Photo")
left.axis("off")
right.imshow([palette])                          # 5 colour blocks in a row
right.set_title("Palette (k-means, k=5)")
right.axis("off")
plt.savefig("palette.png", dpi=150, bbox_inches="tight")
plt.show()