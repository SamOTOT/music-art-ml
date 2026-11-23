import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image

def get_palette(path, k=5):
    img = np.array(Image.open(path).convert("RGB"))
    pixels = img.reshape(-1, 3)
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    km.fit(pixels)
    palette = km.cluster_centers_.astype(np.uint8)
    share = np.bincount(km.labels_) / len(pixels)
    return palette, share

if __name__ == "__main__":
    img = plt.imread("photo.jpg")
    palette, share = get_palette("photo.jpg")
    print(palette)
    print(np.round(share * 100), "%")

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 5))
    left.imshow(img)
    left.set_title("Photo")
    left.axis("off")
    right.imshow([palette])
    right.set_title("Palette (k-means, k=5)")
    right.axis("off")
    plt.savefig("palette.png", dpi=150, bbox_inches="tight")
    plt.show()