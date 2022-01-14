import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def channel_svg(channel, p=0.1):
    U, S, VT = np.linalg.svd(channel)
    k = int(len(S) * p)  # take the first k values by percentage
    sigma = np.diag(S[:k])  # build singular matrix
    ret = U[:, :k] @ sigma @ VT[:k, :]  # @ => dot product
    ret[ret < 0] = 0
    ret[ret > 255] = 255
    ret = np.rint(ret).astype("uint8")
    return ret


def rebuild_img_by_numpy(img, p):
    (R, G, B) = cv2.split(img)
    ret = cv2.merge([channel_svg(R, p), channel_svg(G, p), channel_svg(B, p)])
    return ret


def rebuild_img_by_sklearn(img, p):
    p = int(img.shape[1] * p)
    (R, G, B) = cv2.split(img)

    pca = PCA(n_components=p)
    r = pca.inverse_transform(pca.fit_transform(R))
    g = pca.inverse_transform(pca.fit_transform(G))
    b = pca.inverse_transform(pca.fit_transform(B))

    ret = cv2.merge([r, g, b])
    return np.rint(ret).astype("uint8")


def show_compressed_img(method, img, p, idx):
    plt.subplot(2, 3, idx)
    plt.axis("off")
    plt.title(str(p))
    plt.imshow(method(img, p))


def plt_init(img):
    plt.figure(figsize=(26, 12))
    plt.tight_layout()

    plt.subplot(2, 3, 1)  # 2 rows 3 colums, 6 images, first one
    plt.title(f"original")
    plt.axis("off")
    plt.imshow(img)


def load_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


if __name__ == "__main__":
    img = load_img("1.jpeg")
    plt_init(img)
    print(img.shape)

    i = 2
    for p in [0.01, 0.05, 0.1, 0.2, 0.5]:
        # show_compressed_img(rebuild_img_by_sklearn, img, p, i)
        show_compressed_img(rebuild_img_by_numpy, img, p, i)
        i += 1
    plt.show()
