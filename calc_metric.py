from sklearn.cluster import KMeans
from collections import Counter


def calc_metric(image, x, y, w, h):
    crop_image = image[y:y+h, x:x+w]
    crop_image = crop_image.reshape((crop_image.shape[0] * crop_image.shape[1], 3))

    clt = KMeans(n_clusters=5)
    labels = clt.fit_predict(crop_image)

    label_counts = Counter(labels)
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

    return tuple(dominant_color.astype(int))
