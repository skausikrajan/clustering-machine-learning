import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.cluster import KMeans
from google.colab import files

def load_and_preprocess_image(image_path):
    image = io.imread(image_path)
   
    image = np.array(image, dtype=np.float64) / 255
    
    w, h, d = original_shape = tuple(image.shape)
    assert d == 3
    image_array = np.reshape(image, (w * h, d))
    return image, image_array, w, h
def kmeans_segmentation(image_array, n_colors, w, h):
    
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array)
    
    labels = kmeans.predict(image_array)
   
    segmented_image = labels.reshape(w, h)
    
    cluster_centers = kmeans.cluster_centers_
    return segmented_image, cluster_centers

def visualize_segmentation(segmented_image, cluster_centers, w, h):
    plt.figure(figsize=(10, 5))
    plt.imshow(segmented_image, cmap=plt.cm.nipy_spectral)
    plt.title('Segmented Image')
    plt.axis('off')
    
    plt.figure(figsize=(10, 2))
    bar_width = w // len(cluster_centers)
    bar_height = 20
    cluster_centers = cluster_centers * 255
    for i, color in enumerate(cluster_centers):
        plt.gca().add_patch(plt.Rectangle((i * bar_width, 0), bar_width, bar_height, fc=color / 255))
    plt.title('Cluster Colors')
    plt.axis('off')
    plt.show()

uploaded = files.upload()


image_path = list(uploaded.keys())[0]


image, image_array, w, h = load_and_preprocess_image(image_path)

n_colors = 5  

segmented_image, cluster_centers = kmeans_segmentation(image_array, n_colors, w, h)

visualize_segmentation(segmented_image, cluster_centers, w, h)

