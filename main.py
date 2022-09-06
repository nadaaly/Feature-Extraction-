from sklearn.metrics import accuracy_score
from keras.datasets import mnist
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def convert_to_binary(photo, lower, upper):
    photo = (lower < photo) & (photo < upper)
    return photo

def calculate_centriods(photo, r, c):
    centroids = []
    binary_image = 1 * convert_to_binary(photo, 127, 255)
    for row in range(0, photo.shape[0] - r, r):
        for col in range(0, photo.shape[1] - c, c):
            filter = binary_image[row:row + r, col:col + c]
            x_center, y_center = np.argwhere(filter == 1).sum(0) / np.count_nonzero(filter)
            centroids.append(x_center)
            centroids.append(y_center)

    centroids = np.nan_to_num(centroids)
    return centroids


(x_training, y_training), (x_test, y_test) = mnist.load_data()

x_training = x_training[:10000, :, :] #minset training images
y_training = y_training[:10000] #labels
x_test = x_test[:1000, :, :] #minset testing images
y_test = y_test[:1000] #labels for test

final_training = []
for img in x_training:
    centroid = calculate_centriods(img,4,4)
    final_training.append(centroid)

final_test = []
for img in x_test:
    centroid = calculate_centriods(img,4,4)
    final_test.append(centroid)

classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(final_training, y_training)

y_pred = classifier.predict(final_test)

print(accuracy_score(y_test, y_pred))