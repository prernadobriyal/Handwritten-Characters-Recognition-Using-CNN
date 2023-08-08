

from keras.datasets import mnist
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.subplot(221)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[4], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[5], cmap=plt.get_cmap('gray'))

plt.show()
