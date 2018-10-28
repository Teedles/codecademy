import numpy as np
from matplotlib import pyplot as plt

from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()

plt.gray()
plt.matshow(digits.images[100])
plt.show()

k = 10

model = KMeans(n_clusters = k)

model.fit(digits.data)

plt.gray()

plt.suptitle("centroid")
fig = plt.figure(figsize=(8,3))

for i in  range(10):
  ax=fig.add_subplot(2,5,1 + i)
  ax.imshow(model.cluster_centers_[i].reshape((8,8)), cmap=plt.cm.binary)
  
plt.show()