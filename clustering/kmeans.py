import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from webencodings import mklabels



pd.set_option("display.max_rows",101)
pd.set_option("display.max_columns",101)

# import dataset
digits_train = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra")
digits_test = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes")

print('train: ', digits_train.shape, 'test: ',  digits_test.shape)

# print(digits_train.head(1))

#initial setup
# find a good k
# git messing about again
# inertia = []
# 
# for k in range(1, 101):
#     classifier = KMeans(n_clusters=k)
#     classifier.fit(digits_train)
#     inertia.append(classifier.inertia_)
# 
# plt.plot(range(1, 101), inertia)
# plt.show()

# lets call it 50 for now

# digits_train = normalize(digits_train)
# digits_test = normalize(digits_test)

k = 5

model = KMeans(n_clusters=k)
#model = KNeighborsClassifier(n_neighbors=k)
labels = model.fit_predict(digits_train)
#X_trn = model.transform(digits_train)

prediction = model.predict(digits_test)
#X_tst = model.transform(digits_test)

print('testing score', model.score(digits_test))

fig, (ax1, ax2) = plt.subplots(1,2, sharex = True, sharey=True)

X_trn = digits_train.values[:,0-63]
y_trn = digits_train.values[:,64]
# 
X_tst = digits_test.values[:,0-63]
y_tst = digits_test.values[:,64]

ax1.set_title('training set')
ax2.set_title('testing set')
ax1.scatter(X_trn, y_trn, c=labels, alpha=0.5)
ax2.scatter(X_tst, y_tst, c=prediction,  alpha=0.5)

plt.yticks(np.arange(10), step=1)
plt.show()