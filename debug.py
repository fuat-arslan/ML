

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from Models import K_means as qq
from Metrics import Metrics
# create dataset
X, y = make_blobs(
   n_samples=150, n_features=2,
   centers=3, cluster_std=0.5,
   shuffle=True, random_state=0
)

# plot
plt.scatter(
   X[:, 0], X[:, 1],
   c='white', marker='o',
   edgecolor='black', s=50
)
plt.show()

model = qq.KMeans(3, "random", "euclidian")
cv = Metrics.kFold()
error = cv.eval(model,X,y)