from sklearn.datasets import load_digits

digits = load_digits()
#print(digits.DESCR)

print(digits.data[:2])
print(digits.data.shape)
print(digits.target[:2])
print(digits.target.shape)

print(digits.images[:2])

import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=4,ncols=6,figsize=(6,4))

# Zip is usefule for bundling up iterators to produce as one. 
for item in zip(axes.ravel(), digits.images, digits.target):
    axes, image, target = item
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target)

plt.tight_layout()
#plt.show()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

# Load the training data in the model using the fit method.
knn.fit(X=x_train, y=y_train)

predicted = knn.predict(X=x_test)

expected = y_test

print(predicted[:20])
print(predicted[:20])