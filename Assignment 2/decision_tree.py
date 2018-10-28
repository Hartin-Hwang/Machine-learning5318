import time
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

training_images, training_labels = load_mnist('/Users/jaydenli/PycharmProjects/assignment2/new knn/', kind='train')
testing_images, testing_labels = load_mnist('/Users/jaydenli/PycharmProjects/assignment2/new knn/', kind='t10k')

pca = PCA(n_components=100)
pca.fit(training_images)
pca_data = pca.transform(training_images)
pca_test = pca.transform(testing_images)

## 10-fold cross-validate get the best model , depth = 15.

depth= [5,10,15,20,25,30,40,50,100]
d_score=[]
running_time=[]
for d in depth:
    start = time.clock()
    dt = DecisionTreeClassifier(max_depth=d)
    scores = cross_val_score(dt, pca_data, training_labels, cv=10, scoring='accuracy')
    print(scores.mean())
    d_score.append(scores.mean())
    end = time.clock()
    print('the running time is: ', (end - start))
    running_time.append(end - start)
plt.plot(depth, d_score)
plt.xlabel('Value of max_depth')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
plt.plot(depth, running_time)
plt.xlabel('Value of max_depth')
plt.ylabel('running time')
plt.show()

## Using depth=15, predict testing data
start= time.clock()
dt = DecisionTreeClassifier(max_depth=15)
dt.fit(pca_data,training_labels)
predict = dt.predict(pca_test)
print('the accuracy is ', accuracy_score(testing_labels, predict))
print('the recall is ', recall_score(testing_labels, predict,average=None))
print('the precision is ', precision_score(testing_labels, predict,average=None))
print('the confusion_matrix is ', confusion_matrix(testing_labels, predict))
end=time.clock()
print('the running time is: ', (end-start))