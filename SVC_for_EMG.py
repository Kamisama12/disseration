import imp
from operator import imod
from sklearn.svm import SVC
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
X = [[0, 0], [1, 1]]
Y = [0, 1]

clf.fit(X,Y)

print(clf.predict([[2,3]]))