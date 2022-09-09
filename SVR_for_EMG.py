from statistics import mode
from sklearn.svm import SVR,LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.externals import joblib
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# n_samples, n_features = 10, 5
# rng = np.random.RandomState(10)
# y = rng.randn(n_samples)
# X = rng.randn(n_samples, n_features)



# print(y,X)




# X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=2)
data=pd.read_csv('/home/y/mycode/EMG_part/regress data/8-25-GP-SVR/25-20220825T094701Z-001/25/robot_state_svm.csv',header=None)
#data=data.to_numpy()[50:len(data),...]#配合采集时候的处理
data=data.to_numpy()

X=data[...,3:5]
Y=data[...,16]

scaler=StandardScaler()
scaler.fit(X)

SVR_regr = make_pipeline(scaler, SVR(C=1.0, epsilon=0.5))
# linear_SVR = make_pipeline(StandardScaler(),LinearSVR(random_state=0, tol=1e-5))



print(X)
print(X.shape)
SVR_regr.fit(X, Y)
# x_test=np.linspace(0,20,100).reshape(-1,2)
# y=SVR_regr.predict(x_test)

# fig=plt.figure()
# ax=plt.axes(projection='3d')
# ax.plot3D(x_test[...,0],x_test[...,1],y,'gray')
# plt.plot(x_test,y)
# plt.show()

# xx=np.array([ 0.43302619,1.20303737,-0.96506567,1.02827408 ,0.22863013]).reshape(1,-1)
# yy=np.array([1.3315865 ]).reshape(1,-1)

# print(SVR_regr.predict(xx))


# joblib.dump(SVR_regr,'/home/y/mycode/EMG_part/regression_trained_model/SVR_train_xy_force')
model=joblib.load('/home/y/mycode/EMG_part/regression_trained_model/SVR_train_xy_force')
print(model.predict(np.array([1,2]).reshape(-1,2)))

# from sklearn.preprocessing import StandardScaler
# data = [[0, 1], [0, 3], [1, 2], [1, 2]]
# scaler = StandardScaler()
# scaler.fit(data)
# print(scaler.mean_)
# print(scaler.scale_)
# print(scaler.transform(data))
# print(scaler.transform([[2, 2]]))

