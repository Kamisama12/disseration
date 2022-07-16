import imp
import h5py
import numpy as np
import tensorflow as tf 
import keras
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Conv2D, Conv1D, MaxPooling2D, concatenate, BatchNormalization
from keras.models import Model
import matplotlib.pyplot as plt
import sys
sys.path.append("D:\\bristolUNI\\dissertation\\code\\EMG")
from MyLearningModel import MyDataProcess

'''
原始的数据处理代码，因为用我自己的数据集所以先注释掉
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

file = h5py.File('D:\\bristolUNI\\dissertation\\code\\EMG\\sEMG_DeepLearning-master\\NinaPro-DB1\\data\\DB1_S1_image.h5','r')
print(file)
imageData   = file['imageData'][:]
imageLabel  = file['imageLabel'][:] 
print(imageData)
print(imageData.shape)
print(imageLabel.shape)
file.close()
# 随机打乱数据和标签
N = imageData.shape[0]
index = np.random.permutation(N)
data  = imageData[index,:,:]
label = imageLabel[index]

# 对数据升维,标签one-hot
data  = np.expand_dims(data, axis=2)
print(np.shape(data))
print(data)
label = convert_to_one_hot(label,52).T
print("label:",label.shape)
# 划分数据集
N = data.shape[0]
num_train = round(N*0.8)
X_train = data[0:num_train,:,:]
Y_train = label[0:num_train,:]
X_test  = data[num_train:N,:,:]
Y_test  = label[num_train:N,:]

print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
    
'''


X_train, Y_train,X_test, Y_test=MyDataProcess()#处理我自己的数据

def CNN(input_shape=(12,1,8), classes=4): #该深度学习网络的定义用的是keras库里面的定义函数，修改了10通道为8通道
    X_input = Input(input_shape)
    X = Conv2D(filters=32,kernel_size=(6,1),strides=(1,1),padding='same',activation='relu',name='conv1')(X_input)
    #filters: 整数，输出空间的维度 （即卷积中滤波器的输出数量）。可以理解为tf中的卷积核数量=output_channel
    #kernel_size: 一个整数，或者单个整数表示的元组或列表， 指明 1D 卷积窗口的长度。
    #kernel_initializer: kernel 权值矩阵的初始化器 缺省操作时候应该是有默认参数
    
    X = Conv2D(filters=64,kernel_size=(4,1),strides=(1,1),padding='same',activation='relu',name='conv2')(X)
    X = MaxPooling2D(pool_size=(2,1), strides=(2,1), name='pool1')(X)
 
    
    X = Conv2D(filters=128,kernel_size=(2,1),strides=(1,1),padding='same',activation='relu',name='conv3')(X)
    X = MaxPooling2D(pool_size=(2,1), strides=(2,1), name='pool2')(X)
    
    X = Flatten(name='flatten')(X)
    X = Dropout(0.5)(X)#Dropout 包括在训练中每次更新时， 将输入单元的按比率随机设置为 0， 这有助于防止过拟合。
    X = Dense(128,activation='relu',name='fc1')(X)#全连接层，输出是128维vector
    X = Dropout(0.5)(X)
    X = Dense(classes, activation='softmax', name='fc2')(X)#输出52维矩阵，使用softmax激活概率最大的姿势
    
    model = Model(inputs=X_input, outputs=X, name='CNN')
    return model

if __name__ == '__main__':

    model = CNN(input_shape = (12, 1, 8), classes = 4)
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    checkpoint_path = "D:\\bristolUNI\\dissertation\\code\\EMG\\yuzhaohao.h5"

    loss, acc = model.evaluate(X_test, Y_test, verbose=2)
    print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))#format

    model.load_weights(checkpoint_path)
    print(X_test[0])
    print(X_test.shape)


    loss, acc = model.evaluate(X_test,Y_test, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))