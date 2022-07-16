from asyncio.windows_events import NULL
from hashlib import new
from turtle import shape
import h5py
import numpy as np
import tensorflow as tf 
import keras
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Conv2D, Conv1D, MaxPooling2D, concatenate, BatchNormalization
from keras.models import Model
import matplotlib.pyplot as plt



def MyDataProcess():
    def convert_to_one_hot(Y, C):
        Y = np.eye(C)[Y.reshape(-1)].T
        return Y

    file = h5py.File('D:\\\\bristolUNI\\\\dissertation\\\\code\\\\EMG\\\\sEMG_DeepLearning-master\\\\NinaPro-DB1\\\\data\\\\DB1_S1_image.h5','r')#源码使用的数据集
    print(file)
    #下面开始将自己的EMG数据集制作成近似的格式，源码中使用的是10个通道的EMG，我们只有8个通道\
    '''
    源码的数据集是一张图像是120ms*10个通道，若干个对应标签为n的图像。
    我的数据，一个文件代表一个姿势，需要将数据转换成120ms*8个通道，相当于就是12个数据作为一组，每个数据里面是八个通道的读数，以此作为一张图像传入神经网络。
    '''
    #EMG源码中的读取数据集代码
    X = []
    Y = []
    for i in range(10):
        X.append(np.fromfile('D://bristolUNI//dissertation//code//EMG//datacollect//2nd_collect//data%d.dat' % i, dtype=np.uint16).reshape((-1, 8)))
        Y.append(i + np.zeros(X[-1].shape[0]))
    # print("EMG_X:",X,"\n")
    # print("EMG_Y:",Y,"\n")
    # print(X[0],"\n",X[1],"\n",X[2],"\n")
    '''
    print(np.shape(X),np.shape(X[0]))
    print(np.shape(Y),np.shape(Y[0]))
    print(Y[0],"\n",Y[1],"\n",Y[2],"\n")
    print(len(X))#len是10

    '''
    t=np.array([[   [1,2,3],
                    [4,5,6],
                    [7,8,9]  ],

                [   [1,2,3],
                    [4,5,6],
                    [7,8,9]  ],     

                [   [1,2,3],
                    [4,5,6],
                    [7,8,9]  ]    ])
    #print(np.shape(t))
    #print(t[1,:,0:2])#验证过三维的numpy.array数据结构可以用这个方式读取数据

    # X_np=np.array(X)
    # print(np.shape(X_np))
    # for x in X_np:
    #     for j in x:
    #         print(j)
    new_X=[]
    new_Y=[]
    for i in range(len(X)):
        if len(X[i]) :
            for j in X[i]:
                new_X.append(j)
        else:
            print("kong")
    for i in range(len(Y)):
        if len(Y[i]):
            for j in Y[i]:
                new_Y.append(j)
        else:
            print("kong")
    '''
    print(new_X[1])
    print(np.shape(new_X))
    print(new_Y[1])
    print(np.shape(new_Y))

    '''
    # print(12*int((len(new_X)/12)))
    np_X=np.array(new_X[0:12*int((len(new_X)/12))])#抛弃掉不足构成120ms的数据
    print(np_X.shape)
    np_Y=np.array(new_Y[0:12*int((len(new_X)/12))])#抛弃掉不足构成120ms的数据
    print(np_Y.shape)
    #np_X=np_X.flatten().reshape(int(len(new_X)/12),12,8) 
    np_X=np_X.reshape(-1,12,8)#将数据变成符合要求的三维格式，接下的数据处理可以套用原本数据处理的代码
    np_Y=np_Y.reshape(-1,12)[:,0].astype(int)#这里需要转化成整数形式不然后面调用自定义函数时候会报错
    print(np.shape(np_Y))
    print(np_X.shape)
    #理论上到这里数据格式修改完成，后面将所有imageData改成np_X,imageLabel改成np_Y
    imageData   = file['imageData'][:]
    imageLabel  = file['imageLabel'][:] 
    print(imageData)
    print(imageData.shape)
    print(imageLabel.shape)
    file.close()
    # 随机打乱数据和标签
    #N = imageData.shape[0]
    N = np_X.shape[0]
    index = np.random.permutation(N)
    print(index)
    # data  = imageData[index,:,:]
    # label = imageLabel[index]
    data  = np_X[index,:,:]
    label = np_Y[index]
    print(label)
    # 对数据升维,标签one-hot
    data  = np.expand_dims(data, axis=2)
    print(np.shape(data))
    print(data)
    #label = convert_to_one_hot(label,52).T#本来是52，因为我的数据集暂时只有两个姿态，改为2
    label = convert_to_one_hot(label,4).T
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
    return X_train, Y_train,X_test, Y_test

'''
tf.nn.conv2d(
        input,
        filter,
        strides=[1, 1, 1, 1],
        padding='VALID',
        use_cudnn_on_gpu=True,
        data_format='NHWC',
        dilations=[1, 1, 1, 1],
        name=None
    )#该函数形式是用在


'''

'''
input是输入张量，同样包含四个维度，[batch，高度，宽度，通道数]
strides是卷积步长，一个tensor。四个维度分别是[batch_stride, 
    heightstride, widthstride, channel_stride。一般情况下batch_stride和channel_stride都是1
    input是用于做卷积运算的数据，格式为一个tensor，
    拥有四个维度：[batch, in_height, in_width, in_channels]
    filter卷积核信息，同样是一个tensor，卷积核的通道数应该与输入的通道数相当
    四个维度[filter_height, filter_width, in_channels, out_channels]。out_channels是卷积核的数量

'''






def CNN(input_shape=(12,1,8), classes=4): #该深度学习网络的定义用的是keras库里面的定义函数，本来是10个通道，因为我的EMG只有8个通道，改成8使用
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
#model = CNN(input_shape = (12, 1, 8), classes = 2)
# model.summary()
import keras.callbacks
keras.callbacks.Callback
class LossHistory(keras.callbacks.Callback):#继承了回调函数类
    def on_train_begin(self, logs={}):
        print("开始")
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        print(self.accuracy['epoch'].append(logs.get('acc')))
        print("1111111")
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

import time
import os

if __name__ == '__main__':
    model = CNN(input_shape = (12, 1, 8), classes = 4)
    X_train, Y_train,X_test, Y_test=MyDataProcess()
    checkpoint_path = "D:\\bristolUNI\\dissertation\\code\\EMG\\yuzhaohao.h5"
    #checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)



    start = time.time()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = LossHistory() # 创建一个history实例
    model.fit(X_train, Y_train, epochs=80, validation_data=(X_test, Y_test),batch_size=64,callbacks=[history,cp_callback])
    model.save_weights('D:\\bristolUNI\\dissertation\\code\\EMG\\my_checkpoint.h5')
    print(history.accuracy['epoch'])



    preds_train = model.evaluate(X_train, Y_train)
    print("Train Loss = " + str(preds_train[0]))
    print("Train Accuracy = " + str(preds_train[1]))

    preds_test  = model.evaluate(X_test, Y_test)
    print("Test Loss = " + str(preds_test[0]))
    print("Test Accuracy = " + str(preds_test[1]))
    print("model.predict",model.predict(X_test))#训练好的模型直接进行预测
    print("直接调用model：",model(X_test, training=False))#效果和上面的函数调用相同
    
       
    end = time.time()
    print("time:",end-start)

    history.loss_plot('epoch')




    # import pandas as pd
    # file1 = h5py.File('D:\\bristolUNI\\dissertation\\code\\EMG\\sEMG_DeepLearning-master\\SIA_delsys_16_movements\\model_weights.h5','r')
    # #data1=pd.read_hdf('D:\\bristolUNI\\dissertation\\code\\EMG\\sEMG_DeepLearning-master\\SIA_delsys_16_movements\\model_weights.h5')
    # print(file1.keys(),"\n")
    # print(file1.values(),"\n")
    # #h5文件的存储结构
    # print('file1.attrs.keys():', file1.attrs.keys(),"\n")

    # print("f.attrs['layer_names']:", file1.attrs['layer_names'],"\n")
    # print("f['conv2d_3'].attrs.keys():", file1['conv2d_3'].attrs.keys())

    # print("f['conv2d_3'].attrs['weight_names']:", file1['conv2d_3'].attrs['weight_names'])

    # print("f['conv2d_3/kernel:0']:", file1['conv2d_3/conv2d_3/kernel:0'])
    # print(file1['conv2d_3/conv2d_3/kernel:0'][:,:,:,:])#访问group下面的dataset
    # print(file1['conv2d_3/conv2d_3/bias:0'][:])#访问group下面的dataset
    # # print(pra1)
    # file1.close()
    # input_shape=(12,1,10)
    # print(type(input_shape))


    import pandas as pd
    file1 = h5py.File('D:\\bristolUNI\\dissertation\\code\\EMG\\yuzhaohao.h5','r')
    #data1=pd.read_hdf('D:\\bristolUNI\\dissertation\\code\\EMG\\sEMG_DeepLearning-master\\SIA_delsys_16_movements\\model_weights.h5')
    # print(file1.keys(),"\n")
    # print(file1.values(),"\n")
    # #h5文件的存储结构

    # print('file1[conv1].attrs.keys():', file1['conv1'].attrs.keys(),"\n")

    # print('file1.attrs.keys():', file1.attrs.keys(),"\n")

    # print("f.attrs['layer_names']:", file1.attrs['layer_names'],"\n")
    # print("f['conv1'].attrs.keys():", file1['conv1'].attrs.keys())

    # print("f['conv1'].attrs['weight_names']:", file1['conv1'].attrs['weight_names'])

    # print("f['conv1/kernel:0']:", file1['conv1/conv1/kernel:0'])
    # print(file1['conv1/conv1/kernel:0'][:,:,:,:])#访问group下面的dataset
    # print(file1['conv1/conv1/bias:0'][:])#访问group下面的dataset
    # # print(pra1)
    # file1.close()
    # input_shape=(12,1,10)
    # print(type(input_shape))
