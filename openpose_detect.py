# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time
import pandas as pd
from yulib.yudata_preprocessing import pose_normalization
from yulib.yzh_display import drawActionResult
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from keras.optimizers import Adam, SGD
from keras.models import Model

#下面代码开始
try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            #sys.path.append('../../python');
            sys.path.append('/usr/local/python')
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../openpose/models/"
    params["net_resolution"] = "-1x80"#"320x320"
    #params["face"] = True
    params["hand"] = False
    params["camera_resolution"] = "1280x1280"
    #params["hand_net_resolution"] = "368x368"

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    #WrapperPython是在C++里面定义的一个类，初始化的时候需要传入线程管理状态，默认是异步的。
    opWrapper.configure(params)
    #configure是传入用户定义的传输，用dict（）的形式
    opWrapper.start()
    datum = op.Datum()
    cap = cv2.VideoCapture(0)#捕获摄像头
    #cv2.namedWindow("capture",0)#创建视频窗口
    #设置窗口大小
    #cv2.resizeWindow("capture", 1280, 1280)
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (50, 50)#字体的位置
    # fontScale
    fontScale = 1
    # Blue color in BGR
    color = (255, 0, 0)
    
    # Line thickness of 2 px
    thickness = 2
    # model = Sequential()
    # model.add(Dense(units=128, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dense(units=64, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dense(units=16, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dense(units=2, activation='softmax'))
    # model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'],run_eagerly=True)


    class ActionClassifier(object):
    
        def __init__(self, model_path):
            from keras.models import load_model

            self.dnn_model = load_model(model_path)
            self.action_dict = ["handsdown","handsup"]

        def predict(self, skeleton):

            # Preprocess data
            tmp = pose_normalization(skeleton)
            #print(tmp)
            skeleton_input = np.array(tmp).reshape(-1, len(tmp))
            # print(skeleton_input)
                
            # Predicted label: int & string
            predicted_idx = np.argmax(self.dnn_model.predict(skeleton_input))
            prediced_label = self.action_dict[predicted_idx]

            return prediced_label
    classifer=ActionClassifier("./MyModel/yzh_action_recognition.h5")
    
    
    while True:
        ret, frame = cap.read()
        datum.cvInputData=frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        #image=cv2.resize(frame,(640,480))
        # image = cv2.putText(datum.cvOutputData, 'test', org, font, 
        #             fontScale, color, thickness, cv2.LINE_AA)
        '''
        datum.poseKeypoints[0,0:25,0:2]传回来的格式是二维矩阵，需要将他展平
        '''
        input=datum.poseKeypoints[0,0:25,0:2].reshape(-1,datum.poseKeypoints[0].shape[0]*2)
        # print(np.shape(input))
        # print(input)
        # print(input[0])
        predicted_label=classifer.predict(input[0])
        print(type(predicted_label))
        print("prediced label is :", predicted_label)
        image_disp=datum.cvOutputData
        drawActionResult(image_disp, input[0], predicted_label)
        cv2.imshow("acation_recognition", image_disp)
        if cv2.waitKey(20) & 0xFF == ord('q'):#waitkey同时表示每一帧的显示间隔
            break
        #datum.poseKeypoints 是一个三维的数组，
        # 第一个参数代表画面中的人数，
        # 第二个参数代表关节点数量，
        # 第三个参数代表每个关节点的坐标数量
        '''
        我们使用pandas存储每一帧的数据，抛弃置信度。

        '''
        # print("Body keypoints: \n" + str(datum.poseKeypoints[0,0:25,0:2]))
        # print("Body keypoints: \n" , datum.poseKeypoints[0][1][0:2])
        # print("Body keypoints: \n" + str(datum.poseKeypoints.shape[1]))
        
        #print("hand keypoints: \n" + str(datum.handKeypoints))#这个数据可以用narray形式去提取
except Exception as e:
    print(e)
    sys.exit(-1)
