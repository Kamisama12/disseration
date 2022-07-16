# From Python
# It requires OpenCV installed for Python
from dataclasses import dataclass
import sys
import cv2
import os
from sys import platform
import argparse
import time
from numpy import imag, true_divide
import pandas as pd
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

    # Using cv2.putText() method
    image_count=0
    #objectPath="./pose1_handsup/"
    objectPath="./pose2_handsdown/"
    start=time.time()
    #创建一个字典存储数据
    Mydata={#"Nose":[[1,2,3],[1,2,3]],
    #         "Neck":[[4,5,6],[4,5,6]],
    #         "RShoulder":[[7,8,9],[4,5,6]],
            "Nose":[],
            "Neck":[],
            "RShoulder":[],
            "RElbow":[],
            "RWrist":[],
            "LShoulder":[],
            "LElbow":[],
            "LWrist":[],
            "MidHip":[],
            "RHip":[],
            "RKnee":[],
            "RAnkle":[],
            "LHip":[],
            "LKnee":[],
            "LAnkle":[],
            "REye":[],
            "LEye":[],
            "REar":[],
            "LEar":[],
            "LBigToe":[],
            "LSmallToe":[],
            "LHeel":[],
            "RBigToe":[],
            "RSmallToe":[],
            "RHeel":[]
            }
    Mydatacollecting=[[] for i in range(25)]
    while True:
        ret, frame = cap.read()
        datum.cvInputData=frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        image=cv2.resize(frame,(640,480))
        #原始图片保存
        if time.time()-start >= 0.1:#每过0.1s存储一帧
            if ret:
                for i in range(datum.poseKeypoints[0].shape[0]):#长度25  0-24
                    Mydatacollecting[i].append(str(datum.poseKeypoints[0][i][0:2]))#将每个关键点的x，y存到对应位置
                    # print(str(datum.poseKeypoints[0][i][0:2]))
                    # print("测试数据：",str(datum.poseKeypoints[0][i]))
                    # print(Mydatacollecting)
                if not cv2.imwrite(objectPath+"pose_handsdown_%d.jpg"%image_count,image) or \
                    not cv2.imwrite(objectPath+"pose_handsdown_pro%d.jpg"%image_count,datum.cvOutputData):
                    raise Exception("Could not write image")
                image_count+=1
            start=time.time()
        image = cv2.putText(datum.cvOutputData, 'test', org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow("capture", image)
        if cv2.waitKey(20) & 0xFF == ord('q'):#waitkey同时表示每一帧的显示间隔
            break
        #datum.poseKeypoints 是一个三维的数组，
        # 第一个参数代表画面中的人数，
        # 第二个参数代表关节点数量，
        # 第三个参数代表每个关节点的坐标数量
        '''
        我们使用pandas存储每一帧的数据，抛弃置信度。

        '''
        print("Body keypoints: \n" + str(datum.poseKeypoints[0][1][0:2]))
        print("Body keypoints: \n" + str(datum.poseKeypoints.shape[1]))
        
        #print("hand keypoints: \n" + str(datum.handKeypoints))#这个数据可以用narray形式去提取
    start=0
    for i in Mydata:
        Mydata[i]=Mydatacollecting[start]
        start+=1
    df = pd.DataFrame(Mydata)
    df.to_csv('./Mydata_handsdown.csv')
except Exception as e:
    print(e)
    sys.exit(-1)
