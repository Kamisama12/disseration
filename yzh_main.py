from EMG_part.TrainingmodelCompare import My_worker, Yzh_CNN_Model, KNN_Classifier, emg_mode
from pyomyo.Classifier import Live_Classifier, Classifier, MyoClassifier, EMGHandler
import pygame
from pygame.locals import *
import numpy as np
from pynput.keyboard import Key, Controller, Listener
import argparse
from openpose_part.yulib.yzh_display import drawActionResult
from openpose_part.yulib.yudata_preprocessing import pose_normalization
import threading
import multiprocessing
import cv2
import time
import os
from sys import platform
import sys

sys.path.append("./EMG_part")


lock = threading.Lock()
emg_que = multiprocessing.Queue()  # 用作进程之间的通讯队列


'''
下面是EMG的参数
'''


def on_press(key):
    print('{0} 按下'.format(key))
# 释放键盘时回调的函数,如果需要从按键按下的情况去做对应的动作就在这里修改


def on_release(key):
    print('{0} 松开'.format(key))
    if key == Key.esc:
        # 停止监听
        return False


def key_emg_handler(pose):
    TRAINING_MODE = False  # 训练模型的时候设置为True
    keyboard = Controller()  # 初始化按键控制器
    print("Pose detected:", pose)
    # 根据我们的姿势类别进行动作
    if ((pose == 0) and (TRAINING_MODE == False)):
        print("0,w")
        # Press and release space
        keyboard.press('w')
        time.sleep(0.1)  # 按下一秒
        keyboard.release('w')
    if ((pose == 1) and (TRAINING_MODE == False)):
        print("1,s")
        # Press and release space
        keyboard.press('s')
        time.sleep(0.1)
        keyboard.release('s')


def YZH_run_gui(yzh_classifier, scr,font, w=800, h=320):
    
    # 显示EMG的数据函数
    # Handle keypresses
    
    for ev in pygame.event.get():
        if ev.type == QUIT or (ev.type == KEYDOWN and ev.unicode == 'q'):
            raise KeyboardInterrupt()

    # Plotting
    scr.fill((0, 0, 0), (0, 0, w, h))
    r = yzh_classifier.history_cnt.most_common(1)[0][0]

    for i in range(10):
        x = 0
        y = 0 + 30 * i
        # Set the barplot color
        # 判断当前的姿势是不是r（预测姿势），是的话设置当前的color。
        clr = yzh_classifier.cls.color if i == r else (255, 255, 255)

        txt = font.render('%5d' % (yzh_classifier.cls.Y == i).sum(),
                          True, (255, 255, 255))
        scr.blit(txt, (x + 20, y))

        txt = font.render('%d' % i, True, clr)
        scr.blit(txt, (x + 110, y))

        # Plot the barchart plot
        scr.fill((0, 0, 0), (x+130, y + txt.get_height() /
                 2 - 10, len(yzh_classifier.history) * 20, 20))
        scr.fill(clr, (x+130, y + txt.get_height() /
                 2 - 10, yzh_classifier.history_cnt[i] * 20, 20))

    pygame.display.flip()


def YZH_emg_init(on_press, on_release, key_emg_handler):  # 在主进程
    # m = MyoClassifier(KNN_Classifier(),
    #                   mode=emg_mode.PREPROCESSED, Networkmode=False)
    m = MyoClassifier(
        Yzh_CNN_Model(), mode=emg_mode.PREPROCESSED, Networkmode=True)
    hnd = EMGHandler(m)
    m.connect()
    m.add_raw_pose_handler(key_emg_handler)
    yzh_CnnModel = Yzh_CNN_Model()

    def My_Handler(emg, movement):
        # q.put(emg)  # 用于打印数据的多进程队列
        hnd(emg, movement)  # 存储数据的处理函数，hnn.emg存储实时数据
    m.add_emg_handler(My_Handler)
    m.set_leds([128, 0, 0], [128, 0, 0])
    # Vibrate to know we connected okay
    m.vibrate(1)
    listener = Listener(on_press=on_press, on_release=on_release)
    listener.start()
    pygame.init()
    w=800
    h=320
    scr = pygame.display.set_mode((w, h))
    font = pygame.font.Font(None, 30)
    return m, yzh_CnnModel,scr,font


def process_emg_data_get(m):
    # 传入一个Myoclassifier类，运行run函数接受数据，数据存在类里面的data_sav变量
    while(1):
        m.run()
        


def get_emg_data(m,emg_raw_data_sav):  # 返回一帧可以直接放入predict的数据
    if not (m.data_sav.empty()):
        emgdata = m.data_sav.get()
        # print("emgdata:",emgdata)
        emg_raw_data_sav.append(emgdata)
        print("emg_raw_data_sav:",emgdata)
        print("shape of emg_raw_data_sav",np.shape(emgdata))
        print("type of emg_raw_data_sav:",type(emgdata))
        return emgdata


'''
下面是openpose的参数
'''
op_que = multiprocessing.Queue()  # 用作进程之间的通讯队列


class ActionClassifier(object):
    def __init__(self, model_path):
        from keras.models import load_model

        self.dnn_model = load_model(model_path)
        # 标签的位置顺序需要和对yutraining.py文件里面转化成虚拟变量之后的标签序号对应
        self.action_dict = ["handsdown", "handsup"]

    def predict(self, skeleton):

        # Preprocess data
        tmp = pose_normalization(skeleton)
        # print(tmp)
        skeleton_input = np.array(tmp).reshape(-1, len(tmp))
        # print(skeleton_input)
        # Predicted label: int & string
        predicted_idx = np.argmax(self.dnn_model.predict(skeleton_input))
        prediced_label = self.action_dict[predicted_idx]
        return prediced_label,predicted_idx


def predict_fun(classifier, data):

    predicted_label,predicted_idx = classifier.predict(data)
    print(type(predicted_label))
    print("prediced label is :", predicted_label)
    # if predicted_label == 'handsdown':
    #     print("this is handsdown ")
    # if predicted_label == 'handsup':
    #     print("this is handsup ")
    return predicted_label,predicted_idx


def YZH_Openpose_init():
    try:
        # Import Openpose (Windows/Ubuntu/OSX)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        try:
            # Windows Import
            if platform == "win32":
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append(dir_path + '/../../python/openpose/Release')
                os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + \
                    '/../../x64/Release;' + dir_path + '/../../bin;'
                import pyopenpose as op
            else:
                # Change these variables to point to the correct folder (Release/x64 etc.)
                # sys.path.append('../../python');
                sys.path.append('/usr/local/python')
                # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
                # sys.path.append('/usr/local/python')
                from openpose import pyopenpose as op
        except ImportError as e:
            print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e

        # Flags
        parser = argparse.ArgumentParser()
        parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg",
                            help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
        args = parser.parse_known_args()

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = "../openpose/models/"
        params["net_resolution"] = "80x80"  # "320x320"
        #params["face"] = True
        params["hand"] = False
        # params["model_pose"]="COCO"
        #params["camera_resolution"] = "1280x1280"
        #params["hand_net_resolution"] = "192x192"

        # Add others in path?
        for i in range(0, len(args[1])):
            curr_item = args[1][i]
            if i != len(args[1])-1:
                next_item = args[1][i+1]
            else:
                next_item = "1"
            if "--" in curr_item and "--" in next_item:
                key = curr_item.replace('-', '')
                if key not in params:
                    params[key] = "1"
            elif "--" in curr_item and "--" not in next_item:
                key = curr_item.replace('-', '')
                if key not in params:
                    params[key] = next_item

        # Starting OpenPose
        opWrapper = op.WrapperPython()
        # WrapperPython是在C++里面定义的一个类，初始化的时候需要传入线程管理状态，默认是异步的。
        opWrapper.configure(params)
        # configure是传入用户定义的传输，用dict（）的形式
        opWrapper.start()
        datum = op.Datum()
        cap = cv2.VideoCapture(0)  # 捕获摄像头
        classifer_openpose = ActionClassifier(
            "./openpose_part/MyModel/yzh_action_recognition.h5")
        return opWrapper, datum, op, cap, classifer_openpose
    except Exception as e:
        print(e)
        sys.exit(-1)


def YZH_Openpose_detect(predict, opWrapper, datum, op, cap, classifer_openpose,op_raw_sav,op_label_sav,op_saving_flag):  #这个函数放在循环里面调用
    # 下面代码开始
    # try:
    # opWrapper,datum,op=YZH_Openpose_init()
    # cap = cv2.VideoCapture(0)  # 捕获摄像头
    # # cv2.namedWindow("capture",0)#创建视频窗口
    # # 设置窗口大小
    # #cv2.resizeWindow("capture", 1280, 1280)
    # classifer_openpose=ActionClassifier("./openpose_part/MyModel/yzh_action_recognition.h5")
    # while True:
    ret, frame = cap.read()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    # image=cv2.resize(frame,(640,480))
    # image = cv2.putText(datum.cvOutputData, 'test', org, font,
    #             fontScale, color, thickness, cv2.LINE_AA)
    '''
            datum.poseKeypoints[0,0:25,0:2]传回来的格式是二维矩阵，需要将他展平，展平之后就是每个关节的xy分开，总共会是50个成员
            第一个元素的人数序号，因为我们默认只有一个用户，所以切片第一个维度直接取0
            '''
    input = datum.poseKeypoints[0, 0:25,
                                0:2].reshape(-1, datum.poseKeypoints[0].shape[0]*2)
    predicted_label,predicted_label_id = predict(classifer_openpose, input[0])
    if op_saving_flag is True:
        op_raw_sav.append(input[0])
        print("shape of openpose raw data after reshape:",np.shape(input[0]))
        print("openpose raw data after reshape:",input[0])
        op_label_sav.append(predicted_label_id)
        print("my openpose return label index:",predicted_label_id)
        print("type of my label index:",type(predicted_label_id))
    image_disp = datum.cvOutputData
    #print("shape of output:", image_disp.shape)
    # input = pose_normalization(input[0])
    drawActionResult(image_disp, input[0], predicted_label)
    # image_disp = cv2.putText(image_disp,'test', org, font,
    #             fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow("acation_recognition", cv2.resize(
        image_disp, (0, 0), fx=1.5, fy=1.5))
        #if cv2.waitKey(50) & 0xFF == ord('q'):  # waitkey同时表示每一帧的显示间隔
            #break

    # datum.poseKeypoints 是一个三维的数组，
    # 第一个参数代表画面中的人数，
    # 第二个参数代表关节点数量，
    # 第三个参数代表每个关节点的坐标数量
    '''
            我们使用pandas存储每一帧的数据，抛弃置信度。
            '''
    # print("Body keypoints: \n" + str(datum.poseKeypoints[0,0:25,0:2]))
    # print("Body keypoints: \n" , datum.poseKeypoints[0][1][0:2])
    # print("Body keypoints: \n" + str(datum.poseKeypoints.shape[1]))
    # print("hand keypoints: \n" + str(datum.handKeypoints))#这个数据可以用narray形式去提取
    # except Exception as e:
    #     print(e)
    #     sys.exit(-1)


'''
把主函数初始化需要的内容集成在下面的函数
'''

def YZH_main_init():
    emg_class, yzh_CnnModel ,scr,font= YZH_emg_init(
        on_press, on_release, key_emg_handler)
    opWrapper, datum, op, cap, classifer_openpose = YZH_Openpose_init()

    return emg_class, yzh_CnnModel, scr,font,opWrapper, datum, op, cap, classifer_openpose
    #return opWrapper, datum, op, cap, classifer_openpose

'''
主函数
'''
def YZH_main_run(emg_raw_data_sav,emg_label_sav,op_raw_sav,op_label_sav):
    try:
        op_saving_flag=False
        emg_class, yzh_emg_CnnModel,scr,font,\
                opWrapper, datum, op, cap, classifer_openpose = YZH_main_init()
        # opWrapper, datum, op, cap, classifer_openpose = YZH_main_init()
        emg_dataProcess=multiprocessing.Process(target=process_emg_data_get,args=(emg_class,))
        emg_dataProcess.daemon=True
        emg_dataProcess.start()
        while True:
            emg_data_forNetwork = get_emg_data(emg_class,emg_raw_data_sav)
            YZH_run_gui(emg_class,scr,font)
            if emg_data_forNetwork != None:
                    # print("emg_data_forNetwork", emg_data_forNetwork)
                op_saving_flag=True
                y = yzh_emg_CnnModel.classify(emg_data_forNetwork)
                emg_label_sav.append(y)#把每一帧识别的结果存下来
                print("posedect:", y)
                emg_class.history_cnt[emg_class.history[0]] -= 1
                    #最左端出队列，计数器中最左端的键的计数减一
                emg_class.history_cnt[y] += 1
                    #新传进来的键值y加一
                emg_class.history.append(y)

                r, n = emg_class.history_cnt.most_common(1)[0]
                    #counter.most_common(n)函数返回的是列表中出现次数排在前n位的元素，元组形式返回('值'，'出现次数')
                if emg_class.last_pose is None or (n > emg_class.history_cnt[emg_class.last_pose] + 5 and n > emg_class.hist_len / 2):
                    	#队列中的出现最多的判断姿态大于上一次判断的姿态的个数5个同时超过一半是这个姿态就刷新姿势
                    emg_class.on_raw_pose(r)#打印pose
                    emg_class.last_pose = r
            YZH_Openpose_detect(predict_fun, opWrapper, datum, op, cap, classifer_openpose,op_raw_sav,op_label_sav,op_saving_flag)
            op_saving_flag=False
            if cv2.waitKey(20) & 0xFF == ord('q'):  # waitkey同时表示每一帧的显示间隔
                break
        print(np.shape(emg_raw_data_sav),'\n',np.shape(emg_label_sav),'\n',np.shape(op_raw_sav),'\n',np.shape(op_label_sav))
    except Exception as e:
        print(e)
        print("程序结束")
        sys.exit(-1)



if __name__ == "__main__":
    emg_test1=[]
    emg_test2=[]
    op_test1=[]
    op_test2=[]
    YZH_main_run(emg_test1,emg_test2,op_test1,op_test2)