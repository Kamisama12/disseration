


from sklearn.metrics import mean_squared_error  # metric MSE
from sklearn.metrics import r2_score  # metric R^2
from sklearn.model_selection import train_test_split  # dataset split
import multiprocessing
from collections import Counter, deque
from concurrent.futures import process
import struct
import sys
import time

import pygame
from pygame.locals import *
import numpy as np
from sklearn import neighbors

from pyomyo import Myo, emg_mode
from pyomyo.Classifier import Live_Classifier, Classifier, MyoClassifier, EMGHandler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from pynput.keyboard import Key, Controller, Listener
from RFmodel import *
import os
#下面的代码出现DNNlibrary notfound的时候加入
os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
sys.path.append("./EMG_part")
#import MyLearningModelPredict

SUBSAMPLE = 3
K = 15


class KNN_Classifier(Classifier):
    '''Live implimentation of SkLearns KNN'''
    # 继承了Classifier，重写了train和classify
    # Classify类里面有存储和读取数据的函数,需要指定路径

    def __init__(self):
        Classifier.__init__(self)

    def train(self, X, Y):
        self.X = X
        self.Y = Y
        self.model = None
        if self.X.shape[0] >= K * SUBSAMPLE:
            self.model = neighbors.KNeighborsClassifier(
                n_neighbors=K, algorithm='kd_tree')
            self.model.fit(self.X[::SUBSAMPLE], self.Y[::SUBSAMPLE])

    def classify(self, emg):
        # 分类函数调用在MyoClassifier里面的计数函数内，实时值存在last_pose中
        # 最终输出用户的值为经过25帧过滤之后的结果。
        x = np.array(emg).reshape(1, -1)  # 压缩成一行
        pred = self.model.predict(x)
        return int(pred[0])


class Yzh_CNN_Model(Classifier):
    '''这个分类器自定义自己用的神经网络'''
    # 继承了Classifier，重写了train和classify
    # Classify类里面有存储和读取数据的函数,需要指定路径

    def __init__(self):
        Classifier.__init__(self)
        from keras.models import load_model
        # self.model = MyLearningModelPredict.CNN(
        #     input_shape=(12, 1, 8), classes=4)
        # self.model.summary()
        # self.model.compile(
        #     optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # self.chenkpoint = "./EMG_part/yuzhaohao.h5"
        #self.model.load_weights(self.chenkpoint)
        # self.model=load_model('/home/y/mycode/EMG_part/yzh_3poses_50epo_3_collect.h5')
        # self.model=load_model('/home/y/mycode/EMG_part/yuzhaohaoConv2D_200ms_3poses_100epo_3collect.h5')
        self.model=load_model('/home/y/mycode/EMG_part/200ms_3collect_100epo_20step.h5')
        
    def classify(self, emg):
        # 分类函数调用在MyoClassifier里面的计数函数内，实时值存在last_pose中
        # 因为这个classify是在底层会调用的，因此引入CNN网络就在这里使用
        # 最终输出用户的值为经过25帧过滤之后的结果。
        # x = np.array(emg).reshape(1,-1)#源码中是压缩成一行,1*8?，我们修改成神经网络预测需要的数据形状
        x = np.array(emg).reshape(1, 200,8,1)
        # print(x)

        result = self.model.predict(x)  # 该函数返回具体概率数值
        # result=self.model(x, training=False)#调用model预测结果,这个函数调用返回的是tensor类型
        # print(result)
        # print(max(result[0]))
        #arg_maxindex = np.argmax(result, axis=1)
        #aaa = arg_maxindex[0]
        #print(np.argmax(result, axis=1))  # 按每行求出最大值的索引
        # 返回计算概率最大的值对应的索引，引出[0]，否则narray数据类型是unhashable的
        return np.argmax(result, axis=1)[0]
        '''
		
		'''


def text(scr, font, txt, pos, clr=(255, 255, 255)):
    scr.blit(font.render(txt, True, clr), pos)


class SVM_Classifier(Live_Classifier):
    '''
    Live implimentation of an SVM Classifier
    '''
    # 重写了live分类器里面的train和classify

    def __init__(self):
        Live_Classifier.__init__(self, None, "SVM", (100, 0, 100))

    def train(self, X, Y):
        self.X = X
        self.Y = Y
        try:
            if self.X.shape[0] > 0:
                clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
                #clf = make_pipeline(StandardScaler(), SVC(kernel="linear", C=0.025))

                clf.fit(self.X, self.Y)
                self.model = clf
        except:
            # SVM Errors when we only have data for 1 class.
            self.model = None

    def classify(self, emg):
        if self.X.shape[0] == 0 or self.model == None:
            # We have no data or model, return 0
            return 0

        x = np.array(emg).reshape(1, -1)
        pred = self.model.predict(x)
        return int(pred[0])


class DC_Classifier(Live_Classifier):
    '''
    Live implimentation of Decision Trees
    '''

    def __init__(self):
        Live_Classifier.__init__(self, DecisionTreeClassifier(
        ), name="DC_Classifier", color=(212, 175, 55))


class XG_Classifier(Live_Classifier):
    '''
    Live implimentation of XGBoost
    '''

    def __init__(self):
        Live_Classifier.__init__(
            self, XGBClassifier(), name="xgboost", color=(0, 150, 150))


class LR_Classifier(Live_Classifier):
    '''
    Live implimentation of Logistic Regression
    '''

    def __init__(self):
        Live_Classifier.__init__(self, None, name="LR", color=(100, 0, 100))

    def train(self, X, Y):
        self.X = X
        self.Y = Y
        try:
            if self.X.shape[0] > 0:
                self.model = LogisticRegression()
                self.model.fit(self.X, self.Y)
        except:
            # LR Errors when we only have data for 1 class.
            self.model = None

    def classify(self, emg):
        if self.X.shape[0] == 0 or self.model == None:
            # We have no data or model, return 0
            return 0

        x = np.array(emg).reshape(1, -1)
        pred = self.model.predict(x)
        return int(pred[0])

        # run_gui该函数包括EMG的识别图形界面和姿势预测
        # Handle keypresses
        for ev in pygame.event.get():
            if ev.type == QUIT or (ev.type == KEYDOWN and ev.unicode == 'q'):
                raise KeyboardInterrupt()
            elif ev.type == KEYDOWN:
                if K_0 <= ev.key <= K_9:
                    # Labelling using row of numbers
                    hnd.recording = ev.key - K_0
                elif K_KP0 <= ev.key <= K_KP9:
                    # Labelling using Keypad
                    hnd.recording = ev.key - K_Kp0
                elif ev.unicode == 'r':
                    hnd.cl.read_data()
                elif ev.unicode == 'e':
                    print("Pressed e, erasing local data")
                    self.cls.delete_data()
            elif ev.type == KEYUP:
                if K_0 <= ev.key <= K_9 or K_KP0 <= ev.key <= K_KP9:
                    # Don't record incoming data
                    # 按键控制，修改recording=-1之后底层调用hnd的时候不会存储数据
                    hnd.recording = -1

        # Plotting
        scr.fill((0, 0, 0), (0, 0, w, h))
        r = self.history_cnt.most_common(1)[0][0]

        for i in range(10):
            x = 0
            y = 0 + 30 * i
            # Set the barplot color
            clr = self.cls.color if i == r else (255, 255, 255)

            txt = font.render('%5d' % (self.cls.Y == i).sum(),
                              True, (255, 255, 255))
            scr.blit(txt, (x + 20, y))

            txt = font.render('%d' % i, True, clr)
            scr.blit(txt, (x + 110, y))

            # Plot the barchart plot
            scr.fill((0, 0, 0), (x+130, y + txt.get_height() /
                     2 - 10, len(self.history) * 20, 20))
            scr.fill(clr, (x+130, y + txt.get_height() /
                     2 - 10, self.history_cnt[i] * 20, 20))

        pygame.display.flip()


# ------------ Myo Setup ---------------

# multiprocess.Queue是多进程通讯用的队列，因为不同进程之间不共享全局变量，用来存不同进程中的数据
q = multiprocessing.Queue()
# m = multiprocessing.Queue()#用来存储一帧需要预测数据


def on_press(key):
    print('{0} 按下'.format(key))
# 释放键盘时回调的函数


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
    # if ((pose == 0) and (TRAINING_MODE == False)):
    #     print("0,w")
    #     # Press and release space
    #     keyboard.press('w')
    #     time.sleep(0.1)  # 按下一秒
    #     keyboard.release('w')
    # if ((pose == 1) and (TRAINING_MODE == False)):
    #     print("1,s")
    #     # Press and release space
    #     keyboard.press('s')
    #     time.sleep(0.1)
    #     keyboard.release('s')


def My_worker(q,m,hnd,on_press,on_release,key_emg_handler):  # 获取EMG数据
    # m = Myo(mode=emg_mode.PREPROCESSED)#初始化Myo类，包含通讯协议，原始数据存储的数组
    #m = MyoClassifier(KNN_Classifier(), mode=emg_mode.PREPROCESSED, Networkmode=False)
    # m = MyoClassifier(Yzh_CNN_Model(), mode=emg_mode.RAW, Networkmode=True)
    # hnd = EMGHandler(m)  # EMGHandler中的emg成员直接就是存储了实时的emg数据，每次直接刷新
    # m.add_emg_handler(hnd)
    # 把EMGHandler加入到一个MyoClassifier类中的一个列表[]中，底层调用
    m.connect()  # 链接函数，包含调用了MyoClassifier中的成员BT类，把函数头参数参入底层
    m.add_raw_pose_handler(key_emg_handler)
    # MyoClassifier的处理手势的函数，里面的on_raw_pose函数调用传入的参数
    # on_pose函数里面有调用相关函数
    # m.connect()
    def My_Handler(emg, movement):
        q.put(emg)  # 用于打印数据的多进程队列
        hnd(emg, movement)  # 存储数据的处理函数，hnn.emg存储实时数据
    m.add_emg_handler(My_Handler)
    # 传入的之后我的函数在处理函数队列的第二位，
    # 第一位初始化的时候传入，记录和计算历史帧出现次数
    # 每一帧数据都是遍历执行两个函数
    def print_battery(bat):
        print("Battery level:", bat)

    m.add_battery_handler(print_battery)

    # Orange logo and bar LEDs
    m.set_leds([128, 0, 0], [128, 0, 0])
    # Vibrate to know we connected okay
    m.vibrate(1)
    # 开始按键监听
    # listener = Listener(on_press=on_press, on_release=on_release)
    # listener.start()
    """worker function"""
    while True:
        #print("1111111111111111")
        m.run()
    #     m.run_gui(hnd, scr, font, w, h)
        # time.sleep(0.1)
        # print(m.last_pose,":real time pose(before process)")
        # print(hnd.emg,":real time data")
        # run_gui该函数包括EMG的识别图形界面和姿势预测,包括按键处理,
        # 按下按键的时候实时记录数据
    # print("Worker Stopped")


def data_print(a, m):  # 打印EMG数据
    while True:
        while not(a.empty()):
            emg = list(a.get())
            # get一次是一帧数据是八个通道的数据[n1,n2,n3,n4,n5,n6,n7,n8]
            # print(emg)#这个是用异步进程打印数据
            if m.empty():
                m.put(emg)
        '''
		q.empty() ：如果调用此方法时 q为空，返回True。
		如果其他进程或线程正在往队列中添加项目，结果是不可靠的。
		也就是说，在返回和使用结果之间，队列中可能已经加入新的项目。
		'''
def feature_extraction(data):
    rms=featureRMS(data)
    mav=featureMAV(data)
    wl=featureWL(data)
    zc=featureZC(data)
    iemg=featureIEMG(data)
    wamp=feartureWAMP(data)
    var=feartureVAR(data)
    lodD=featureLogD(data)
    return rms,mav,wl,zc,iemg,wamp,var,lodD


if __name__ == "__main__":
    '''
    # 第一个子进程，跑EMG的run函数接受数据------------------------------------------------------------------------------------------------------
    # first = multiprocessing.Process(target=My_worker, args=(q,on_press,on_release,key_emg_handler))
    # first.daemon = True  # 设置守护选项，主进程运行完成之后, 那么子进程也将会自动结束
    # first.start()
    # ------------------------------------------------------------------------------------------------------------------------------------

    # 第二个子进程，打印接受到的数据------------------------------------------------------------------------------------------------------
    # second = multiprocessing.Process(target=data_print, args=(q,m))
    # second.daemon=True
    # second.start()
    # 多进程，打印接受到的数据------------------------------------------------------------------------------------------------------

    # pool=multiprocessing.Pool(3)#创建进程池防止使用需要，进程容量为3，按需更改
# for i in range(10):
    # 利⽤进程池同步执⾏work任务，进程池中的进程会等待上⼀个进程执行完任务后才能执⾏下⼀个进程
    # pool.apply(work, (i, ))
    # 使⽤异步⽅式执⾏work任务
    #pool.apply_async(work, (i, ))

    # My_worker(q,on_press,on_release,key_emg_handler)
    '''
    pygame.init()
    w, h = 800, 320
    scr = pygame.display.set_mode((w, h))
    font = pygame.font.Font(None, 30)
    m = MyoClassifier(Yzh_CNN_Model(), mode=emg_mode.RAW, Networkmode=True)
    hnd = EMGHandler(m)
    # Set pygame window name
    # pygame.display.set_caption(m.cls.name)

    t1 = multiprocessing.Process(target=My_worker, args=(q,m,hnd,on_press,on_release,key_emg_handler))
    t1.daemon = True
    t1.start()
    
    yzh = Yzh_CNN_Model()
    data=[]
    model_RF=joblib.load('/home/y/mycode/EMG_part/RF_test')
    try:
        while True:
            # m.run()
            if not(m.data_sav.empty()):
                data.append(m.data_sav.get())
                # print("main process data",data)
                while len(data) >= 200:
                    print("len of data:", len(data))
                    rms,mav,wl,zc,iemg,wamp,var,lodD=feature_extraction(np.array(data))#提取emg信号的特征
                    temp=np.array([rms,mav,wl,zc,iemg,wamp,var,lodD]).reshape(-1,64)
                    print(model_RF.predict(temp))
                    # print("process data",data_toprocess[0])
                    # print(np.array(data_toprocess))
                    y=yzh.classify(data)
                    print("pose:",y)
                    del data[0:50]
                    m.history_cnt[m.history[0]] -= 1
                    m.history_cnt[y] += 1
                # #新传进来的键值y加一
                    m.history.append(y)
                    r, n = m.history_cnt.most_common(1)[0]
                # counter.most_common(n)函数返回的是列表中出现次数排在前n位的元素，元组形式返回('值'，'出现次数')
                    if m.last_pose is None or (n > m.history_cnt[m.last_pose] + 5 and n > m.hist_len / 2):
	# 队列中的出现最多的判断姿态大于上一次判断的姿态的个数5个同时超过一半是这个姿态就刷新姿势
                        m.on_raw_pose(r)  # 打印pose
                        print("this is raw pose!")
                        m.last_pose = r
                    m.run_gui(hnd, scr, font, w, h)
                
            # m.run_gui(hnd, scr, font, w, h)

    except KeyboardInterrupt:
        pass
    finally:
        # m.disconnect()
        print()
        t1.kill()
        pygame.quit()


# Read data
# 数据集读取和处理-------------------------------------------------------------------------------------------
    # data=pd.read_csv("D:/bristolUNI/dissertation/code/EMG/datacollect/pyomyo-main/10gesture_1__test_emg.csv")#read data
    # print(data)
# #preprocess
    # data_np=data.to_numpy()#transform the data to numpy
    # data_X=data_np[...,0:8]#分离训练数据和手势结果
    # print(data_X[...,0:8])
    # data_Y=data_np[...,1]
    # Xtr, Xtest, Ytr, Ytest = train_test_split(data_X, data_Y, test_size = 0.2,random_state=25,shuffle=True)#split training data and testing data
    # 数据集读取和处理-------------------------------------------------------------------------------------------
