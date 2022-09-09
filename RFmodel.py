


from cmath import pi
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
# from sklearn.externals import joblib 
import joblib
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.signal import butter,lfilter
import matplotlib.pyplot as plt

def butter_lowpass(cutoff, fs, data,order=5):
	nyq = 0.5 * fs#nyquist frequency， 是采样频率的一半，等于最大的信号频率
	normal_cutoff = cutoff / nyq #3dB带宽点
	b, a = butter(order, normal_cutoff, btype='low', analog=False)#返回滤波器的系数,应该是一个传递函数的系数
	y=lfilter(b, a, data)
	return y

#下面四个是时域（TD），增强型（TD）里面经典机器学习的特征方法
def featureRMS(data):#均方根植
    return np.sqrt(np.mean(data**2, axis=0))#axis=0沿着行计算平均值

def featureMAV(data):#平均绝对值
    return np.mean(np.abs(data), axis=0) 

def featureWL(data):#波形长度
    return np.sum(np.abs(np.diff(data, axis=0)),axis=0)/data.shape[0]

def featureZC(data, threshold=10e-7):#过零
    numOfZC = []
    channel = data.shape[1]
    length  = data.shape[0]
    
    for i in range(channel):
        count = 0
        for j in range(1,length):
            diff = data[j,i] - data[j-1,i]
            mult = data[j,i] * data[j-1,i]
            
            if np.abs(diff)>threshold and mult<0:
                count=count+1
        numOfZC.append(count/length)#计算每个通道的过零长度
    return np.array(numOfZC)

def featureSSC(data,threshold=10e-7):#斜率符号变化
    numOfSSC = []
    channel = data.shape[1]
    length  = data.shape[0]
    
    for i in range(channel):
        count = 0
        for j in range(2,length):
            diff1 = data[j,i]-data[j-1,i]
            diff2 = data[j-1,i]-data[j-2,i]
            sign  = diff1 * diff2
            
            if sign>0:
                if(np.abs(diff1)>threshold or np.abs(diff2)>threshold):
                    count=count+1
        numOfSSC.append(count/length)
    return np.array(numOfSSC)


def featureIEMG(data):
    return np.sum(np.abs(data),axis=0)

def feartureWAMP(data,threshold=10e-7):#衡量信号幅值的变化次数
    WAMP=[]
    channel = data.shape[1]
    length  = data.shape[0]
    for i in range(channel):
        count=0
        for j in range(1,length):
            diff=np.abs(data[j,i] - data[j-1,i])
            if diff >= threshold:
                threshold=diff
                count=count+1
        WAMP.append(count)
    return np.array(WAMP)

def feartureVAR(data):#variance
    #  return np.sum(np.square(data),axis=0)/(data.shape[0]-1)
    return np.sum(np.square(data),axis=0)

def featureLogD(data):#log dector ,provide an estimate of the muscle contraction force
    return np.exp(np.mean(np.log(np.abs(data)+10e-7),axis=0))

def feature_extraction(data):
    rms=featureRMS(data)
    mav=featureMAV(data)
    wl=featureWL(data)
    zc=featureZC(data)
    iemg=featureIEMG(data)
    wamp=feartureWAMP(data)
    var=feartureVAR(data)
    lodD=featureLogD(data)
    # print("shape of rms:",np.shape(rms))
    # print("shape of mav:",np.shape(mav))
    # print("shape of wl:",np.shape(wl))
    # print("shape of zc:",np.shape(zc))
    # print("shape of iemg:",np.shape(iemg))
    # print("shape of wamp:",np.shape(wamp))
    # print("shape of var:",np.shape(var))
    # print("shape of lodD:",np.shape(lodD))
    return rms,mav,wl,zc,iemg,wamp,var,lodD

def regre_data_process():
    '''
    X = []
    Y = []
    yzh_feature=[]
    yzh_feature_map=[]
    for i in range(10):
        X.append(np.fromfile('./EMG_part/datacollect/3nd_collect/data%d.dat' % i, dtype=np.float32).reshape((-1, 8)))
        Y.append(i + np.zeros(X[-1].shape[0]))#不知道数组长度，下标给-1。
        if len(X[i]) != 0:  #用滑窗的方式构建我们神经网络的输入数据集，覆盖率百分之75
            data_number=math.floor((len(X[i])-200)/50)+1#计算我们采用滑窗之后理论总共有多少个数据
            for d in range(data_number):
                rms,mav,wl,zc,iemg,wamp,var,lodD=feature_extraction(np.array(X[i][d*50:d*50+200]))
                temp=np.array([rms,mav,wl,zc,iemg,wamp,var,lodD])
                yzh_feature.append(temp)
                yzh_feature_map.append(np.sin(2*np.pi*200*(np.mean(yzh_feature[d]))))
                # print(yzh_image[d])
                # print(yzh_image[d].shape)
                # print(yzh_imgage_label[d])
                # assert(yzh_feature[d].shape[0]==200 and yzh_feature[d].shape[1]==8)#如何我们的数据形状有错误就引发错误


    yzh_regr_x=np.array(yzh_feature).reshape(-1,64)
    yzh_regr_y=np.array(yzh_feature_map)
    return yzh_regr_x,yzh_regr_y
    
    '''

    np_emg=np.array([])
    print(np_emg.shape)
    for i in range(1,5,1):
        data_emg=pd.read_csv('/home/y/mycode/EMG_part/regress data/zhao hao_emg/%d.csv'%i,header=None)
        if i==4:
            np_emg=np.append(np_emg,data_emg.to_numpy()[len(data_emg)-400:len(data_emg),:])
        else:
            np_emg=np.append(np_emg,data_emg.to_numpy()[len(data_emg)-550:len(data_emg),:])
        # print("this is data emg_to numpy:",np_emg)
        # print(np.shape(np_emg))
    np_emg=np_emg.reshape(-1,8)
    print(np_emg.shape)
    print(np_emg)
    data_emg_filter=butter_lowpass(1,100,np_emg,1)

    data_number=math.floor((len(data_emg_filter)-20)/1)+1
    emg_for_return=[]
    for d in range(data_number):
        emg_for_return.append(data_emg_filter[d*1:d*1+20])
        # print(emg_for_return)
        # print(np.shape(emg_for_return))
    np_emg_for_return=np.array(emg_for_return)
    print(np_emg_for_return.shape)


    # print("emg滤波数据：\n",data_emg_filter)
    np_robostate=np.array([])

    for i in range(1,5,1):
        data_robotstate=pd.read_csv('/home/y/mycode/EMG_part/regress data/Zhao hao/%d.csv'%i,header=None)
        # print(data_robotstate)
        # print(len(data_robotstate))
        if i==4:
            np_robostate=np.append(np_robostate,data_robotstate.to_numpy()[len(data_robotstate)-400:len(data_robotstate),3:6])
        else:
            np_robostate=np.append(np_robostate,data_robotstate.to_numpy()[len(data_robotstate)-550:len(data_robotstate),3:6])
        # print("机器人状态标签数据：\n",np_robostate)

        # print(np_robostate.shape)
    np_robostate=np_robostate.reshape(-1,3)
    print(np_robostate.shape)
    print(np_robostate)
    np_robostate_inc=np.diff(np_robostate,axis=0)[:len(np_robostate)-19]
    print(np_robostate_inc,"\n",np.shape(np_robostate_inc))
    
    return np_emg_for_return,np_robostate_inc
        
def regre_data_process_force_data():
    '''
    X = []
    Y = []
    yzh_feature=[]
    yzh_feature_map=[]
    for i in range(10):
        X.append(np.fromfile('./EMG_part/datacollect/3nd_collect/data%d.dat' % i, dtype=np.float32).reshape((-1, 8)))
        Y.append(i + np.zeros(X[-1].shape[0]))#不知道数组长度，下标给-1。
        if len(X[i]) != 0:  #用滑窗的方式构建我们神经网络的输入数据集，覆盖率百分之75
            data_number=math.floor((len(X[i])-200)/50)+1#计算我们采用滑窗之后理论总共有多少个数据
            for d in range(data_number):
                rms,mav,wl,zc,iemg,wamp,var,lodD=feature_extraction(np.array(X[i][d*50:d*50+200]))
                temp=np.array([rms,mav,wl,zc,iemg,wamp,var,lodD])
                yzh_feature.append(temp)
                yzh_feature_map.append(np.sin(2*np.pi*200*(np.mean(yzh_feature[d]))))
                # print(yzh_image[d])
                # print(yzh_image[d].shape)
                # print(yzh_imgage_label[d])
                # assert(yzh_feature[d].shape[0]==200 and yzh_feature[d].shape[1]==8)#如何我们的数据形状有错误就引发错误


    yzh_regr_x=np.array(yzh_feature).reshape(-1,64)
    yzh_regr_y=np.array(yzh_feature_map)
    return yzh_regr_x,yzh_regr_y
    
    '''

    np_emg=np.array([])
    print(np_emg.shape)
    for i in range(6,8,1):
        data_emg=pd.read_csv('/home/y/mycode/EMG_part/regress data/zhao hao_emg/%d.csv'%i,header=None)
        if i==4:
            np_emg=np.append(np_emg,data_emg.to_numpy()[len(data_emg)-400:len(data_emg),:])
        else:
            np_emg=np.append(np_emg,data_emg.to_numpy()[len(data_emg)-550:len(data_emg),:])
        # print("this is data emg_to numpy:",np_emg)
        # print(np.shape(np_emg))
    np_emg=np_emg.reshape(-1,8)
    print(np_emg.shape)
    print(np_emg)
    data_emg_filter=butter_lowpass(1,100,np_emg,1)

    data_number=math.floor((len(data_emg_filter)-20)/1)+1
    emg_for_return=[]
    for d in range(data_number):
        emg_for_return.append(data_emg_filter[d*1:d*1+20])
        # print(emg_for_return)
        # print(np.shape(emg_for_return))
    np_emg_for_return=np.array(emg_for_return)
    print(np_emg_for_return.shape)


    # print("emg滤波数据：\n",data_emg_filter)
    np_robostate=np.array([])

    for i in range(6,8,1):
        np_robostate=pd.read_csv('/home/y/mycode/EMG_part/regress data/Zhao hao/%d.csv'%i,header=None)
        # print(data_robotstate)
        # print(len(data_robotstate))
        if i==4:
            np_robostate=np.append(np_robostate,np_robostate_withF.to_numpy()[len(data_robotstate)-400:len(data_robotstate),[3,4,5,14,15,16]])
        else:
            np_robostate=np.append(np_robostate,np_robostate_withF.to_numpy()[len(data_robotstate)-550:len(data_robotstate),[3,4,5,14,15,16]])
        # print("机器人状态标签数据：\n",np_robostate)

        # print(np_robostate.shape)
    np_robostate_withF=np_robostate.reshape(-1,6)
    print(np_robostate_withF.shape)
    print(np_robostate_withF)
    np_robostate_inc=np.diff(np_robostate_withF,axis=0)[:len(np_robostate_withF)-19]
    print(np_robostate_inc,"\n",np.shape(np_robostate_inc))
    
    return np_emg_for_return,np_robostate_inc

if __name__ == '__main__':
    '''
sklearn.ensemble.RandomForestRegressor(
n_estimators=100, *, 				# 树的棵树，默认是100
criterion='mse', 					# 默认“ mse”，衡量质量的功能，可选择“mae”。
max_depth=None, 					# 树的最大深度。
min_samples_split=2, 				# 拆分内部节点所需的最少样本数：
min_samples_leaf=1, 				# 在叶节点处需要的最小样本数。
min_weight_fraction_leaf=0.0, 		# 在所有叶节点处的权重总和中的最小加权分数。
max_features='auto', 				# 寻找最佳分割时要考虑的特征数量。
max_leaf_nodes=None, 				# 以最佳优先方式生长具有max_leaf_nodes的树。
min_impurity_decrease=0.0, 			# 如果节点分裂会导致杂质的减少大于或等于该值，则该节点将被分裂。
min_impurity_split=None, 			# 提前停止树木生长的阈值。
bootstrap=True, 					# 建立树木时是否使用bootstrap抽样。 如果为False，则将整个数据集用于构建每棵决策树。
oob_score=False, 					# 是否使用out-of-bag样本估算未过滤的数据的R2。
n_jobs=None, 						# 并行运行的Job数目。
random_state=None, 					# 控制构建树时样本的随机抽样
verbose=0, 							# 在拟合和预测时控制详细程度。
warm_start=False, 					# 设置为True时，重复使用上一个解决方案，否则，只需拟合一个全新的森林。
ccp_alpha=0.0,
max_samples=None)					# 如果bootstrap为True，则从X抽取以训练每个决策树。
    '''
    # X,Y=regre_data_process()
    # print(X.shape,Y.shape)
    # print(len(X))
    #data=pd.read_csv('/home/y/mycode/EMG_part/regress data/EMG_force_regression_data/To_Zhaohao_24_08/my_emg.csv',header=None)#这个数据是和机械臂配合采集的
    data=pd.read_csv('/home/y/mycode/EMG_part/regress data/datatest_for_RF_force_without_robot_arm_2.csv',header=None)
    #data=data.to_numpy()[50:len(data),...]#配合采集时候的处理
    data=data.to_numpy()
    # print("data:",data)
    order = 1
    fs = 100.0       # sample rate, Hz
    cutoff = 1  # desired cutoff frequency of the filter, Hz
    data=butter_lowpass(cutoff, fs, data,order)
    data_number=math.floor((len(data)-200)/1)+1#计算我们采用滑窗之后理论总共有多少个数据
    yzh_image=[]
    for d in range(data_number):
            # print("test data shape:",X[i][0:12])
            # print(X[i][0:12].shape)
        yzh_image.append(data[d*1:d*1+200])
    yzh_image=np.array(yzh_image)
    print("yzh_image.shape:",yzh_image.shape)
    feature=np.array([]).reshape(-1,1)
    for x in range(len(yzh_image)):
        rms,mav,wl,zc,iemg,wamp,var,lodD=feature_extraction(yzh_image[x])
        # print([rms,mav,wl,zc,iemg,wamp,var,lodD])
        feature=np.append(feature,[rms,mav,wl,zc,iemg,wamp,var,lodD])
    print(feature.shape)
    ttt=feature.reshape(-1,8,8)
    print(ttt[...,0].shape)
    plt.figure()
    #plt.plot(np.linspace(0,1000,len(ttt)),np.sum(ttt[...,0],axis=1))
    plt.plot(np.linspace(0,1000,len(ttt)),ttt[...,0,0].ravel())
    # plt.show()
    # feature=np.array([])
    # print(X[0],X[0].shape)
    # for i in range(len(X)):
    #     # print("X[i]：",X[i])
    #     # print(np.shape(X[i]))
    #     rms,mav,wl,zc,iemg,wamp,var,lodD=feature_extraction(X[i])
    #     # print(rms,mav,wl,zc,iemg,wamp,var,lodD)
    #     # print(rms,"\n",mav,"\n",wl,"\n",zc,"\n",iemg,"\n",wamp,"\n",var,"\n",lodD)
    #     feature=np.append(feature,[rms,mav,wl,zc,iemg,wamp,var,lodD])
        
    X=feature.reshape(-1,64)

    data_force=pd.read_csv('/home/y/mycode/EMG_part/regress data/EMG_force_regression_data/To_Zhaohao_24_08/robot_state.csv',header=None)
    data_force=data_force.to_numpy()[...,16]#读取力数据训练
    # data_number_force=math.floor((len(data_force)-200)/1)+1#计算我们采用滑窗之后理论总共有多少个数据
    # yzh_image_force=[]
    # for d in range(data_number_force):
    #         # print("test data shape:",X[i][0:12])
    #         # print(X[i][0:12].shape)
    #     yzh_image_force.append(np.mean(data_force[d*1:d*1+200],axis=0))
    # yzh_image_force=np.array(yzh_image_force)


    # Y=data_force[len(data_force)-len(yzh_image):len(data_force),...]
    '''
    #这里用的sin函数拟合。
    # x=np.linspace(0,data_number,data_number)
    # Y=-1*np.abs(np.sin((3*np.pi/data_number)*x))
    
    '''
    #这里用线性函数拟合
    Y=np.linspace(0,-20,data_number)


    print(Y.shape)
    plt.figure()
    plt.plot(np.linspace(0,1000,len(Y)),Y)
    plt.show()
    print(feature,feature.shape)
    regr=RandomForestRegressor(n_estimators=200,random_state=20,verbose=0)
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
    
    print(X_train,Y_train)
    print(np.shape(X_train),np.shape(Y_train))
    print("Start")
    regr.fit(X_train,Y_train)
    y_pred = regr.predict(X_test)
    print(np.shape(y_pred))
    print("预测：",y_pred[0])
 
    #GMM
    
    # from sklearn.mixture import GaussianMixture
    # X = np.array([[1,2],[2,3],[3,4],[3,5]])
    # gm = GaussianMixture(n_components=3, random_state=0).fit(X)
    # print(gm.means_)
    # print(gm.predict([[0,10]]))
    
    
    #SVR
    



    
    

    joblib.dump(regr,'/home/y/mycode/EMG_part/regression_trained_model/RF_train_force_8-24')
    # model=joblib.load('/home/y/mycode/EMG_part/regression_trained_model/RF_train_1_4_without_F')

    # output=model.predict(X)
    # print(output)
    # # 评估回归性能
    from sklearn import metrics
    metrics.recall_score
    print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
    print('Root Mean Squared Error:',
        np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
