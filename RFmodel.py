from itertools import count
from tabnanny import verbose
from unicodedata import name
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib 
import math

from sklearn.model_selection import train_test_split


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
     return np.sum(np.square(data),axis=0)/(data.shape[0]-1)

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
    
    regr=RandomForestRegressor(n_estimators=200,random_state=20,verbose=0)
    X,Y=regre_data_process()
    print(np.shape(X))
    print(np.shape(Y))
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
    # print(X_train,Y_train)
    print(np.shape(X_train),np.shape(Y_train))
    print("Start")
    regr.fit(X_train,Y_train)
    y_pred = regr.predict(X_test)




    # joblib.dump(regr,'/home/y/mycode/EMG_part/RF_test')
    # model=joblib.load('/home/y/mycode/EMG_part/RF_test')

    # output=model.predict([[5,5,3],[2,2,1]])
    # print(output)
    # 评估回归性能
    from sklearn import metrics

    print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
    print('Root Mean Squared Error:',
        np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
