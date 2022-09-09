from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import numpy as np
import matplotlib.pyplot as plt;            # doctest: +SKIP
import scipy.interpolate as si
import pandas as pd
# from sklearn.externals import joblib 
import joblib 
import fastdtw
from scipy.spatial.distance import euclidean

def plot1(t_predict,y_mean,y_std,i=0):
    plt.figure()
    plt.plot(t_predict,y_mean[:],color='darkblue')
    plt.fill_between(t_predict.ravel(),
    y_mean.ravel() - 1.96 * y_std.ravel(),
    y_mean.ravel() + 1.96 * y_std.ravel(),
    alpha=0.7,
    color='cornflowerblue'
    )
    plt.fill_between(t_predict.ravel(),
    y_mean.ravel() - 3 * y_std.ravel(),
    y_mean.ravel() + 3 * y_std.ravel(),
    alpha=0.3,
    color='cornflowerblue'
    )

def get_via_point_sigma_function(sigma_max,x,sigma):
    #给定最大的方差，以及每个时间点的方差，得到二次差值平滑的方差函数曲线
    rows = x.shape[0]
    for i in range(rows):
            if i==rows-1:
                    break
            x=np.append(x,(x[i]+x[i+1])/2)
            sigma=np.append(sigma,sigma_max)
    quadratic = si.interp1d(x, sigma, kind="quadratic")#时间维度的插值
    return quadratic   

#得到人工干预的轨迹t，y，和方差，这部分的话，又臭又长，大概就是，要把干预线段的GP模型给建出来 
def get_interactive(end_std=0.001,mid_std=10e-8,left_time = 1,right_time = 2,pre_std=5):
    human_wantted_tra=pd.read_csv('/home/y/下载/robot_human_update.csv',header=None)
    human_wantted_tra=human_wantted_tra.to_numpy()
    time=human_wantted_tra[...,0]
    tra_xyz=human_wantted_tra[...,4:7]
    #改变起点和终点的情况下进行干预的拟合

    # t=np.array((0))     #将起点先加进去
    # sigma_interact=np.array([end_std]) #起点和终点的方差
    # y=np.array((0.55,0.9,0.5))   #起点的y设置为轨迹的起点y
    #给定中间的干预线段
    # for i in range((right_time-left_time)*100):
    #     t = np.append(t,1+0.01*i)
    #     y = np.append(y,[0.55+0.0005*i,0.55+0.0005*i,0.55+0.0005*i])
    #     sigma_interact = np.append(sigma_interact,mid_std)
    #YZH修改干预代码
    sigma_interact=np.array([end_std])
    for i in range(len(tra_xyz)-1):
        sigma_interact = np.append(sigma_interact,mid_std)
    #将终点加进去
    # t = np.append(t,4)
    # y = np.append(y,[0.497725779242184,-0.077063355529191,0.270777250284063])
    # sigma_interact = np.append(sigma_interact,0.1)
    # t = t.reshape(-1,1)
    # y = y.reshape(-1,3)
    sigma_interact = sigma_interact.reshape(-1,1)#这个interact_sigma是用来传入GPR中的alpha
    time=time.reshape(-1,1)
    tra_xyz=tra_xyz.reshape(-1,3)
    #下面这部分是要得到除了干预位置其他位置的方差
    t_predict1 = np.linspace(0,left_time,100).reshape(-1,1)
    t_predict2 = np.linspace(right_time,4,200).reshape(-1,1)
    quadratic1 = get_via_point_sigma_function(30,np.array((0,1)),np.array((pre_std,pre_std)))
    quadratic2 = get_via_point_sigma_function(30,np.array((2,4)),np.array((pre_std,pre_std)))
    # quadratic1 = 2
    # quadratic2 = 2
    sigma_all = quadratic1(t_predict1)#0到1
    for i in range((right_time-left_time)*100):#1到2
        sigma_all = np.append(sigma_all,mid_std)
    sigma_all = np.append(sigma_all,quadratic2(t_predict2))#2到4
    return (time,tra_xyz,sigma_interact,sigma_all)


def DTW(datalist1,datalist2):
    
    if len(datalist1)>=len(datalist2):#用数据长度长的做横轴
        # data=np.zeros((len(datalist2),len(datalist1)))
        data=datalist1
        distance, path=fastdtw.fastdtw(datalist1,datalist2)
    else:
        data=datalist2
        distance, path=fastdtw.fastdtw(datalist2,datalist1)
    data_new=np.array([])
    for i in range(len(path)):
        # print(data[path[i][1]])
        data_new=np.append(data_new,np.array(data[path[i][1]]))
    return data_new
    # else:
    #     data=np.zeros((len(datalist1),len(datalist2)))
    #     data[len(datalist1)-1]=datalist2#横轴
    #     data[...,0]=datalist1[::-1]#纵轴
    #     distance, path=fastdtw.fastdtw(datalist2,datalist1)
    #     data_new=np.array([])
    #     for i in range(len(path)):
    #         data_new=np.append(data_new,data[len(datalist1)-1][path[i][0]])
    #     return data_new




def get_interactive_model_f_l_point(end_time=35,start_point=[0.525,-0.1,0.3],end_point=[0.5225,-0.2,0.3],std=10e-10):
    t=np.array([0,end_time]).reshape(-1,1)#加入起点终点的时间
    tra_xyz_point=np.array([start_point,end_point]).reshape(-1,3)#加入起点终点
    sigma_interact_f_l=np.array([std,std])#加入起点终点方差
    kernel = 1.0 * RBF(length_scale=1e0, length_scale_bounds=(1e-8, 1e3))
    gpr_point = GaussianProcessRegressor(kernel=kernel,random_state=0,n_restarts_optimizer=5,alpha=sigma_interact_f_l.ravel())
    gpr_point.fit(t,tra_xyz_point)
    quadratic=get_via_point_sigma_function(0.1,np.array((0,35)),np.array((0.01,0.01)))

    return gpr_point,quadratic


def GP_mix_predict(GP_orginal,GP_huamn,time):
    y_mean1,std1=GP_orginal.predict(time,return_std=True)   
    y_mean2,std2=GP_huamn.predict(time,return_std=True)   
    std1 += 1e-8
    std2 += 1e-8
    if std1-std2>=1e-3:
        std2+=0.05
    elif std2-std1>=1e-3:
        std1+=0.05
    blend_std = 1/(1/(std2)+1/(std2))
    
    blend_mean = blend_std*(1/(std2)*y_mean1 + 1/(std2)*y_mean2 )
    return blend_mean


#初始部分GPR训练
# t=np.linspace(0,4,1000).reshape(-1,1)
# y=np.linspace(100,1000,1000).reshape(-1,1)+np.random.normal(0,0.02,size=t.shape)

'''YZH
np_robostate=np.array([]).reshape(-1,3)
for i in range(1,5,1):
    data_robotstate=pd.read_csv('/home/y/mycode/EMG_part/regress data/Zhao hao/%d.csv'%i,header=None)
    # print(data_robotstate.to_numpy())
    # print(data_robotstate.to_numpy().shape)
    # print(len(data_robotstate))
    if i==4:
        np_robostate=np.append(np_robostate,data_robotstate.to_numpy()[len(data_robotstate)-400:len(data_robotstate),3:6],axis=0)
    else:
        np_robostate=np.append(np_robostate,data_robotstate.to_numpy()[len(data_robotstate)-550:len(data_robotstate),3:6],axis=0)
        print(np_robostate.shape)

'''
def get_data():

    t=np.array([])
    data_robotstate=pd.read_csv('/home/y/mycode/EMG_part/regress data/28_GP_data_multiple_regre/28-20220828T144633Z-001/28/robot_state_for_GP_date27.csv',header=None)
    # data1=pd.read_csv('/home/y/mycode/EMG_part/regress data/28_GP_data_multiple_regre/28-20220828T144633Z-001/28/robot_state_1.csv',header=None).to_numpy()[...,3:6]
    # data2=pd.read_csv('/home/y/mycode/EMG_part/regress data/28_GP_data_multiple_regre/28-20220828T144633Z-001/28/robot_state_2.csv',header=None).to_numpy()[...,3:6]
    # data3=pd.read_csv('/home/y/mycode/EMG_part/regress data/28_GP_data_multiple_regre/28-20220828T144633Z-001/28/robot_state_3.csv',header=None).to_numpy()[...,3:6]
    # data4=pd.read_csv('/home/y/mycode/EMG_part/regress data/28_GP_data_multiple_regre/28-20220828T144633Z-001/28/robot_state_4.csv',header=None).to_numpy()[...,3:6]
    # data5=pd.read_csv('/home/y/mycode/EMG_part/regress data/28_GP_data_multiple_regre/28-20220828T144633Z-001/28/robot_state_5.csv',header=None).to_numpy()[...,3:6]
    # data6=pd.read_csv('/home/y/mycode/EMG_part/regress data/28_GP_data_multiple_regre/28-20220828T144633Z-001/28/robot_state_6.csv',header=None).to_numpy()[...,3:6]
    # state=DTW(data1,data2)
    # t=np.append(t,np.linspace(0,0.1*state.shape[0]/3,int(state.shape[0]/3))).reshape(-1,1)

    # state=np.append(state,DTW(data3,data4))
    # t=np.append(t,np.linspace(0,0.1*len(DTW(data3,data4))/3,int(len(DTW(data3,data4))/3) )).reshape(-1,1)


    # state=np.append(state,DTW(data5,data6))
    # t=np.append(t,np.linspace(0,0.1*len(DTW(data5,data6))/3,int(len(DTW(data5,data6))/3)  )).reshape(-1,1)
    # data_robot=state.reshape(-1,3)

    
    # data_robot=data_robot.reshape(-1,3)
    # print(data_robot.shape)

    np_robostate=data_robotstate.to_numpy()[...,3:6]
    # print(np_robostate.shape)
    y=np_robostate.reshape(-1,3)
    t=np.linspace(0,0.1*y.shape[0],y.shape[0]).reshape(-1,1)
    return np_robostate,t



y,t=get_data()
kernel1 = 1.0 * RBF(length_scale=1e0, length_scale_bounds=(1e-8, 1e3)) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-10, 1e10))
#RBF生成核函数，用作计算协方差矩阵
gpr1 = GaussianProcessRegressor(kernel=kernel1,random_state=0,n_restarts_optimizer=2,normalize_y=True)
gpr1.fit(t,y)
t_predict = np.linspace(0,0.1*y.shape[0],400).reshape(-1,1)
y_mean1,y_std1 = gpr1.predict(t_predict.reshape(-1,1),return_std=True)
# y_std1=y_std1+0.009
# joblib.dump(gpr1,'/home/y/mycode/EMG_part/regression_trained_model/GP_original_traj_date_28_train_dtw')



#干预部分GPR训练
t_,y_,sigma_int,sigma_all = get_interactive()
sigma_all=np.expand_dims(sigma_all,axis=1)
sigma_all=np.repeat(sigma_all,3,1)#为了和最后的方差运算维度对齐，需要把我们的一维方差扩充到三维
print(t_.shape,y_.shape,sigma_all.shape)
kernel2 = 1.0 * RBF(length_scale=1e0, length_scale_bounds=(1e-8, 1e3))
gpr2 = GaussianProcessRegressor(kernel=kernel2,random_state=0,n_restarts_optimizer=5,alpha=sigma_int.ravel())
#alpha函数是填进核函数的对角线中的数据
gpr2.fit(t_,y_)
# joblib.dump(gpr2,'/home/y/mycode/EMG_part/regression_trained_model/GP_original_traj_date_27_train_human')
# gpr2=joblib.load('/home/y/mycode/EMG_part/regression_trained_model/GP_original_traj_date_27_train_human')
t_predict_1 = np.linspace(0,0.1*y.shape[0],400).reshape(-1,1)
y_mean2,y_std2 = gpr2.predict(t_predict_1,return_std=True)
# print(y_std2[1500:1600])
# y_std2 +=sigma_all

gpr3,quadratic=get_interactive_model_f_l_point()
y3,std3=gpr3.predict(t_predict,return_std=True)
sig_yzh=quadratic(t_predict)
std3=std3+sig_yzh
# x=[1,2,3,4,4,4,45,6,7]
# y=[1,3,4,5,6,4,5,7]
# newdata=DTW(x,y)
# print(newdata)



# joblib.dump(gpr1,'/home/y/mycode/EMG_part/regression_trained_model/GPR_train')
# model=joblib.load('/home/y/mycode/EMG_part/regression_trained_model/GPR_train')

#画图
# plot1(t,y[...,0],np.array([0.001]))
plot1(t_predict,y_mean1[...,2],y_std1[...,2])
# plot1(t_predict,y3[...,1],std3[...,1])
plot1(t_predict,y_mean2[...,2],y_std2[...,2])
# plot1(t,y[...,1],np.array([0.001]))
# plot1(t_predict,y_mean1[...,1],y_std1[...,1])
# # plot1(t_predict,y_mean2[...,1],y_std2[...,1])
# # plot1(t,y[...,2],np.array([0.001]))
# plot1(t_predict,y_mean1[...,2],y_std1[...,2])
# plot1(t_predict,y_mean2[...,2],y_std2[...,2])

y_std1 += 1e-8
y_std2 += 1e-8

# for i in range(len(t_predict)):
#     if y_std1[i][2]/y_std2[i][2]<=1e-2:
#         y_std1[i][2]=y_std1[i][2]+0.01
#     elif y_std2[i][2]/y_std1[i][2]<=1e-2:
#         y_std2[i][2]=y_std2[i][2]+0.01

blend_std = 1/(1/(y_std2)+1/(y_std1))
# y_mean1 = y_mean1.ravel()
# y_mean2 = y_mean2.ravel()

blend_mean = blend_std*(1/(y_std1)*y_mean1 + 1/(y_std2)*y_mean2 )
# plot1(t_predict_1,blend_mean[...,1],blend_std[...,1])
plot1(t_predict,blend_mean[...,2],blend_std[...,2])
# plot1(t_predict,blend_mean[...,2],blend_std[...,2])

plt.show()