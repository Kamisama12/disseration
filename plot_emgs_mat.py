import multiprocessing
import queue
import numpy as np
import mpl_toolkits.mplot3d as plt3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.cm import get_cmap
from scipy.fftpack import fft,ifft
from pyomyo import Myo, emg_mode

print("Press ctrl+pause/break to stop")

# ------------ Myo Setup ---------------
q = multiprocessing.Queue()

def worker(q):
	m = Myo(mode=emg_mode.RAW)
	m.connect()
	
	def add_to_queue(emg, movement):
		q.put(emg)

	def print_battery(bat):
			print("Battery level:", bat)

	# Orange logo and bar LEDs
	m.set_leds([128, 0, 0], [128, 0, 0])
	# Vibrate to know we connected okay
	m.vibrate(1)
	m.add_battery_handler(print_battery)
	m.add_emg_handler(add_to_queue)

	"""worker function"""
	while True:
		try:
			m.run()
			# print(1)
		except:
			print("Worker Stopped")
			quit()

# ------------ Plot Setup ---------------
QUEUE_SIZE = 100
SENSORS = 8
subplots = []
lines = []
# Set the size of the plot
plt.rcParams["figure.figsize"] = (4,8)
# using the variable axs for multiple Axes
fig, subplots = plt.subplots(SENSORS, 1)

fig.canvas.manager.set_window_title("8 Channel EMG plot")
fig.tight_layout()
# Set each line to a different color

name = "tab10" # Change this if you have sensors > 10
cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
colors = cmap.colors  # type: list

for i in range(0,SENSORS):
	ch_line,  = subplots[i].plot(range(QUEUE_SIZE),[0]*(QUEUE_SIZE), color=colors[i])
	lines.append(ch_line)

emg_queue = queue.Queue(QUEUE_SIZE)


class KalmanFilter(object):
    def __init__(self, F=None, B=None, H=None, Q=None, R=None, P=None, x0=None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]  # F是状态转移矩阵，取他的行数初始化卡尔曼滤波的其他参数，适配矩阵运算
        # H理解成是在一个状态值下到观测值的转移矩阵，一般我们就默认是1，为了适配矩阵运算，初始化成eye(8)
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u=0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        # 在EMG下，F是8x8点成8x1，原始数据是1x8的，结果才能匹配成8x1的形状
        # 需要将转置之后的结果传进来
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        # P是我们每个状态数据的协方差矩阵，根据公式写出上面的代码预测下一个状态的协方差矩阵
        # 我们初始化参数的时候一般就会将P初始化成1，就是eye（8）在emg下面
        # Q是状态值的过程噪声的协方差矩阵，调参的时候可以调节这个值改变我们的卡尔曼增益结果从而看我们是更相信预测值还是更相信观测值
        return self.x.T  # 返回的时候需要变回1x8的数组方便我们观察

    def update(self, z):
        y = z - np.dot(self.H, self.x)  # y需要8x1
        print(np.dot(self.H, self.x))
        print("y",y)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        #print(S)
        # R是我们的观测噪声的协方差矩阵
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # K需要8x8
        #print(K)
        self.x = self.x + np.dot(K, y)  # x需要8x1
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P),
                        (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

from scipy.signal import butter, lfilter
#低通滤波器
def butter_lowpass(cutoff, fs, data,order=5):
	nyq = 0.5 * fs#nyquist frequency， 是采样频率的一半，等于最大的信号频率
	normal_cutoff = cutoff / nyq #3dB带宽点
	b, a = butter(order, normal_cutoff, btype='low', analog=False)#返回滤波器的系数,应该是一个传递函数的系数
	y=lfilter(b, a, data)
	return y


fatigue_list=[]
import time
def animate(i):
	# Myo Plot
	F = np.eye(8)
	H = np.eye(8)
    # np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
	Q = 0.05*np.eye(8)  # 状态噪声的协方差矩阵
	R = 0.5*np.eye(8)  # 观测噪声的协方差矩阵 # np.array([0.5]).reshape(1, 1)
	kf = KalmanFilter(F=F, H=H, Q=Q, R=R)

	
	global fatigue_list
	# Filter requirements.
	order = 1
	fs = 100.0       # sample rate, Hz
	cutoff = 1  # desired cutoff frequency of the filter, Hz
#按照我们EMG数据发送的频率是200hz ，定义200ms的window存储数据就是40个数据
#需要增加滑窗处理数据的代码
#不知道需不需要增加映射到0到255的代码
#数据训练的时候需要全部转换成列向量来做堆叠，所以我们传入神经网络训练的时候也要转成列向量
#为了减少延时，在real time的时候就处理一组数据，去除一帧，增加一帧。
	global t
	# rr3['Filtered'] = filtered_rr3
	while not(q.empty()):
		myox = list(q.get())
		#myox=100*myox
		# myox = np.array(q.get()).reshape(1,8)
		# print("emg数据：",myox)
		# print("emg.T",myox.T)
		# myox_kf=kf.predict()
		# kf.update(myox.T)
		filtered_rr3 = butter_lowpass(cutoff, fs, myox,order)
		fatigue_list.append(myox)
		# print("长度：",len(fatigue_list))
		if (time.time()-t)>=1:
			print("到达1s",len(fatigue_list))
			np_fatigue_list=np.array(fatigue_list)
			fft_data=fft(np_fatigue_list.T)
			for n in range(fft_data.shape[0]):
				fft_data[n]=np.mean(np.abs(fft_data[n]))
			print("the mean of the data in power spectrum domain:",np.abs(fft_data.T[0]))
			print("Sum the mean of the data in power spectrum domain:",np.sum(np.abs(fft_data.T[0])))
			print("Max mean of the data in power spectrum domain:",fft_data.T[0][np.argmax(np.abs(fft_data.T[0]))])
			print("median of the data in power spectrum domain:",np.median(np.abs(fft_data.T[0])))
			fatigue_list=[]
			t=time.time()
			
		# if len(fatigue_list)>=200:
		# 	fft_data=fft(fatigue_list)
		# 	fatigue_list=[]
		# 	print(fft_data.T.shape)
		# 	print("the mean of the data in power spectrum domain:",np.mean(fft_data[0]))
		# 	# plt.figure()
		# 	# x=np.linspace(0,200,200)
		# 	# plt.plot(x,fft_data.T[0]) 
		# 	# plt.title('1')
		# 	# plt.show()
		# 	# print(fft_data)
		# 	print("the median frequencies of the data in power spectrum domain:",np.median(fft_data))
		# print(filtered_rr3)
		# print("processing")
		if (emg_queue.full()):
			emg_queue.get()
		emg_queue.put(filtered_rr3)

	channels = np.array(emg_queue.queue)

	if (emg_queue.full()):
		for i in range(0,SENSORS):
			channel = channels[:,i]
			lines[i].set_ydata(channel)
			subplots[i].set_ylim(-10,max(10,max(channel)))




if __name__ == '__main__':
	# Start Myo Process
	p = multiprocessing.Process(target=worker, args=(q,))
	p.start()
	fatigue_list=[]
	t=time.time()
	# while True:
	while(q.empty()):
		# Wait until we actually get data
		print("waiting")
		continue
		# global fatigue_list
		
		# Filter requirements.
		# order = 1
		# fs = 100.0       # sample rate, Hz
		# cutoff = 1  # desired cutoff frequency of the filter, Hz
		# while not(q.empty()):
		# 	myox = list(q.get())
		# 	#myox=100*myox
		# 	# myox = np.array(q.get()).reshape(1,8)
		# 	# print("emg数据：",myox)
		# 	# print("emg.T",myox.T)
		# 	# myox_kf=kf.predict()
		# 	# print("1111111111111111111:",myox_kf[0])
		# 	# kf.update(myox.T)
		# 	filtered_rr3 = butter_lowpass(cutoff, fs, myox,order)
		# 	fatigue_list.append(myox)
		# 	print("长度：",len(fatigue_list))
		# 	if len(fatigue_list)>=200:
		# 		fft_data=fft(fatigue_list)
		# 		fatigue_list=[]
		# 		print(fft_data.T.shape)
		# 		print("the mean of the data in power spectrum domain:",np.mean(fft_data[0]))
		# 		x=np.linspace(0,200,200)
		# 		plt.subplot(111)
		# 		plt.plot(x,fft_data.T[0]) 
		# 		plt.title('1')
		# 		plt.show()
		# 		print(fft_data)
		# 		print("the median frequencies of the data in power spectrum domain:",np.median(fft_data))






	anim = animation.FuncAnimation(fig, animate, blit=False, interval=2)
	def on_close(event):
		p.terminate()
		raise KeyboardInterrupt
		print("On close has ran")
	fig.canvas.mpl_connect('close_event', on_close)

	try:
		plt.show()
	except KeyboardInterrupt:
		plt.close()
		p.close()
		quit()