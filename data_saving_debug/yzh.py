from scipy.signal import butter, lfilter
import multiprocessing 
from pyomyo import Myo,emg_mode
from pyomyo.Classifier import EMGHandler
import csv
import time
import pygame
import numpy as np



#低通滤波器
def butter_lowpass(cutoff, fs, data,order=5):
	nyq = 0.5 * fs#nyquist frequency， 是采样频率的一半，等于最大的信号频率
	normal_cutoff = cutoff / nyq #3dB带宽点
	b, a = butter(order, normal_cutoff, btype='low', analog=False)#返回滤波器的系数,应该是一个传递函数的系数
	y=lfilter(b, a, data)
	return y

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

last_vals=None
def plot(scr, vals,w,h):
    DRAW_LINES = True

    global last_vals
    if last_vals is None:
        last_vals = vals
        return

    D = 5
    scr.scroll(-D)
    scr.fill((0, 0, 0), (w - D, 0, w, h))
    for i, (u, v) in enumerate(zip(last_vals, vals)):
        if DRAW_LINES:
            pygame.draw.line(scr, (0, 255, 0),
                             (w - D, int(h/9 * (i+1 - u))),
                             (w, int(h/9 * (i+1 - v))))
            pygame.draw.line(scr, (255, 255, 255),
                             (w - D, int(h/9 * (i+1))),
                             (w, int(h/9 * (i+1))))
        else:
            c = int(255 * max(0, min(1, v)))
            scr.fill((c, c, c), (w - D, i * h / 8,
                     D, (i + 1) * h / 8 - i * h / 8))

    pygame.display.flip()
    last_vals = vals


def data_saving(q,fre_in_sec,filepath):
    m = Myo(mode=emg_mode.RAW)
    hnd=EMGHandler(m)
    start_time=time.time()
    flag=False
    w, h = 800, 600
    scr = pygame.display.set_mode((w, h))
    try:
        while True:
            pygame.event.pump()
            if not (q.empty()):
                hnd.emg=list(q.get())
                plot(scr, [e / 500. for e in hnd.emg],w,h)
                flag=True
            if time.time()-start_time>=fre_in_sec and flag:
                data = butter_lowpass(1, 100, hnd.emg,1)
                f=open(filepath,'a')#添加模式或者重建模式
                writer=csv.writer(f)
                print("是否已经关闭：",f.closed)
                print("saving....")
                print(data)
                writer.writerow(data)
                f.close()
                print("是否已经关闭：",f.closed)
                flag=False
                start_time=time.time()
                

    except Exception as e: 
        print(e)
        print("datasaving stop")
        quit()



if __name__ =="__main__":
    first = multiprocessing.Process(target=worker, args=(q,))
    first.daemon=True
    first.start()
    data_saving(q,0.1,'/home/y/datatest.csv')
