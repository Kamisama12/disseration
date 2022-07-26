'''
The MIT License (MIT)
Copyright (c) 2020 PerlinWarp
Copyright (c) 2014 Danny Zhu

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

from collections import Counter, deque
import struct
import sys
import time

import pygame
from pygame.locals import *
import numpy as np

from pyomyo import Myo, emg_mode

import torch

SUBSAMPLE = 3
K = 15

class Classifier(object):
	'''A wrapper for nearest-neighbor classifier that stores
	training data in vals0, ..., vals9.dat.'''

	def __init__(self, name="Classifier", color=(0,200,0)):
		# Add some identifiers to the classifier to identify what model was used in different screenshots
		self.name = name
		self.color = color

		for i in range(10):
			with open('D://bristolUNI//dissertation//code//EMG//datacollect//2nd_collect//data%d.dat' % i, 'ab') as f: pass
		self.read_data()

	def store_data(self, cls, vals):
		#cls传入的是EMGhandler的类
		with open('D://bristolUNI//dissertation//code//EMG//datacollect//2nd_collect//data%d.dat' % cls, 'ab') as f:
			f.write(pack('8H', *vals))#pack表示按照给定的格式(fmt)，把数据封装成字符串，采用小段存储
		#采集数据，先不训练
		self.Y=np.hstack([self.Y, [cls]])
		self.X=np.vstack([self.X, vals])
		#self.train(np.vstack([self.X, vals]), np.hstack([self.Y, [cls]]))
		#在原来的数据上继续堆叠新的数据
	def read_data(self):
		X = []
		Y = []
		for i in range(10):
			X.append(np.fromfile('D://bristolUNI//dissertation//code//EMG//datacollect//2nd_collect//data%d.dat' % i, dtype=np.uint16).reshape((-1, 8)))
			Y.append(i + np.zeros(X[-1].shape[0]))
			#按照读取顺序，读取数据的同时加上姿态的标签
		self.train(np.vstack(X), np.hstack(Y))
		#vstack竖直方向堆叠
	def delete_data(self):
		for i in range(10):
			with open('D://bristolUNI//dissertation//code//EMG//datacollect//2nd_collect//data%d.dat' % i, 'wb') as f: pass
		self.read_data()

	def train(self, X, Y):
		self.X = X
		self.Y = Y
		self.model = None

	def nearest(self, d):
		dists = ((self.X - d)**2).sum(1)
		ind = dists.argmin()#np的函数，返回最小值的序号
		return self.Y[ind]

	def classify(self, d):
		if self.X.shape[0] < K * SUBSAMPLE: return 0
		return self.nearest(d)

class MyoClassifier(Myo):
	'''Adds higher-level pose classification and handling onto Myo.'''

	def __init__(self, cls, tty=None, mode=emg_mode.PREPROCESSED, hist_len=25,Networkmode=False):#源码中的hist_len长度是25，实验的CNN神经网络模型是用的12帧作为一张图片，改成12
		Myo.__init__(self, tty, mode=mode)
		# Add a classifier
		self.cls = cls
		self.hist_len = hist_len#25帧
		self.history = deque([0] * self.hist_len, self.hist_len)
		#设定一个队列，最大长度25，初始是25个0
		self.history_cnt = Counter(self.history)
		#数队列初始化状态，返回字典类型
		self.add_emg_handler(self.emg_handler)
		self.last_pose = None
		self.yuzhaohao_datacount=[]
		self.pose_handlers = []
		self.networkmode=Networkmode

	def emg_handler(self, emg, moving):#这个函数用于分类和对比历史帧的次数
		#y = self.cls.classify(emg)#这里的数据应该是只有一帧，用 CNN网络的时候需要有12帧
		self.yuzhaohao_datacount.append(emg)
		if self.networkmode==True:
			if len(self.yuzhaohao_datacount) is 12:#存够12个数据之后当作一帧图像用于传入神经网络
				y=self.cls.classify(self.yuzhaohao_datacount)
				self.yuzhaohao_datacount=[]#清空我们的缓存数据
				self.history_cnt[self.history[0]] -= 1
				#最左端出队列，计数器中最左端的键的计数减一
				self.history_cnt[y] += 1
				#新传进来的键值y加一
				self.history.append(y)

				r, n = self.history_cnt.most_common(1)[0]
				#counter.most_common函数返回的是列表中出现最多次数的元素，元组形式返回('值'，'出现次数')
				if self.last_pose is None or (n > self.history_cnt[self.last_pose] + 5 and n > self.hist_len / 2):
					#队列中的出现最多的判断姿态大于上一次判断的姿态的个数5个同时超过一半是这个姿态就刷新姿势
					self.on_raw_pose(r)#打印pose
					print("this is raw pose!")
					self.last_pose = r
		else:
			y=self.cls.classify(emg)
			self.history_cnt[self.history[0]] -= 1
			self.history_cnt[y] += 1
			self.history.append(y)
			r, n = self.history_cnt.most_common(1)[0]
			if self.last_pose is None or (n > self.history_cnt[self.last_pose] + 5 and n > self.hist_len / 2):
				self.on_raw_pose(r)#打印pose
				print("this is raw pose!")
				self.last_pose = r


	def add_raw_pose_handler(self, h):
		self.pose_handlers.append(h)

	def on_raw_pose(self, pose):#打印pose
		for h in self.pose_handlers:#传进来的是print
			h(pose)

	def run_gui(self, hnd, scr, font, w, h):#传入的参数是hnd-EMGhandler类
		
		#run_gui该函数包括EMG的识别图形界面和姿势预测
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
					#按键控制，修改recording=-1之后底层调用hnd的时候不会存储数据
					hnd.recording = -1

		# Plotting
		scr.fill((0, 0, 0), (0, 0, w, h))
		r = self.history_cnt.most_common(1)[0][0]

		for i in range(10):
			x = 0
			y = 0 + 30 * i
			# Set the barplot color
			clr = self.cls.color if i == r else (255,255,255)

			txt = font.render('%5d' % (self.cls.Y == i).sum(), True, (255,255,255))
			scr.blit(txt, (x + 20, y))

			txt = font.render('%d' % i, True, clr)
			scr.blit(txt, (x + 110, y))

			# Plot the barchart plot
			scr.fill((0,0,0), (x+130, y + txt.get_height() / 2 - 10, len(self.history) * 20, 20))
			scr.fill(clr, (x+130, y + txt.get_height() / 2 - 10, self.history_cnt[i] * 20, 20))

		pygame.display.flip()

def pack(fmt, *args):
	return struct.pack('<' + fmt, *args)#<表示采用小端存储

def unpack(fmt, *args):
	return struct.unpack('<' + fmt, *args)

def text(scr, font, txt, pos, clr=(255,255,255)):
	scr.blit(font.render(txt, True, clr), pos)


class EMGHandler(object):
	def __init__(self, m):
		self.recording = -1#Live_classifier.recording
		self.m = m
		self.emg = (0,) * 8#扩充元组到八个元素
		#元组中只包含一个元素时，需要在元素后面添加逗号 , ，否则括号会被当作运算符使用：

	def __call__(self, emg, moving):
		self.emg = emg
		if self.recording >= 0:
			self.m.cls.store_data(self.recording, emg)

class Live_Classifier(Classifier):#继承于Classifier
	'''
	General class for all Sklearn classifiers
	Expects something you can call .fit and .predict on
	'''
	def __init__(self, classifier, name="Live Classifier", color=(0,55,175)):
		self.model = classifier
		Classifier.__init__(self, name=name, color=color)

	def train(self, X, Y):
		self.X = X
		self.Y = Y

		if self.X.shape[0] > 0 and self.Y.shape[0] > 0: 
			self.model.fit(self.X, self.Y)

	def classify(self, emg):
		if self.X.shape[0] == 0 or self.model == None:
			# We have no data or model, return 0
			return 0

		x = np.array(emg).reshape(1,-1)
		pred = self.model.predict(x)
		return int(pred[0])

if __name__ == '__main__':
	pygame.init()
	w, h = 800, 320
	scr = pygame.display.set_mode((w, h))
	font = pygame.font.Font(None, 30)

	m = MyoClassifier(Classifier())
	hnd = EMGHandler(m)
	m.add_emg_handler(hnd)
	m.connect()

	m.add_raw_pose_handler(print)

	# Set Myo LED color to model color
	m.set_leds(m.cls.color, m.cls.color)
	# Set pygame window name
	pygame.display.set_caption(m.cls.name)

	try:
		while True:
			m.run()
			m.run_gui(hnd, scr, font, w, h)

	except KeyboardInterrupt:
		pass
	finally:
		m.disconnect()
		print()
		pygame.quit()
