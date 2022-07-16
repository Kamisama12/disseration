
import imp
from unicodedata import name
import pygame
from pygame.locals import *
import numpy as np
from collections import Counter, deque
import csv
import struct
import pandas as pd
title = ["姓名","性别","分数"]
# def pack(fmt, *args):
#     return  struct.pack('<' + fmt, *args)#<表示采用小端存储
def pack(fmt, *args):
    return  struct.pack('<' + fmt, *args)
#with open('D:/bristolUNI/dissertation/code/OpenBot-0.6.0/controller/python/pyomyo-main/src/pyomyo/data%d.dat' % cls, 'ab') as f:
with open('D:/bristolUNI/dissertation/code/EMG/datacollect/dataset/data_test' , 'ab') as f:
    pass
#f.write(pack('8H',1))
'''
%为数据占位符，‘ab’表示打开mode为'a'：以追加模式打开。若文件存在，则会追加到文件的末尾；若文件不存在，则新建文件。该模式不能使用read*()方法
'b'：以二进制模式打开 8H表示8个unsigned short数据
'''

data=np.fromfile('D:/bristolUNI/dissertation/code/EMG/datacollect/dataset/data_test' , dtype=np.uint16).reshape((-1, 8))#表示转化成八列的矩阵，行数未知直接取-1
print(data)


a=np.array([[1,2,3]])
#a[0]=np.column_stack((a[0],[1]))
b=np.linspace(1, 1, num=50, endpoint=True, retstep=False, dtype=None)
# a[0]=np.hstack((a[0],[1]))
print(b)
print(a[0])
print(len(a))