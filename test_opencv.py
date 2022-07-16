from logging import exception
from signal import raise_signal
import cv2


cap = cv2.VideoCapture(0)
cap_usb=cv2.VideoCapture(2)
#传入参数参数2的时候可以调用USB摄像头

while True:

    ret, frame = cap.read()
    ret_usb, frame_usb = cap_usb.read()
    if (not ret) or (not ret_usb):
        print("can't read frame")
        break
    print(frame)
    #frame=cv2.resize(frame,(1280,480))
    cv2.imshow('frame',frame)
    cv2.imshow('usb',frame_usb)
    if cv2.waitKey(1) & 0xFF == ord('q'):#&是与运算
        break
#cv2.waitKey()#表示等待键盘输入，必须要有窗口的情况下waitkey才能有用，不能在cmd窗口下按下按键

# cv2.imshow("capture", frame)


# cv2.waitKey(5000)
cap.release() 
cap_usb.release() 
cv2.destroyAllWindows()
print("end")




# import argparse

# # (1) 声明一个parser
# parser = argparse.ArgumentParser()
# # (2) 添加参数
# parser.add_argument("parg")             # 位置参数，这里表示第一个出现的参数赋值给parg
# parser.add_argument("--digit",type=int,help="输入数字") # 通过 --echo xxx声明的参数，为int类型
# parser.add_argument("--name",help="名字",default="cjf") # 同上，default 表示默认值
# # (3) 读取命令行参数
# args = parser.parse_args()
 
# # (4) 调用这些参数
# print(args.parg)
# print("echo ={0}".format(args.digit))
# print("name = {}".format(args.name))


# class AuctionException(Exception): pass
# class AuctionTest:
#   def __init__(self, init_price):
#     self.init_price = init_price
#   def bid(self, bid_price):
#     d = 0.0
#     try:
#       d = float(bid_price)
#     except Exception as e:
#       # 此处只是简单地打印异常信息
#       print("转换出异常：", e)
#       # 再次引发自定义异常
#       #raise AuctionException("竞拍价必须是数值，不能包含其他字符！") # ①
#       raise AuctionException(e)
#     if self.init_price < d:
#       raise AuctionException("竞拍价比起拍价低，不允许竞拍！")
#     initPrice = d
# def main():
#   at = AuctionTest(20.4)
#   try:
#     at.bid("df")
#   except AuctionException as ae:
#     # 再次捕获到bid()方法中的异常，并对该异常进行处理
#     print('main函数捕捉的异常：', ae)
# main()

import pandas as pd
   
# 三个字段 name, site, age
nme = [["Google", "Runoob", "Taobao", "Wiki"]]
st = [["www.google.com", "www.runoob.com", "www.taobao.com", "www.wikipedia.org"]]
ag = [[90, 40, 80, 98]]
   
# 字典
dict = {'name': nme, 'site': st, 'age': ag}
     
df = pd.DataFrame(dict)
print(dict)
# 保存 dataframe
# df.to_csv('./site.csv')

y=[1,2,3,4,5,6,7]
x=[1,2,3,4,5,6,7]

yu=[]

yu1=[[1,2],[2,3]]

yu=yu1
print(yu)
for i in x:
    print(i)



