



1.YZH_main_init是程序的初始化函数，已经将openpose和emg的初始化参数封装在里面。


2.YZH_main_run是主函数，直接调用可以实现两个模块同时运行，各自判断姿态，自定义四个列表传进这个函数里面可以将emg的原始数据（已经封装了12帧作为一个list成员传给神经网络做判断），emg的判断标签，openpose的原始数据，和openpose的判断结果标签存储下来。


3.key_emg_handler，on_release，on_press是在emg模式下开启了模拟键盘按键按下的情况，如果不需要可以直接注释掉if里面的语句。

4.emg因为CNN网络判断需要12帧才判断一次，底层的缓存队列也需要进行25次判断且超过一半判断结果相同才会认为用户姿态为该结果，所以数据存储和姿势刷新的速率比较慢。需要通过修改Myo类的队列长度优化

5.openpose的数据集还只有两个非常简单的，还需要扩充。。。。
