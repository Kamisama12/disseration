#!/usr/bin/env python3

#test file input and output by numpy
from os import read
from matplotlib import markers
# from matplotlib.lines import _LineStyle
import matplotlib.lines as lines
# from matplotlib.lines import _LineStyle
import matplotlib.pyplot
import numpy as np

import matplotlib.pylab as plt
# from matplotlib import lines





import matplotlib.pyplot as plt
import csv

# from franka_ros.franka_example_controllers.scripts.control_end_effector import Force_EE

def main():

    x = []
    y = []
    z = []

    desired_x = []
    desired_y = []
    desired_z = []

    Fx = []
    Fy = []
    Fz = []

    Fx_EE = []
    Fy_EE = []
    Fz_EE = []

    qx = []
    qy = []
    qz = []
    qw = []

    Dqx = []
    Dqy = []
    Dqz = []
    Dqw = []



    euler_angle_x = []
    euler_angle_y = []
    euler_angle_z = []

    euler_angle_dx = []
    euler_angle_dy = []
    euler_angle_dz = []


    joint_angle_0 = []
    joint_angle_1 = []
    joint_angle_2 = []
    joint_angle_3 = []

    joint_angle_4 = []
    joint_angle_5 = []
    joint_angle_6 = []



    joint_tau_0 = []
    joint_tau_1 = []
    joint_tau_2 = []
    joint_tau_3 = []

    joint_tau_4 = []
    joint_tau_5 = []
    joint_tau_6 = []

    t = []
    time = 0
  
    with open('/home/y/下载/robot_state(2).csv','r') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        for row in lines:
            desired_x.append(float(row[0]))
            desired_y.append(float(row[1]))
            desired_z.append(float(row[2]))

            x.append(float(row[3]))
            y.append(float(row[4]))
            z.append(float(row[5]))

            qx.append(float(row[6]))
            qy.append(float(row[7]))
            qz.append(float(row[8]))
            qw.append(float(row[9]))

            Dqx.append(float(row[10]))
            Dqy.append(float(row[11]))
            Dqz.append(float(row[12]))
            Dqw.append(float(row[13]))


            Fx.append(float(row[14]))
            Fy.append(float(row[15]))
            Fz.append(float(row[16]))


            euler_angle_x.append(float(row[17]))
            euler_angle_y.append(float(row[18]))
            euler_angle_z.append(float(row[19]))


            euler_angle_dx.append(float(row[20]))
            euler_angle_dy.append(float(row[21]))
            euler_angle_dz.append(float(row[22]))

            Fx_EE.append(float(row[23]))
            Fy_EE.append(float(row[24]))
            Fz_EE.append(float(row[25]))

            t.append(time)
            time =time + 0.1
 


    # plt.ylim(0.02,0.04)   
    # print(max(z)-min(z))

    plt.figure(1)
    plt.subplot(2,2,1)
    plt.xlabel("Time/s")
    plt.ylabel("X/m")
    plt.title("X")
    plt.plot(t, desired_x,label='desired_x',color = 'r',)
    plt.plot(t, x,label="x ",color = 'g',)
    plt.legend()


    plt.subplot(2,2,2)
    plt.xlabel("Time/s")
    plt.ylabel("Y/m")
    plt.title("Y")
    plt.plot(t, desired_y,label='desired_y',color = 'r',)
    plt.plot(t, y,label="Y ",color = 'g',)
    plt.legend()




    plt.subplot(2,2,3)
    plt.xlabel("Time/s")
    plt.ylabel("Z/m")
    plt.title("Z")
    plt.plot(t, desired_z,label='desired_z',color = 'r',)
    plt.plot(t, z,label="Z ",color = 'g',)
    plt.legend()


    plt.figure(2)
    plt.subplot(2,2,1)
    plt.xlabel("Time/s")
    plt.ylabel("qx")
    plt.title("qx")
    plt.plot(t, Dqx,label='Dqx',color = 'r',)
    plt.plot(t, qx,label="qx ",color = 'g',)
    plt.legend()


    plt.subplot(2,2,2)
   
    plt.xlabel("Time/s")
    plt.ylabel("qy")
    plt.title("qy")
    plt.plot(t, Dqy,label='Dqy',color = 'r',)
    plt.plot(t, qy,label="qy ",color = 'g',)
    plt.legend()




    plt.subplot(2,2,3)

    plt.xlabel("Time/s")
    plt.ylabel("qz")
    plt.title("qz")
    plt.plot(t, Dqx,label='Dqz',color = 'r',)
    plt.plot(t, qx,label="qz",color = 'g',)
    plt.legend()

    plt.subplot(2,2,4)
    plt.xlabel("Time/s")
    plt.ylabel("qw")
    plt.title("qw")
    plt.plot(t, Dqx,label='Dqw',color = 'r',)
    plt.plot(t, qx,label="qw",color = 'g',)
    plt.legend()


    plt.figure(3)

    plt.subplot(2,2,1)

    plt.xlabel("Time/s")
    plt.ylabel("X/m")
    plt.title("X")
    plt.plot(t, euler_angle_x,label='euler_angle_x',linestyle=":",marker="o", color = 'r')
    plt.plot(t, euler_angle_dx,label='euler_angle_dx',color = 'g',)

    plt.legend()


    plt.subplot(2,2,2)
    plt.xlabel("Time/s")
    plt.ylabel("Y/m")
    plt.title("Y")
    plt.plot(t, euler_angle_y,label='euler_angle_y',linestyle=":",marker="o",color = 'r',)
    plt.plot(t, euler_angle_dy,label='euler_angle_dy',color = 'g',)
    plt.legend()


    plt.subplot(2,2,3)
    plt.xlabel("Time/s")
    plt.ylabel("Z/m")
    plt.title("Z")
    plt.plot(t, euler_angle_z,label='euler_angle_z',linestyle=":",marker="o",color = 'r',)
    plt.plot(t, euler_angle_dz,label='euler_angle_dz',color = 'g',)
    plt.legend()


    # plt.subplot(2,2,4)
    # plt.xlabel("Time/s")
    # plt.ylabel("Force_Z/N")
    # plt.title("Force_Z")
    # plt.plot(t, z,color = 'g',)



    plt.figure(4)
    plt.subplot(2,3,1)
    plt.xlabel("Time/s")
    plt.ylabel("Fx/m")
    plt.title("Fx")
    plt.plot(t, Fx,label='Fx',color = 'r',)
    # plt.plot(t, x,label="x ",color = 'g',)
    plt.legend()


    plt.subplot(2,3,2)
    plt.xlabel("Time/s")
    plt.ylabel("Fy/m")
    plt.title("Fy")
    plt.plot(t, Fy,label='Fy',color = 'r',)
    plt.legend()




    plt.subplot(2,3,3)
    plt.xlabel("Time/s")
    plt.ylabel("Fz/N")
    plt.title("Fz")
    plt.plot(t, Fz,label='Fz',color = 'r',)
    plt.legend()


    plt.subplot(2,3,4)
    plt.xlabel("Time/s")
    plt.ylabel("Fx_EE/m")
    plt.title("Fx_EE")
    plt.plot(t, Fx_EE,label='Fx_EE',color = 'r',)
    # plt.plot(t, x,label="x ",color = 'g',)
    plt.legend()


    plt.subplot(2,3,5)
    plt.xlabel("Time/s")
    plt.ylabel("Fy_EE/m")
    plt.title("Fy_EE")
    plt.plot(t, Fy_EE,label='Fy_EE',color = 'r',)
    plt.legend()




    plt.subplot(2,3,6)
    plt.xlabel("Time/s")
    plt.ylabel("Fz_EE/N")
    plt.title("Fz_EE")
    plt.plot(t, Fz_EE,label='Fz_EE',color = 'r',)
    plt.legend()

    # plt.show()

    # with open('/home/swy/catkin_ws/src/franka_ros/franka_example_controllers/scripts/data/test12.csv','r') as csvfile:
    #     lines = csv.reader(csvfile, delimiter=',')
    #     for row in lines:
    #         desired_x.append(float(row[0]))
    #         desired_y.append(float(row[1]))
    #         desired_z.append(float(row[2]))

    #         x.append(float(row[3]))
    #         y.append(float(row[4]))
    #         z.append(float(row[5]))

    #         qx.append(float(row[6]))
    #         qy.append(float(row[7]))
    #         qz.append(float(row[8]))
    #         qw.append(float(row[9]))

    #         Dqx.append(float(row[10]))
    #         Dqy.append(float(row[11]))
    #         Dqz.append(float(row[12]))
    #         Dqw.append(float(row[13]))


    #         Fx.append(float(row[14]))
    #         Fy.append(float(row[15]))
    #         Fz.append(float(row[16]))
    #         t.append(time)
    #         time =time + 0.1
 
    # plt.figure(1)
    # plt.subplot(2,2,1)
    # plt.xlabel("Time/s")
    # plt.ylabel("X/m")
    # plt.title("X")
    # plt.plot(t, desired_x,label='desired_x',color = 'r',)
    # plt.plot(t, x,label="x ",color = 'g',)
    # plt.legend()


    # plt.subplot(2,2,2)
    # plt.xlabel("Time/s")
    # plt.ylabel("Y/m")
    # plt.title("Y")
    # plt.plot(t, desired_y,label='desired_y',color = 'r',)
    # plt.plot(t, y,label="Y ",color = 'g',)
    # plt.legend()




    # plt.subplot(2,2,3)
    # plt.xlabel("Time/s")
    # plt.ylabel("Z/m")
    # plt.title("Z")
    # plt.plot(t, desired_z,label='desired_z',color = 'r',)
    # plt.plot(t, z,label="Z ",color = 'g',)
    # plt.legend()


    # plt.figure(2)
    # plt.subplot(2,2,1)
    # plt.xlabel("Time/s")
    # plt.ylabel("qx")
    # plt.title("qx")
    # plt.plot(t, Dqx,label='Dqx',color = 'r',)
    # plt.plot(t, qx,label="qx ",color = 'g',)
    # plt.legend()


    # plt.subplot(2,2,2)
   
    # plt.xlabel("Time/s")
    # plt.ylabel("qy")
    # plt.title("qy")
    # plt.plot(t, Dqy,label='Dqy',color = 'r',)
    # plt.plot(t, qy,label="qy ",color = 'g',)
    # plt.legend()




    # plt.subplot(2,2,3)

    # plt.xlabel("Time/s")
    # plt.ylabel("qz")
    # plt.title("qz")
    # plt.plot(t, Dqx,label='Dqz',color = 'r',)
    # plt.plot(t, qx,label="qz",color = 'g',)
    # plt.legend()

    # plt.subplot(2,2,4)
    # plt.xlabel("Time/s")
    # plt.ylabel("qw")
    # plt.title("qw")
    # plt.plot(t, Dqx,label='Dqw',color = 'r',)
    # plt.plot(t, qx,label="qw",color = 'g',)
    # plt.legend()


    # plt.figure(3)
    # plt.subplot(2,2,1)
    # plt.xlabel("Time/s")
    # plt.ylabel("Fx/m")
    # plt.title("Fx")
    # plt.plot(t, Fx,label='Fx',color = 'r',)
    # # plt.plot(t, x,label="x ",color = 'g',)
    # plt.legend()


    # plt.subplot(2,2,2)
    # plt.xlabel("Time/s")
    # plt.ylabel("Fy/m")
    # plt.title("Fy")
    # plt.plot(t, Fy,label='Fy',color = 'r',)
    # plt.legend()




    # plt.subplot(2,2,3)
    # plt.xlabel("Time/s")
    # plt.ylabel("Fz/N")
    # plt.title("Fz")
    # plt.plot(t, Fx,label='Fz',color = 'r',)
    # plt.legend()


    t = []
    time = 0
    with open('/home/y/下载/robot_state(2).csv','r') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        for row in lines:
            joint_angle_0.append(float(row[0]))
            joint_angle_1.append(float(row[1]))
            joint_angle_2.append(float(row[2]))

            joint_angle_3.append(float(row[3]))
            joint_angle_4.append(float(row[4]))
            joint_angle_5.append(float(row[5]))

            joint_angle_6.append(float(row[6]))
         
            t.append(time)
            time =time + 0.1
 
    plt.figure(5)
    plt.subplot(2,4,1)
    plt.xlabel("Time/s")
    plt.ylabel("joint0")
    plt.title("joint0")
    plt.plot(t, joint_angle_0,label='joint_angle_0',color = 'r',)
    # plt.plot(t, qx,label="qx ",color = 'g',)
    plt.legend()

    plt.subplot(2,4,2)
    plt.xlabel("Time/s")
    plt.ylabel("joint1")
    plt.title("joint1")
    plt.plot(t, joint_angle_1,label='joint_angle_0',color = 'r',)
    # plt.plot(t, qx,label="qx ",color = 'g',)
    plt.legend()

    plt.subplot(2,4,3)
    plt.xlabel("Time/s")
    plt.ylabel("joint2")
    plt.title("joint2")
    plt.plot(t, joint_angle_2,label='joint_angle_2',color = 'r',)
    # plt.plot(t, qx,label="qx ",color = 'g',)
    plt.legend()

    plt.subplot(2,4,4)
    plt.xlabel("Time/s")
    plt.ylabel("joint3")
    plt.title("joint3")
    plt.plot(t, joint_angle_3,label='joint_angle_3',color = 'r',)
    # plt.plot(t, qx,label="qx ",color = 'g',)
    plt.legend()



    plt.subplot(2,4,5)
    plt.xlabel("Time/s")
    plt.ylabel("joint4")
    plt.title("joint4")
    plt.plot(t, joint_angle_4,label='joint_angle_4',color = 'r',)
    # plt.plot(t, qx,label="qx ",color = 'g',)
    plt.legend()


    plt.subplot(2,4,6)
    plt.xlabel("Time/s")
    plt.ylabel("joint5")
    plt.title("joint5")
    plt.plot(t, joint_angle_5,label='joint_angle_5',color = 'r',)
    # plt.plot(t, qx,label="qx ",color = 'g',)
    plt.legend()


    plt.subplot(2,4,7)
    plt.xlabel("Time/s")
    plt.ylabel("joint6")
    plt.title("joint6")
    plt.plot(t, joint_angle_6,label='joint_angle_6',color = 'r',)
    # plt.plot(t, qx,label="qx ",color = 'g',)
    plt.legend()



    t = []
    time = 0
    with open('/home/y/下载/robot_state(2).csv','r') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        for row in lines:
            joint_tau_0.append(float(row[0]))
            joint_tau_1.append(float(row[1]))
            joint_tau_2.append(float(row[2]))

            joint_tau_3.append(float(row[3]))
            joint_tau_4.append(float(row[4]))
            joint_tau_5.append(float(row[5]))

            joint_tau_6.append(float(row[6]))
         
            t.append(time)
            time =time + 0.1
 
    plt.figure(6)
    plt.subplot(2,4,1)
    plt.xlabel("Time/s")
    plt.ylabel("joint0")
    plt.title("joint0")
    plt.plot(t, joint_tau_0,label='joint_angle_0',color = 'r',)
    # plt.plot(t, qx,label="qx ",color = 'g',)
    plt.legend()

    plt.subplot(2,4,2)
    plt.xlabel("Time/s")
    plt.ylabel("joint1")
    plt.title("joint1")
    plt.plot(t, joint_tau_1,label='joint_angle_0',color = 'r',)
    # plt.plot(t, qx,label="qx ",color = 'g',)
    plt.legend()

    plt.subplot(2,4,3)
    plt.xlabel("Time/s")
    plt.ylabel("joint2")
    plt.title("joint2")
    plt.plot(t, joint_tau_2,label='joint_angle_2',color = 'r',)
    # plt.plot(t, qx,label="qx ",color = 'g',)
    plt.legend()

    plt.subplot(2,4,4)
    plt.xlabel("Time/s")
    plt.ylabel("joint3")
    plt.title("joint3")
    plt.plot(t, joint_tau_3,label='joint_angle_3',color = 'r',)
    # plt.plot(t, qx,label="qx ",color = 'g',)
    plt.legend()



    plt.subplot(2,4,5)
    plt.xlabel("Time/s")
    plt.ylabel("joint4")
    plt.title("joint4")
    plt.plot(t, joint_tau_4,label='joint_angle_4',color = 'r',)
    # plt.plot(t, qx,label="qx ",color = 'g',)
    plt.legend()


    plt.subplot(2,4,6)
    plt.xlabel("Time/s")
    plt.ylabel("joint5")
    plt.title("joint5")
    plt.plot(t, joint_tau_5,label='joint_angle_5',color = 'r',)
    # plt.plot(t, qx,label="qx ",color = 'g',)
    plt.legend()


    plt.subplot(2,4,7)
    plt.xlabel("Time/s")
    plt.ylabel("joint6")
    plt.title("joint6")
    plt.plot(t, joint_tau_6,label='joint_angle_6',color = 'r',)
    # plt.plot(t, qx,label="qx ",color = 'g',)
    plt.legend()


    plt.show()

if __name__ == '__main__':
    main()
