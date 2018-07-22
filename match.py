# -*- coding: utf-8 -*-
"""
Created on Tue May 16 16:01:36 2017

@author: Administrator
"""
import numpy as np
import cv2 


def des_distance(deep_des1,deep_des2):
    error = deep_des1-deep_des2
    RMSE = np.sqrt(np.sum(np.square(error),axis=1))/deep_des1.shape[0]
    
    return RMSE
    
def deep_match(kp1_location,kp2_location,deep_des1,deep_des2,ratio):
    deep_kp1 = []
    deep_kp2 = []
    for i in range(deep_des1.shape[0]):
        des = np.tile(deep_des1[i],(deep_des2.shape[0],1))
        error = des - deep_des2
        RMSE = np.sqrt(np.sum(np.square(error),axis=1)/error.shape[1])
        small_index = np.argsort(RMSE, axis=0)
        if RMSE[small_index[0]]< RMSE[small_index[1]]*ratio:
            deep_kp1.append((kp1_location[i][0],kp1_location[i][1]))
            deep_kp2.append((kp2_location[small_index[0]][0],kp2_location[small_index[0]][1]))
            #deep_des2 = np.delete(deep_des2, small_index[0], 0)
    return deep_kp1,deep_kp2

#match sift keypoints
def match(kp1_location, kp2_location, deep_des1, deep_des2, ratio):  # ratio = 0.7
    deep_kp1 = []
    deep_kp2 = []
    des1 = np.matrix(deep_des1)
    des2 = np.matrix(deep_des2)
    for i in range(des1.shape[0]):  # 遍历des1中1158个点
        des1_ = np.tile(des1[i], (des2.shape[0], 1))  # 使des[i]扩展成des2.shape[0]×128的矩阵
        error = des1_ - des2  # 相减得到描述符的差值矩阵
        RMSE = np.sqrt(np.sum(np.square(error), axis=1)/float(error.shape[1]))   # 均方根误差
        small_index = np.argsort(RMSE, axis=0)  # 从小到大排得到1510×1的索引矩阵
        if RMSE[small_index[0, 0], 0] < RMSE[small_index[1, 0], 0]*ratio:  # 如果最小的RMSE值比次小的RMSE×0.7还小，则匹配
            deep_kp1.append((kp1_location[i][0], kp1_location[i][1]))  # 在deep_kp1中添加匹配点坐标（x，y）
            deep_kp2.append((kp2_location[small_index[0, 0]][0], kp2_location[small_index[0, 0]][1]))  # 在deep_kp2中添加匹配点坐标（x，y）
            #deep_des2 = np.delete(deep_des2, small_index[0], 0)
    return deep_kp1, deep_kp2