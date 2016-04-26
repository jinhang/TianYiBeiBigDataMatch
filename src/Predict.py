__author__ = 'keehang'
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import os
import Preprocess
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # 分割原始的数据
    # filePath = "/home/keehang/Temp/TianYi/part-r-00000"
    # saveDir = "/home/keehang/Temp/TianYi/splitCSV"
    # count = 20
    # if not os.path.exists(saveDir):
    #     os.makedirs(saveDir)
    # Preprocess.split_ori_data(filePath, saveDir, count)

    # 按照星期来进行统计
    # source = pd.read_table("/home/keehang/Temp/TianYi/part-r-00000", names=['Userid', 'Date', 'Type', 'Count'])
    # Preprocess.get_10sites_everyDay_data_without_split(source, "/home/keehang/Temp/TianYi/按照星期统计")

    # 分割每个用户的数据文件
    oriPath = "/home/keehang/Temp/TianYi/part-r-00000"
    source = pd.read_table("/home/keehang/Temp/TianYi/part-r-00000", names=['Userid', 'Date', 'Type', 'Count'])
    saveDir = "/home/keehang/Temp/TianYi/everyOneData"
    #Preprocess.get_everyOne_data(source, saveDir)

    Preprocess.linear_predict_with_everySiteData(saveDir)







