___author___ = 'keehang'
# -*- coding:utf-8 -*-

("\n"
 "    此程序文件包含对TianYi竞赛数据的相关的预处理操作，包括数据的读入、写出、分割和一些数据的索引分片等\n"
 "    ___date___ = '2015/12/18'\n"
 "    ___version___ = 1.0\n")

import pandas as pd
import numpy as np
import os
from pandas import DataFrame
from sklearn import linear_model
import math


def load_data(filePath):
    """
    读入TianYi的数据文件
    :param filepath: 原始的文件路径
    :return: 一个包含所有数据的DataFrame对象
    """
    oriSource = pd.read_table(filePath, sep=',', names=['Userid', 'Date', 'Type', 'Count'])
    return oriSource


def split_ori_data(filePath, saveDir, count):
    """
    将原始的数据文件分割成子文件，分割按照文件行数来进行
    :param filePath: 原始数据的路径
    :param saveDir: 分割后数据的存储路径
    :param count: 需要将原始数据分割成几份
    :return:
    """
    oriSource = load_data(filePath)
    # 获得原始数据的行数
    rows = oriSource.count()[0]
    nrows = rows/count
    # 前面count-1份行数是nrows，最后一份是rows-nrows*(count-1)
    for index in xrange(0, count-1):
        indexStart = nrows*index
        indexEnd = nrows*(index+1)
        tempSource = oriSource[indexStart:indexEnd]
        # 不写表头和index
        saveCSVPath = saveDir + "/part-" + str(index+1) + ".csv"
        tempSource.to_csv(saveCSVPath, index=False, header=False)
    # 存储最后一个CSV文件
    saveCSVPath = saveDir + "/part-" + str(count) + ".csv"
    tempSource = oriSource[indexEnd:]
    tempSource.to_csv(saveCSVPath, index=False, header=False)


def decard_analamous_data(limit, source):
    """
    丢弃那些Count值可能很大的异常数据
    :param limit: "Count"的最大值
    :param source: 需要丢弃异常数据的元数据
    :return: 新的source
    """
    return source[source['Count'] <= limit]


def get_all_users(source):
    """
    获得所有不同用户的ID，并统计其出现的次数
    :param source: 原始的数据
    :return: 一个numpy的ndarray对象，包含所有不重复的用户ID和一个Series对象，包含相应的不重复的用户的条目数
    """
    nameArray = source['Userid'].unique()
    countSeries = source['Userid'].value_counts()
    return nameArray, countSeries


def split_data_by_count(countList, source, saveDir):
    """
    根据Count的值的范围将原始的数据分割成几个段并分别存储
    :param countList: countList是一个列表，包含想分段的值(依据Count的数量对数据进行分段)，比如[0,30,60,90,120,150,200]
    :param source: source是原始的数据
    :param saveDir: 分割数据的存储目录
    :return:
    """
    # 获得source中"Count"字段的最大值
    maxCount = source['Count'].max()
    # 创建目录
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    for index in range(0, len(countList)):
        if index == len(countList)-1:
            rangeData = range(countList[index], maxCount+1)
        rangeData = range(countList[index], countList[index+1])
        segementData = source[source['Count'].isin(rangeData)]
        if len(segementData['Count']) > 0:
            # 存入文件
            savePath = saveDir + "/count_" + str(min(rangeData)) + "-" + str(max(rangeData)) + ".csv"
            segementData.to_csv(savePath, index=False, header=False)


def get_10sites_everyDay_data_after_split(sourceDir, saveDir):
    """
    将数据按照每一天来进行统计，每一天每个用户对10个网站的访问量，存储为7个CSV文件
    :param sourceDir:进行提取的数据文件目录，里面是分割过的CVS文件
    :param saveDir:数据存储的文件目录
    :return:
    """
    # 获得所有的不同用户的ID
    for root, childDir, fileNames in os.walk(sourceDir):
        for index in xrange(0, len(fileNames)):
            fileName = fileNames[index]
            print fileName
            # 得到分割的文件数据
            tempSource = load_data(os.path.join(root, fileName))
            #print tempSource['Userid'].values.shape
            # 得到分割文件的不同的用户的ID
            if index == 0:
                allUserName, tempUserCount = get_all_users(tempSource)
                #print allUserName.shape, len(tempUserCount)
            if index > 0:
                tempUserName, tempUserCount = get_all_users(tempSource)
                allUserName = np.append(allUserName, tempUserName)
                #print allUserName.shape

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    # 保存用户姓名数据
    saveNamePath = saveDir + "/unique_userId.csv"
    NameDataframe = DataFrame(allUserName)
    NameDataframe.to_csv(saveNamePath, index=False, header=False)
    # np.savetxt(saveNamePath, allUserName)
    """
    此时对应于每个用户都有7行10列的数据
    7行表示周一到周日，10列表示10个网站的访问量数据
    """
    # 对星期进行统计
    for count in xrange(0, 7):
        allUserData = np.zeros((len(allUserName), 10))
        # 在每个分割的文件中进行统计
        for root, childDir, fileNames in os.walk(sourceDir):
            for index in xrange(0, len(fileNames)):
                fileName = fileNames[index]
                # 得到分割的文件数据
                tempSource = load_data(os.path.join(root, fileName))
                nameSeries = tempSource['Userid'].values
                # 得到每个人的数据
                for personIndex in xrange(0, len(allUserName)):
                    for x in xrange(0, len(nameSeries)):
                        #print nameSeries[x], allUserName[personIndex]
                        if nameSeries[x] == allUserName[personIndex]:
                            #print tempSource['Date'].values[x][3]
                            if tempSource['Date'].values[x][3] == str(count+1):
                                #print tempSource['Type'].values[x][1]
                                whichSite = int(tempSource['Type'].values[x][1:])
                                allUserData[personIndex][whichSite-1] += int(tempSource['Count'].values[x])
        # 保存此星期几的数据
        dateDataFrame = DataFrame(allUserData)
        dateDataPath = saveDir + "/" + str(count) + "_Day_for_all_user.csv"
        dateDataFrame.to_csv(dateDataPath, index=False, header=False)


def get_10sites_everyDay_data_without_split(source, saveDir):
    """
     将数据按照每一天来进行统计，每一天每个用户对10个网站的访问量，存储为7个CSV文件
    :param source: 原始的数据文件
    :param saveDir: 统计后保存的目录
    :return:
    """
    # print source['Userid'].values.shape
    uniqueNameArray, uniqueCount = get_all_users(source)
    # print uniqueNameArray.shape
    # 保存人名
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    uniqueNameDataframe = DataFrame(uniqueNameArray)
    saveUniqueNameDataframePath = saveDir + "/unique_userId.csv"
    uniqueNameDataframe.to_csv(saveUniqueNameDataframePath, index=False, header=False)
    # 按照星期统计
    for count in xrange(0, 7):
        savePath = saveDir + "/" + str(count+1) + "_Day_for_all_user.csv"
        allNameSeries = source['Userid'].values
        saveData = np.zeros((len(uniqueNameArray), 10))
        for index in xrange(0, len(allNameSeries)):
            print index
            # 匹配到星期
            if source['Date'].values[index][3] == str(count+1):
                # print source.ix[index]
                # 获得网站的编号
                siteIndex = int(source['Type'].values[index][1:])
                # 获得访问量的值
                siteCount = int(source['Count'].values[index])
                # 获得原始未去重的人名在去重之后的数组中的索引值
                boolIndex = (uniqueNameArray == allNameSeries[index])
                indexOfUnique = boolIndex.argmax()
                # print uniqueNameArray[indexOfUnique], source['Userid'][index]
                # 更新数据
                saveData[indexOfUnique][siteIndex-1] += siteCount
        saveDataframe = DataFrame(saveData)
        saveDataframe.to_csv(savePath, index=False, header=False)
        print "Has Done wit Day " + str(count+1)
    print "All Done!"


def get_everyOne_data(source, saveDir):
    """
    将每个用户的数据存储成一个文件
    :param source: 原始的数据文件
    :param saveDir: 保存分割数据的目录
    :return:
    """
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    count = 0
    for name, group in source.groupby('Userid'):
        count += 1
        saveChildDir = saveDir + "/" + str(count/1000+1)
        if not os.path.exists(saveChildDir):
            os.makedirs(saveChildDir)
        savePath = saveChildDir + "/" + str(count) + ".csv"
        group.to_csv(savePath, index=False, header=False)


def linear_predict_with_everySiteData(onePersonDataDir):
    """
    对每个人的数据进行预测，使用线性回归作为预测
    :param onePersonDataDir: 获得的每个人的数据文件目录
    :return:
    """
    predict_answer = open(onePersonDataDir + "/predict_8th_week_V3.txt", 'w')
    # 迭代获得每个CSV文件
    for root, childDirs, files in os.walk(onePersonDataDir):
        for index in xrange(0, len(childDirs)):
            childDirName = childDirs[index]
            childDirPath = os.path.join(root, childDirName)
            for rootChild, emptyDirs, csvFiles in os.walk(childDirPath):
                for csvIndex in xrange(0, len(csvFiles)):
                    csvName = csvFiles[csvIndex]
                    csvPath = os.path.join(childDirPath, csvName)
                    # 获得每个CSV文件数据
                    source = pd.read_table(csvPath, sep=',', names=['Userid', 'Date', 'Type', 'Count'])
                    # 最终的预测结果有10＊7=70个数据
                    final_predict = np.zeros((10, 7), dtype=np.int32)
                    # 使用Type对数据进行分组
                    for name, group in source.groupby('Type'):
                        # 创建目录
                        # saveDataDir = childDirPath + "/" + csvName.split('.')[0]
                        # if not os.path.exists(saveDataDir):
                        #     os.makedirs(saveDataDir)
                        # print group
                        # print name
                        # 迭代每个DataFrame(此时按照的是每一个网站来分)
                        # 对每一个用户来说，对每一个网站有7天的数据
                        # 对网站的访问次数(7周内)
                        countDay = np.zeros((1, 7))
                        # 对网站的访问总次数(7周内)
                        countAllDay = np.zeros((1, 7))
                        # 每个网站的bias
                        bias = np.array([[1, 1, 1, 1, 1, 5, 5], [1, 1, 1, 1, 1, 2, 2], [1, 1, 1, 1, 1, 4, 4],
                                        [1, 1, 1, 1, 1, 5, 5], [1, 1, 1, 1, 1, 2, 2], [1, 1, 1, 1, 1, 3, 3],
                                        [1, 1, 1, 1, 1, 2, 2], [1, 1, 1, 1, 1, 2, 2], [1, 1, 1, 1, 1, 3, 3],
                                        [1, 1, 1, 1, 1, 7, 7]])
                        for x in xrange(0, len(group)):
                            # 获得第几天
                            whichDay = int(group['Date'].values[x][3:])
                            # 获得访问次数
                            countSite = int(group['Count'].values[x])
                            # 更新
                            countDay[0, whichDay-1] += 1
                            countAllDay[0, whichDay-1] += countSite
                        for y in xrange(0, 7):
                                if countDay[0, y] != 0:
                                    # 计算平均值
                                    countAllDay[0, y] = countAllDay[0, y]/countDay[0, y]
                        # countAllDay += bias
                        # 对数据进行一次线性回归预测(可能需要除去异常的点)
                        train_x = np.arange(1, 8).reshape(7, 1)
                        train_y = list(countAllDay)[0]
                        test_x = train_x
                        test_y = linear_predict(train_x, train_y, test_x)
                        # 更新数据
                        line_index = int(name[1:])
                        for data_index in xrange(0, len(test_y)):
                            if test_y[data_index] == 0:
                                test_y[data_index] = bias[line_index-1][data_index]
                        # 写入每个人中的答案数据
                        final_predict[line_index-1:line_index, :] = test_y
                    # 对有的网站用户前7周没有访问，需要直接将其置为bias的值
                    for final_index in xrange(0, 10):
                        boolIndex = (final_predict[final_index] == 0)
                        if boolIndex.all():
                            final_predict[final_index] = bias[final_index]
                    final_predict_write = final_predict.T.flatten()
                    # 写入最终的全部预测的答案
                    line_to_write = source['Userid'].values[0] + "\t"
                    for spaceIndex in xrange(0, len(final_predict_write)):
                        line_to_write += str(final_predict_write[spaceIndex])
                        if spaceIndex != (len(final_predict_write)-1):
                            line_to_write += ","
                    predict_answer.write(line_to_write)
                    predict_answer.write("\r\n")
                    predict_answer.flush()
                    # 存储数据
                    # saveDataframe = DataFrame(countAllDay)
                    # saveDataPath = saveDataDir + "/Site_" + name[1:] + ".csv"
                    # saveDataframe.to_csv(saveDataPath, index=False, header=False)
    predict_answer.close()


def linear_predict(train_x, train_y, test_x):
    """
    对数据进行线性回归预测
    :param train_x: 训练的x数据，N*1
    :param train_y: 训练的y数据，1*N
    :param test_x: 测试的x数据，N*1
    :return: 测试的y数据，1*N
    """
    # 训练线性模型
    regr = linear_model.LinearRegression()
    regr.fit(train_x, train_y)
    # predict
    test_y = regr.predict(test_x)
    # 向上取整，忽略负数
    for index in xrange(0, len(test_y)):
        test_y[index] = math.ceil(test_y[index])
        if test_y[index] <= 0:
            test_y[index] = 0
    return test_y




























