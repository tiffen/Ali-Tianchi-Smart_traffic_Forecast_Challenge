# -*- coding: utf-8 -*-
# @Time    : 2017/8/8 14:46
# @Author  : LiYun
# @File    : yun_predict_based_on_ffnn_median_assist.py
'''description:
定义一些yun_predict_based_on_ffnn_median.py中使用的函数
'''
import numpy as np
import tensorflow as tf
import ffnn_inference as ffnn1
import ffnn_inference2 as ffnn2

# 每个月有几天
mdays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
#2017年每个月第一天是星期几
first_weekday = (7,3,3,6,1,4,6,2,5,7,3,5)
# 输入数据中每行元素的起始位置，分别为
# link的结束位置，月，日，（起始时，分，秒），（终止时，分，秒），通行时间
Lp = [19, 25, 28, 43, 46, 49, 63, 66, 69, 73]
# 2017年3-7月节假日的列表
festerval=[[4,5,11,12,18,19,25,26],
           [2,3,4,8,9,15,16,22,23,29,30],
           [1,6,7,13,14,20,21,28,29,30],
           [3,4,10,11,17,18,24,25],
           [1,2,8,9,15,16,22,23,29,30]]

def get_all_links(linkinfo):
    '''给定link信息文件，返回所有路段的名称与对应序号所构成的字典'''
    Links=[]
    with open(linkinfo) as f:
        f.readline()
        while True:
            line=f.readline()
            if not line:
                break
            Links.append(line[:Lp[0]])
    Links=sorted(Links)
    Links2={}
    Links3={}
    num=0
    for l in Links:
        Links2[l]=num
        Links3[num]=l
        num+=1
    return Links2,Links3

def readdata(infile):
    '''从输入文件中读取数据，将数据存储在列表里并返回，注意输入文件没有表头'''
    data = []
    with open(infile) as f:
        while True:
            line = f.readline()
            if not line:
                break
            data.append(line[:-1])
    return data

def get_day(d,btime):
    '''给定日期，推出这是第几天'''
    if d[:2]==btime[:2]: #相同月份
        return int(d[3:5])-int(btime[3:5])
    else: #不同月份
        a=int(btime[3:5])
        b=int(d[3:5])
        for i in range(int(btime[:2]),int(d[:2])):
            b+=mdays[i-1]
        return b-a

def get_week(d):
    '''给定日期，推出这是周几'''
    weekday = (int(d[3:5]) + first_weekday[int(d[:2]) - 1] - 2) % 7 + 1
    return weekday

def get_workday():
    workday = np.empty(153, np.int8)
    knt=0
    for m in range(3,8):
        for d in range(1,mdays[m-1]+1):
            if d in festerval[m-3]:
                workday[knt]=0
            else:
                workday[knt]=1
            knt+=1
    return workday

def get_slide(t):
    '''给定时间，推出这是第几个时间片'''
    bm=int(t[:2])*60+int(t[3:5])
    em=int(t[6:8])*60+int(t[9:])
    return int((em-bm)/2)

def get_history(data):
    '''得到3 月至6 月每天 05:00 到 10:00 的历史数据
    和7月每天 06:00 到 08:00 的当前数据'''
    his=np.empty((122,150),np.float) #一共是122天，5个小时是5*30=150个时间片，1条Link
    cur=np.empty((31,60),np.float) #一共是31天，60个时间片，1条Link
    for info in data:
        if int(info[Lp[1]:Lp[1] + 2]) == 7:
            if info[Lp[3]:Lp[3] + 2]!='06' and info[Lp[3]:Lp[3] + 2]!='07': continue
            day = get_day(info[Lp[1]:Lp[1] + 5], '07-01')  # 第几天，从'07-01'算起
            slide = get_slide('06:00-' + info[Lp[3]:Lp[3] + 5])  # 第几个时间片，从'06:00'算起
            cur[day, slide] = float(info[Lp[9]:])
        elif int(info[Lp[3]:Lp[3]+2])<5 or int(info[Lp[3]:Lp[3]+2])>=10:
            # 不使用3-6月05:00 到 10:00以外的数据
            continue
        else:
            day=get_day(info[Lp[1]:Lp[1]+5],'03-01') #第几天，从'03-01'算起
            slide=get_slide('05:00-'+info[Lp[3]:Lp[3]+5]) #第几个时间片，从'05:00'算起
            his[day,slide]=float(info[Lp[9]:])
    return his,cur

def construct_train(data,work_day):
    len_w=np.sum(work_day)
    len_f=len(data)-len_w
    wtrainX,wtrainY,ftrainX,ftrainY = np.empty([len_w , 60, 30]),np.empty([len_w , 60, 30]),np.empty([len_f , 60, 30]),np.empty([len_f , 60, 30])
    dw,df=0,0
    for d in range(len(data)):
        if work_day[d]==1: #工作日
            for slide in range(60):
                wtrainX[dw , slide] = np.mean(data[d, slide:slide + 60].reshape([-1,2]),axis=1) #每相邻的两个时间片求平均
                wtrainY[dw , slide] = data[d, slide + 60:slide + 90]
            dw+=1
        else: #节假日
            for slide in range(60):
                ftrainX[df , slide] = np.mean(data[d, slide:slide + 60].reshape([-1,2]),axis=1) #每相邻的两个时间片求平均
                ftrainY[df , slide] = data[d, slide + 60:slide + 90]
            df+=1
    return wtrainX,wtrainY,ftrainX,ftrainY

def construct_val(data):
    valX = np.empty([len(data), 60, 30])
    valY = np.empty([len(data), 60, 30])
    testX = np.empty([len(data), 30])
    testY = np.empty([len(data), 30])
    for d in range(len(data)):
        for slide in range(60):
            valX[d, slide] = np.mean(data[d, slide:slide + 60].reshape([-1,2]),axis=1) #每相邻的两个时间片求平均
            valY[d, slide] = data[d, slide + 60:slide + 90]
        slide = 30  # his从5点开始，第30个时间片是6点
        testX[d] = np.mean(data[d, slide:slide + 60].reshape([-1,2]),axis=1) #每相邻的两个时间片求平均
        testY[d] = data[d, slide + 60:slide + 90]
    return valX, valY,testX,testY

def construct_PreX(data):
    dataX = np.empty([len(data),30])
    for d in range(len(data)): dataX[d] = np.mean(data[d].reshape([-1,2]),axis=1) #每相邻的两个时间片求平均
    return dataX

def construct_sample_set(his,cur,work_day):
    # work_day=(153) 判断从3月1号到7月31号，每天是工作日还是节假日，工作日=1，节假日=0，返回一个153维的numpy数组
    # his=(122,150) 一共是122天，5个小时是5*30=150个时间片，1条Link
    # cur=(31,60) 一共是31天，60个时间片，1条Link
    # 训练集：  针对工作日和节假日分别构造训练集，训练两套模型
    #   工作日：  0301-0616  5:00-10:00  每天有60个样本  一共是75天  wtrainX[75,60,30]  wtrainY[75,60,30]
    #   节假日：  0301-0616  5:00-10:00  每天有60个样本  一共是33天  ftrainX[33,60,30]  ftrainY[33,60,30]
    # 验证集：  0617-0630  5:00-10:00  每天有60个样本  一共是14天  valX[14,60,30]  valY[14,60,30]
    # 验证子集：0617-0630  6:00-9:00  每天有1个样本  一共是14天  testX[14,30]  testY[14,30]
    # 测试集：  0701-0731  6:00-8:00  每天有1个样本  一共是31天  PredX[31,30]
    wtrainX,wtrainY,ftrainX,ftrainY=construct_train(his[:108],work_day[:108]) #his[:108] [108,150]
    valX,valY,testX,testY=construct_val(his[-14:])     # his[-14:]   [14,150]
    PredX=construct_PreX(cur)
    return wtrainX,wtrainY,ftrainX,ftrainY,valX,valY,testX,testY,PredX

def construct_nn_graph1(Train,x,y_,keep_prob,input_node,output_node):#构建神经网络1
    learning_rate = 0.0005
    if Train==0:
        regularizer=tf.contrib.layers.l2_regularizer(0.0005)
        y = ffnn1.inference(x, keep_prob, input_node=input_node, output_node=output_node,regularizer=regularizer)
        cost = tf.reduce_mean(tf.div(tf.abs(y-y_),y_)) + tf.add_n(tf.get_collection('losses'))
    else:
        y=ffnn1.inference(x,keep_prob, input_node=input_node, output_node=output_node)
        cost = tf.reduce_mean(tf.div(tf.abs(y - y_), y_))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    return train_op,cost,y

def construct_nn_graph2(Train,x,y_,keep_prob,input_node,output_node):#构建神经网络2
    learning_rate = 0.0005
    if Train==0:
        regularizer=tf.contrib.layers.l2_regularizer(0.0005)
        y = ffnn2.inference(x, keep_prob, input_node=input_node, output_node=output_node,regularizer=regularizer)
        cost = tf.reduce_mean(tf.div(tf.abs(y-y_),y_)) + tf.add_n(tf.get_collection('losses'))
    else:
        y=ffnn2.inference(x,keep_prob, input_node=input_node, output_node=output_node)
        cost = tf.reduce_mean(tf.div(tf.abs(y - y_), y_))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    return train_op,cost,y

def find_a_best_number(data):#从一个一维数组中找到一个数，使得评价函数最小
    data=np.sort(data)
    data_len=len(data)
    index=int(data_len/2)
    num=data[index]
    score=1e8
    score2=np.mean(np.abs(data-num)/data)
    while score2<=score and index>0:
        index-=1
        score=score2
        score2=np.mean(np.abs(data-data[index])/data)

    if score2>score:
        return data[index+1]
    else:
        return data[index]
























