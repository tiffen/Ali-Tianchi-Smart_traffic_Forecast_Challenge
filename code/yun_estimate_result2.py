# -*- coding: utf-8 -*-
# @Time    : 2017/7/26 21:51
# @Author  : LiYun
# @File    : yun_estimate_result.py
'''description:
评估结果所调用的函数
'''
import os

class estimate_result(object):

    def __init__(self,datafile,pre_filename,begindate,timeinterval,days):
        '''输入分别为数据集文件名，待评估文件名，开始日期，时间间隔，持续时间'''
        #数据中每行元素的起始位置，分别为
        #link的结束位置，月，日，（起始时，分，秒），（终止时，分，秒），通行时间
        self.index=[19,25,28,43,46,49,63,66,69,73]
        self.Lnum=132 #一共132条路
        self.total_time=self.get_time(timeinterval) #每一天一共是多少分钟
        true_value=self.abstract_true(datafile,begindate,timeinterval,days)
        prediction_value=self.read_predict(pre_filename)
        score=self.estimate(prediction_value,true_value,days)
        print(score)

    def read_predict(self,filename):
        '''从输入文件中读取数据，排序，返回预测值列表'''
        data = []
        with open(filename) as f:
            for line in f:
                data.append(line[:-1])
        return sorted(data)

    def abstract_true(self,datafile,begindate,timeinterval,days):
        '''返回真实值的路况列表，已排序，所有的 ； 改为 #'''
        true_value = []
        if os.path.exists(datafile)==False:
            bmonth=int(begindate[:2])
            bday=int(begindate[2:])
            bbh=int(timeinterval[:2])
            bbmnt=int(timeinterval[3:5])
            beh=int(timeinterval[6:8])
            bemnt=int(timeinterval[9:])
            with open('yun_complement_dataset_3_7.txt') as f:
                for line in f:
                    lmonth=int(line[self.index[1]:self.index[1]+2])
                    lday=int(line[self.index[2]:self.index[2]+2])
                    lbh=int(line[self.index[3]:self.index[3]+2])
                    lbmnt=int(line[self.index[4]:self.index[4]+2])
                    leh=int(line[self.index[6]:self.index[6]+2])
                    lemnt=int(line[self.index[7]:self.index[7]+2])
                    if leh==0 and lemnt==0:
                        continue
                    if bmonth==lmonth and bday<=lday and lday-bday<=days-1 and \
                        (bbh<lbh or bbh==lbh and bbmnt<=lbmnt) and ((beh>leh or beh==leh and bemnt>=lemnt)):
                        true_value.append(line.replace(';', '#'))
            with open(datafile,'w',newline='') as f:
                f.writelines(true_value)

        with open(datafile) as f:
            for line in f:
                true_value.append(line[:-1])
        return sorted(true_value)

    def estimate(self,prediction_value,true_value,days):
        '''检查数据长度，格式，并返回分数'''
        score=0
        if len(prediction_value)!=len(true_value):
            print('length not equal')
            print('length prediction_value: ',len(prediction_value))
            print('length true_value: ',len(true_value))
            return -1
        else:
            print(len(prediction_value))
        for p,t in zip(prediction_value,true_value):
            if p[:self.index[9]]!=t[:self.index[9]]:
                print('format is not corret')
                return -1
            a=float(p[self.index[9]:]) #预测路况
            b=float(t[self.index[9]:]) #真实路况
            score+=abs(a-b)/b
        return score/self.Lnum/self.total_time/days


    def get_time(self,time):
        '''得到一共有多少时间片'''
        bm=int(time[:2])*60+int(time[3:5])
        em=int(time[6:8])*60+int(time[9:])
        return (em-bm)/2


#第2个参数不支持跨年，跨月，第3个参数不支持跨日
if __name__=='__main__':
    estimate_result('yun_true_data_half_6.txt',  'yun_20170903_evaluate_half_6.txt', '0617', '08:00-09:00', 14)































