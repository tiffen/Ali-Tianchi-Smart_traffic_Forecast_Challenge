# -*- coding: utf-8 -*-
# @Time    : 2017/8/8 12:42
# @Author  : LiYun
# @File    : yun_predict_based_on_ffnn_median.py
'''description:
基于神经网络和中位数联合预测，5 个神经网络ensemble
神经网络的样本是3月1号到6月30号每天早上5点到10点的数据
用两个小时的特征预测后一个小时的特征
每条路针对工作日和节假日分别训练两个神经网络
中位数的方法也是针对每条路的工作日和节假日分别提取
6月最后14天的数据作为验证集
'''
import os
import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import yun_estimate_result2
from yun_predict_based_on_ffnn_median_assist2 import readdata,get_history,get_all_links,get_workday,construct_sample_set,find_a_best_number, construct_nn_graph1, construct_nn_graph2

os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
# 每个月有几天
mdays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
# 输入数据中每行元素的起始位置，分别为
# link的结束位置，月，日，（起始时，分，秒），（终止时，分，秒），通行时间
Lp = [19, 25, 28, 43, 46, 49, 63, 66, 69, 73]
work_day = get_workday()# 判断从3月1号到7月31号，每天是工作日还是节假日，工作日=1，节假日=0，返回一个153维的numpy数组
# 所有路段的名称与对应序号所构成的字典，序号与路段构成的字典
Links, Links2 = get_all_links('gy_contest_link_info.txt')

def ffnn_main(datafile,model_name,link_num,Train,model_flag=0,model_name2=''):
    model_name1=model_name
    #读取历史数据
    temp=np.load(datafile+str(link_num)+'.npz')
    # his=(122,150) 一共是122天，5个小时是5*30=150个时间片，1条Link
    # cur=(31,60) 一共是31天，60个时间片，1条Link
    his,cur=temp['arr_0'],temp['arr_1']
    max_his=np.max(his)
    his=his/max_his #标准化
    cur=cur/max_his #标准化
    # 训练集：  针对工作日和节假日分别构造训练集，训练两套模型
    #   工作日：  0301-0616  5:00-10:00  每天有60个样本  一共是75天  wtrainX[75,60,60]  wtrainY[75,60,30]
    #   节假日：  0301-0616  5:00-10:00  每天有60个样本  一共是33天  ftrainX[33,60,60]  ftrainY[33,60,30]
    # 验证集：  0617-0630  5:00-10:00  每天有60个样本  一共是14天  valX[14,60,60]  valY[14,60,30]
    # 验证子集：0617-0630  6:00-9:00  每天有1个样本  一共是14天  testX[14,60]  testY[14,30]
    # 测试集：  0701-0731  6:00-8:00  每天有1个样本  一共是31天  PredX[31,60]
    wtrainX,wtrainY,ftrainX,ftrainY,valX,valY,testX,testY,PredX=construct_sample_set(his,cur,work_day)
    wtrainX,wtrainY=wtrainX[:,15:-15],wtrainY[:,15:-15]
    input_node = 30
    output_node = 30
    if Train==0:
        drop_par=[0.8,0.8,0.6]
        n_epoch=100
    else:
        drop_par=[1,1,1]
        n_epoch=1

    if Train == 0:
        batch_size = 60
        train_loss, train_acc, train_n_batch = 0, 0, 0
        val_loss, val_acc, val_n_batch = 0, 0, 0
        for w_f in range(2):
            tf.reset_default_graph()
            keep_prob = tf.placeholder("float32", shape=[3], name='keep_prob')
            x = tf.placeholder(tf.float32, shape=[None, input_node], name='x')
            y_ = tf.placeholder(tf.float32, shape=[None, output_node], name='y_')
            if model_flag==1:
                train_op, cost, y=construct_nn_graph1(Train,x,y_,keep_prob,input_node,output_node)
            else:
                train_op, cost, y=construct_nn_graph2(Train,x,y_,keep_prob,input_node,output_node)
            saver=tf.train.Saver(tf.global_variables())
            # work_day 判断从3月1号到7月31号，每天是工作日还是节假日，工作日=1，节假日=0，返回一个153维的numpy数组
            if w_f==1:
                trainX,trainY=wtrainX.reshape([-1,30]),wtrainY.reshape([-1,30])
                valX2,valY2=valX[work_day[108:122]==1].reshape([-1,30]),valY[work_day[108:122]==1].reshape([-1,30])
            else:
                trainX, trainY = ftrainX.reshape([-1,30]), ftrainY.reshape([-1,30])
                valX2,valY2=valX[work_day[108:122]==0].reshape([-1,30]),valY[work_day[108:122]==0].reshape([-1,30])
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                start_time=time.time()
                for epoch in range(n_epoch):
                    perm0 = np.arange(len(trainX))
                    np.random.shuffle(perm0)
                    trainX = trainX[perm0]
                    trainY = trainY[perm0]
                    start = 0
                    end = start + batch_size
                    while (end < len(trainX)):
                        x1,y1=trainX[start:end], trainY[start:end]
                        sess.run([train_op], feed_dict={x: x1, y_: y1, keep_prob: drop_par})
                        start += batch_size
                        end = start + batch_size

                print('training model %d_%d_%d tooks %fs' % (link_num,model_flag,w_f, time.time() - start_time))

                start = 0
                end = start + batch_size
                while (end < len(trainX)):
                    x1, y1 = trainX[start:end], trainY[start:end]
                    err = sess.run(cost, feed_dict={x: x1, y_: y1, keep_prob: [1, 1, 1]})
                    start += batch_size
                    end = start + batch_size
                    train_loss += err
                    train_n_batch += 1

                start = 0
                end = start + batch_size
                while (end < len(valX2)):
                    x1, y1 = valX2[start:end], valY2[start:end]
                    err = sess.run(cost, feed_dict={x: x1, y_: y1, keep_prob: [1, 1, 1]})
                    start += batch_size
                    end = start + batch_size
                    val_loss += err
                    val_n_batch += 1
                save_path = saver.save(sess,  model_name+str(link_num)+'_'+str(w_f))
                print('Model saved in file : %s ' % save_path)

        print('train loss : ', train_loss / train_n_batch)
        print('val loss : ', val_loss / val_n_batch)
        return np.array([train_loss / train_n_batch,val_loss / val_n_batch])
    elif Train==1:
        #每条路，按照工作日和节假日，统计8:00-9:00每2分钟的信息，对于每2分钟，找到1个数，使得评价函数的代价最小
        day_length=108
        EstY=np.empty([2,30])
        for w_f in range(2):
            his2=his[:day_length,89:121] #[77,32]
            his2=his2[work_day[:day_length]==w_f] #[53,32]
            for slide in range(len(his2[0])-2):
                temp=his2[:,slide:slide+3].flatten() #利用前后6分钟的信息
                EstY[w_f,slide]=find_a_best_number(temp) #[53]
        midian_y=np.empty([14,30],np.float)
        err2=np.empty([14],np.float)
        for w_f in range(2):
            testY2=testY[work_day[108:122]==w_f]
            err2[work_day[108:122]==w_f]=np.mean(np.abs(testY2-EstY[w_f])/testY2,axis=1)
            midian_y[work_day[108:122]==w_f]=EstY[w_f]
        midian_y*=max_his
        err2=np.mean(err2)

        #求出神经网络1的值
        output_y=np.empty([14,30],np.float)
        err=np.empty([14],np.float)
        for w_f in range(2):
            testX2,testY2=testX[work_day[108:122]==w_f],testY[work_day[108:122]==w_f]
            tf.reset_default_graph()
            keep_prob = tf.placeholder("float32", shape=[3], name='keep_prob')
            x = tf.placeholder(tf.float32, shape=[None, input_node], name='x')
            y_ = tf.placeholder(tf.float32, shape=[None, output_node], name='y_')
            train_op, cost, y=construct_nn_graph1(Train,x,y_,keep_prob,input_node,output_node)
            saver=tf.train.Saver(tf.global_variables())
            with tf.Session() as sess:
                saver.restore(sess,model_name1+str(link_num)+'_'+str(w_f))
                x1,y1 = testX2, testY2
                output_y[work_day[108:122]==w_f],err[work_day[108:122]==w_f]=sess.run([y,cost], feed_dict={x: x1,y_: y1, keep_prob: drop_par}) #output_y [1,30]
        output_y=output_y.reshape([14,30])*max_his
        err=np.mean(err)

        #求出神经网络2的值
        output_y3=np.empty([14,30],np.float)
        err3=np.empty([14],np.float)
        for w_f in range(2):
            testX2,testY2=testX[work_day[108:122]==w_f],testY[work_day[108:122]==w_f]
            tf.reset_default_graph()
            keep_prob = tf.placeholder("float32", shape=[3], name='keep_prob')
            x = tf.placeholder(tf.float32, shape=[None, input_node], name='x')
            y_ = tf.placeholder(tf.float32, shape=[None, output_node], name='y_')
            train_op, cost, y=construct_nn_graph2(Train,x,y_,keep_prob,input_node,output_node)
            saver=tf.train.Saver(tf.global_variables())
            with tf.Session() as sess:
                saver.restore(sess,model_name2+str(link_num)+'_'+str(w_f))
                x1,y1 = testX2, testY2
                output_y3[work_day[108:122]==w_f],err3[work_day[108:122]==w_f]=sess.run([y,cost], feed_dict={x: x1,y_: y1, keep_prob: drop_par}) #output_y [1,30]
        output_y3=output_y3.reshape([14,30])*max_his
        err3=np.mean(err3)

        testY*=max_his
        print(err)
        print(err3)
        print(err2)
        plt.grid()
        plt.plot(output_y.flatten(),label='prediction')
        plt.plot(output_y3.flatten(),label='prediction2')
        plt.plot(midian_y.flatten(),label='midian')
        plt.plot(testY.flatten(),label='true')
        np.save('road_'+str(link_num)+'.npy',testY.flatten())
        plt.legend()
        plt.show()
        return np.array([err,err3,err2]),output_y,output_y3,midian_y
    else:
        #每条路，按照工作日和节假日，统计8:00-9:00每2分钟的信息，对于每2分钟，找到1个数，使得评价函数的代价最小
        day_length=108
        EstY=np.empty([2,30])
        for w_f in range(2):
            his2=his[:day_length,89:121] #[77,32]
            his2=his2[work_day[:day_length]==w_f] #[53,32]
            for slide in range(len(his2[0])-2):
                temp=his2[:,slide:slide+3].flatten() #利用前后6分钟的信息
                EstY[w_f,slide]=find_a_best_number(temp) #[53]
        midian_y=np.empty([31,30],np.float)
        for w_f in range(2):
            midian_y[work_day[122:]==w_f]=EstY[w_f] #7月份的
        midian_y*=max_his

        #求出神经网络1的值
        output_y=np.empty([31,30],np.float)
        for w_f in range(2):
            PredX2=PredX[work_day[122:]==w_f]
            tf.reset_default_graph()
            keep_prob = tf.placeholder("float32", shape=[3], name='keep_prob')
            x = tf.placeholder(tf.float32, shape=[None, input_node], name='x')
            y_ = tf.placeholder(tf.float32, shape=[None, output_node], name='y_')
            train_op, cost, y=construct_nn_graph1(Train,x,y_,keep_prob,input_node,output_node)
            saver=tf.train.Saver(tf.global_variables())
            with tf.Session() as sess:
                saver.restore(sess,model_name1+str(link_num)+'_'+str(w_f))
                x1 = PredX2
                output_y[work_day[122:]==w_f]=sess.run(y, feed_dict={x: x1, keep_prob: drop_par}) #output_y [1,30]
        output_y=output_y.reshape([31,30])*max_his

        #求出神经网络2的值
        output_y3=np.empty([31,30],np.float)
        for w_f in range(2):
            PredX2=PredX[work_day[122:]==w_f]
            tf.reset_default_graph()
            keep_prob = tf.placeholder("float32", shape=[3], name='keep_prob')
            x = tf.placeholder(tf.float32, shape=[None, input_node], name='x')
            y_ = tf.placeholder(tf.float32, shape=[None, output_node], name='y_')
            train_op, cost, y=construct_nn_graph2(Train,x,y_,keep_prob,input_node,output_node)
            saver=tf.train.Saver(tf.global_variables())
            with tf.Session() as sess:
                saver.restore(sess,model_name2+str(link_num)+'_'+str(w_f))
                x1 = PredX2
                output_y3[work_day[122:]==w_f]=sess.run(y, feed_dict={x: x1, keep_prob: drop_par}) #output_y [1,30]
        output_y3=output_y3.reshape([31,30])*max_his
        return output_y,output_y3,midian_y

def train_neural_network(data_file,model_name1,model_name2):
    loss1=np.empty([132,2],np.float)
    for link_num in range(0,132):
        print(link_num)
        loss1[link_num]=ffnn_main(datafile=data_file,model_name=model_name1,link_num=link_num,Train=0,model_flag=1)
    loss2=np.empty([132,2],np.float)
    for link_num in range(0,132):
        print(link_num)
        loss2[link_num]=ffnn_main(datafile=data_file,model_name=model_name2,link_num=link_num,Train=0,model_flag=2)
    print(np.mean(loss1[:,0]))
    print(np.mean(loss1[:,1]))
    print(np.mean(loss2[:,0]))
    print(np.mean(loss2[:,1]))
    plt.grid()
    plt.plot(loss1[:,0], label='model1 train err')
    plt.plot(loss1[:,1], label='model1 val err')
    plt.plot(loss2[:,0], label='model2 train err')
    plt.plot(loss2[:,1], label='model2 val err')
    plt.legend()
    plt.show()

def validation_neural_network(data_file,model_name1,model_name2,loss_file,savefile,true_data_file):
    prediction = np.empty((14, 30, 132), np.float)  # 预测值，6月17号-6月30号，8:00-9:00的数据，一共是14天，30个时间片，132条Link，15*30*132
    loss = np.empty([132,3], np.float) #神经网络1输出的loss，神经网络2输出的loss 和 中位数输出的loss
    for link_num in range(132):
        print(link_num)
        loss[link_num],output_y,output_y3,midian_y=ffnn_main(datafile=data_file,model_name=model_name1,link_num=link_num,Train=1,model_name2=model_name2) # output_y[15,30]
        if np.min([loss[link_num,0],loss[link_num,1]])<loss[link_num,2]+0.001:
            if loss[link_num,0]<loss[link_num,1]:
                prediction[:, :,link_num] = output_y
            else:
                prediction[:, :,link_num] = output_y3
        else:
            prediction[:, :,link_num] = midian_y
    np.savez(loss_file, loss)
    mix_loss=np.min(loss,axis=1)
    print(np.mean(loss[:,0]))
    print(np.mean(loss[:,1]))
    print(np.mean(loss[:,2]))
    print(np.mean(mix_loss))
    a=mix_loss.argsort()[-6:][::-1]
    print(a)
    print(mix_loss[a])
    plt.grid()
    plt.plot(loss[:,0], label='nn1 err')
    plt.plot(loss[:,1], label='nn2 err')
    plt.plot(loss[:,2], label='midian err')
    plt.plot(mix_loss, label='mix err')
    plt.legend()
    plt.show()
    # 将6月份的结果保存为 .txt 文件
    filename = savefile
    knt = 0
    with open(filename, 'w') as f:
        for link in range(132):
            for day in range(16, 30):
                for minute in range(0, 60, 2):
                    if knt == 0:
                        knt += 1
                        temp = ''
                    else:
                        temp = '\n'
                    temp += Links2[link] + '#'
                    date = '2017-06-' + str(day + 1).zfill(2)
                    temp += date + '#[' + date + ' 08:' + str(minute).zfill(2) + ':00,'
                    if minute == 58:
                        temp += date + ' 09:00:00)#'
                    else:
                        temp += date + ' 08:' + str(minute + 2).zfill(2) + ':00)#'
                    temp += str(prediction[day - 16, int(minute / 2), link])
                    f.write(temp)
    yun_estimate_result2.estimate_result(true_data_file,
                                             filename,
                                             '0617', '08:00-09:00', 14)

def predition_neural_network(data_file,model_name1,model_name2,loss_file,savefile,true_data_file):
    temp = np.load(loss_file)
    loss = temp['arr_0']
    prediction = np.empty((31, 30, 132), np.float)  # 预测值，7月1号-7月31号，8:00-9:00的数据，一共是31天，30个时间片，132条Link，30*30*132

    for link_num in range(0,132):
        print(link_num)
        output_y,output_y3,midian_y=ffnn_main(datafile=data_file,model_name=model_name1,link_num=link_num,Train=2,model_name2=model_name2) # output_y[30,30]
        if np.min([loss[link_num,0],loss[link_num,1]])<loss[link_num,2]+0.001:
            if loss[link_num,0]<loss[link_num,1]:
                prediction[:, :,link_num] = output_y
            else:
                prediction[:, :,link_num] = output_y3
        else:
            prediction[:, :,link_num] = midian_y

    # 将7月份的结果保存为 .txt 文件
    filename = savefile
    knt = 0
    with open(filename, 'w') as f:
        for link in range(132):
            for day in range(31):
                for minute in range(0, 60, 2):
                    if knt == 0:
                        knt += 1
                        temp = ''
                    else:
                        temp = '\n'
                    temp += Links2[link] + '#'
                    date = '2017-07-' + str(day + 1).zfill(2)
                    temp += date + '#[' + date + ' 08:' + str(minute).zfill(2) + ':00,'
                    if minute == 58:
                        temp += date + ' 09:00:00)#'
                    else:
                        temp += date + ' 08:' + str(minute + 2).zfill(2) + ':00)#'
                    temp += str(prediction[day, int(minute / 2), link])
                    f.write(temp)
    # yun_estimate_result2.estimate_result(true_data_file,
    #                                     filename,
    #                                     '0701', '08:00-09:00', 31)

if __name__=='__main__':
    model_name1='./model1/ffnn_model'
    model_name2='./model2/ffnn_model'
    data_file='./data/yun_match_data'
    loss_file='./nn_median_loss.npz'
    savefile_half_6='yun_20170903_evaluate_half_6.txt'
    true_data_half_6='yun_true_data_half_6.txt'
    savefile_7='yun_20170903_evaluate_7_6_8.txt'
    true_data_7='gy_contest_result_template.txt'
    regenerate_data=0
    Train=0 # 0 训练 1 验证 2 预测

    if regenerate_data:
        # 将历史数据和当前数据提取并保存
        data=readdata('yun_complement_dataset_3_7.txt') #假设历史数据集是完备的
        print(len(data))
        print(len(data)/132)
        for i in range(132):
            print(i)
            data2=data[93420*i:93420*(i+1)]
            his,cur=get_history(data2)
            np.savez(data_file+str(i),his,cur)
    else:
        if Train==0:
            train_neural_network(data_file,model_name1,model_name2)
        elif Train==1:
            validation_neural_network(data_file,model_name1,model_name2,loss_file,savefile_half_6,true_data_half_6)
        elif Train==2:
            predition_neural_network(data_file,model_name1,model_name2,loss_file,savefile_7,true_data_7)






























































