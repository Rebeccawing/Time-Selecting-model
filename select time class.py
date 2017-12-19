#!/usr/bin/env python
#_*_ coding:utf-8 _*_
# python 2.7.12
#崔诗颖  2017.8.28

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xlrd
import sys
import os
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

class SelectTime(object):
    clo = 0
    high = 0
    low = 0
    date = 0

    def __init__(self, clo, high, low, date):
        self.clo = clo
        self.high = high
        self.low = low
        self.date = date
        print "Initialization has done!"

    def get_ud(self, n1,n2):                                          #算出Ud序列，并找出其中值为+-n2的地点，将
        self.ud = self.clo.copy()          #地址存入到买入启动buy_start和卖出启动sell_start中
        ud1 = self.ud[n1:]
        ud3 = ud1.copy()
        ud2 = self.ud[:-n1]                                           #注意算法中提到的第一根k线，指的就是开始计数时的
        ud2.index = ud1.index
        ud3[ud1>ud2]=1
        ud3[ud1<ud2]=-1
        ud3[ud1==ud2]=0
        ud4 = np.zeros((n1,), dtype=np.int32)
        self.ud = np.append(ud4,ud3)
        self.ud = self.ud.astype(np.int32)
        self.ud_sum = self.ud.copy()
        for i in range(n1,len(self.clo)):
            if self.ud[i]==self.ud[i-1]:
                self.ud_sum[i]=self.ud_sum[i-1]+self.ud[i]
            else:
                self.ud_sum[i]=self.ud[i]
        self.buy_start = np.zeros((len(self.clo),), dtype=np.int32)
        self.sell_start = np.zeros((len(self.clo),), dtype=np.int32)
        k1=k2=0
        for i in range(0,len(self.clo)):
            if self.ud_sum[i]==-n2:
                self.buy_start[k1] = i
                k1 = k1+1
            elif self.ud_sum[i]==n2:
                self.sell_start[k2] = i
                k2 = k2+1
            else:
                pass
        self.buy_start = self.buy_start[self.buy_start!=0]
        self.sell_start = self.sell_start[self.sell_start!=0]
        print "Get the enter-in successfully!"

    def start_buyandsell(self,n3):                                   #找到买入和卖出信号开始的时间点，并把它们分别
                                                                     #存到buy_trade_start和sell_trade_start中
        self.buy_trade_start = np.zeros((len(self.clo),), dtype=np.int32) #计数变量记为buy_count和sell_count
        self.buy_stop_loss = np.zeros((len(self.clo),), dtype=np.float16)
        self.sell_trade_start = np.zeros((len(self.clo),), dtype=np.int32)
        self.sell_stop_loss = np.zeros((len(self.clo),), dtype=np.float16)
        k1=k2=0
        for i in range(0,len(self.buy_start)-1):
            self.buy_count = 1
            s = self.buy_start[i]
            t = self.buy_start[i+1]-self.buy_start[i]
            last_cnt = s+1
            for j in range(2,t):
                if ((self.clo[s+j]>=self.high[s+j-2]) and (self.high[s+j]>self.high[s+j-1]) and (self.clo[s+j]>self.clo[last_cnt])):
                    self.buy_count = self.buy_count+1
                    last_cnt = s+j
                if self.buy_count==n3:
                    self.buy_trade_start[k1]=s+j
                    self.buy_stop_loss[k1] = (self.low[s:s+j]).min()
                    k1 = k1+1
                    break
        for i in range(0,len(self.sell_start)-1):
            self.sell_count = 1
            s = self.sell_start[i]
            t = self.sell_start[i+1]-self.sell_start[i]
            last_cnt = s + 1
            for j in range(2,t):
                if (self.clo[s+j]<=self.low[s+j-2]) and (self.low[s+j]<self.low[s+j-1]) and (self.clo[s+j]<self.clo[last_cnt]):
                    self.sell_count = self.sell_count+1
                    last_cnt = s+j
                    if self.sell_count==n3:
                        self.sell_trade_start[k2]=s+j
                        self.sell_stop_loss[k2] = (self.high[s:s+j]).max()
                        k2 = k2+1
                        break
        self.buy_trade_start = self.buy_trade_start[self.buy_trade_start!=0]
        self.sell_trade_start = self.sell_trade_start[self.sell_trade_start!=0]
        self.buy_stop_loss = self.buy_stop_loss[self.buy_stop_loss!=0]
        self.sell_stop_loss = self.sell_stop_loss[self.sell_stop_loss!=0]
        print "Get the start successfully!"

    def stop_buyandsell(self):                                      #记录交易结束时间，这里有两种机制，第一种是出现
        self.sell_trade_stop = np.zeros((len(self.sell_trade_start),), dtype=np.int32)   #反向信号时，第二种是止损点位
        self.buy_trade_stop = np.zeros((len(self.buy_trade_start),), dtype=np.int32)
        s1 = self.sell_trade_start.copy()
        s2 = self.buy_trade_start.copy()
        a = self.sell_trade_start[-1:]
        b=0
        for i in range(0,len(self.buy_trade_start)):
            if self.buy_trade_start[i]<a:
                b = b+1
        for i in range(0,b):
            t2 = 3000
            for j in range(0,len(self.sell_trade_start)):
                s1[j] = self.sell_trade_start[j]-self.buy_trade_start[i]
            min_p = s1[s1 >0].min()
            t1 = np.where(s1 == min_p)[0][0]
            for j in range(self.buy_trade_start[i],len(self.clo)):
                if self.low[j]<=self.buy_stop_loss[i]:
                    t2=j
                    break
            self.buy_trade_stop[i]=min(self.sell_trade_start[t1],t2)
        i = 0

        a = self.buy_trade_start[-1:]
        b = 0
        for i in range(0,len(self.sell_trade_start)):
            if self.sell_trade_start[i]<a:
                b = b+1
        for i in range(0,b):
            for j in range(0,len(self.buy_trade_start)):
                s2[j] = self.buy_trade_start[j]-self.sell_trade_start[i]
            min_p = s2[s2>0].min()
            t1 = np.where(s2==min_p)[0][0]
            for j in range(self.sell_trade_start[i],len(self.clo)):
                if self.high[j]>=self.sell_stop_loss[i]:
                    t2 = j
                    break
            self.sell_trade_stop[i]=min(self.buy_trade_start[t1],t2)
        i = 0
        while i < len(self.sell_trade_start):
            end = self.sell_trade_stop[i]
            i = i+1
            while(i<len(self.sell_trade_start) and self.sell_trade_start[i]<=end):
                self.sell_trade_start[i] = -1
                self.sell_trade_stop[i] = -1
                i = i+1
        while i < len(self.buy_trade_start):
            end = self.buy_trade_stop[i]
            i = i+1
            while(i<len(self.buy_trade_start) and self.buy_trade_start[i]<=end):
                self.buy_trade_start[i] = -1
                self.buy_trade_stop[i] = -1
                i = i+1
        self.sell_trade_stop = self.sell_trade_stop[self.sell_trade_stop != -1]
        self.buy_trade_stop = self.buy_trade_stop[self.buy_trade_stop != -1]
        self.sell_trade_start = self.sell_trade_start[self.sell_trade_start != -1]
        self.buy_trade_start = self.buy_trade_start[self.buy_trade_start != -1]
        for i in range(0,len(self.buy_trade_start)):
            if self.buy_trade_stop[i]==0:
                self.buy_trade_start[i]=0
        self.buy_trade_stop = self.buy_trade_stop[self.buy_trade_stop != 0]
        self.buy_trade_start = self.buy_trade_start[self.buy_trade_start != 0]
        print "Get the stop successfully!"

    def assess_model(self,n1,n2,n3,FACname):
        self.start_trade = np.append(self.buy_trade_start,self.sell_trade_start)
        self.stop_trade = np.append(self.buy_trade_stop,self.sell_trade_stop)
        self.start_trade = sorted(self.start_trade)
        self.stop_trade = sorted(self.stop_trade)
#将其时间对应到每一年
        self.start_trade = pd.Series(self.start_trade)
        self.stop_trade = pd.Series(self.stop_trade)
        fake_date = self.stop_trade.copy()
        for i in range(0,len(self.stop_trade)):
            s = self.stop_trade[i]
            fake_date.iloc[i] = self.date.iloc[s]
        self.stop_trade.index = fake_date
        self.start_trade.index = self.stop_trade.index
        self.trade_return = self.start_trade.copy()
#算每一次交易的年化收益率和累计收益率
        for i in range(0,len(self.start_trade)):
            clo_start = self.start_trade.iloc[i]
            clo_stop = self.stop_trade.iloc[i]
            self.trade_return.iloc[i]=self.clo[clo_stop]/self.clo[clo_start]-1
        self.trade_return = self.trade_return+1
        self.cum_trade_return = self.trade_return.cumprod()
        self.cum_trade_return = self.cum_trade_return - 1
        self.ann_return = self.trade_return.groupby(level=0).prod()
        self.cum_return = self.ann_return.cumprod()
        self.ann_return = self.ann_return - 1
        self.cum_return =self.cum_return - 1
        self.tot_ann_return = self.trade_return.prod()-1
        self.tot_cum_return = self.cum_return.iloc[-1:]
        self.tot_cum_return.index = ['全样本']
        self.ann_return = pd.Series([self.tot_ann_return], index=['全样本']).append(self.ann_return)
        self.cum_return = self.tot_cum_return.append(self.cum_return)
#算总交易次数
        self.trade_count = self.trade_return.groupby(level=0).count()
        self.tot_trade_count = self.trade_return.count()
        self.trade_count = pd.Series([self.tot_trade_count], index=['全样本']).append(self.trade_count)
#算获胜次数和失败次数
        self.trade_01 = self.trade_return.copy()
        self.trade_01[self.trade_return>=1]=1
        self.trade_01[self.trade_return<1]=2
        def wincount(ser):
            try:
                a = ser.agg(lambda x:x.value_counts().loc[1])
            except:
                a = 0
            return a
        self.win_count = self.trade_01.groupby(level=0).agg(wincount)
        def losecount(ser):
            try:
                a = ser.agg(lambda x: x.value_counts().loc[2])
            except:
                a = 0
            return a
        self.lose_count = self.trade_01.groupby(level=0).agg(losecount)
        self.tot_win_count = self.trade_01.agg(lambda x:x.value_counts().loc[1])
        self.tot_lose_count = self.trade_01.agg(lambda x:x.value_counts().loc[2])
        self.win_count = pd.Series([self.tot_win_count], index=['全样本']).append(self.win_count)
        self.lose_count = pd.Series([self.tot_win_count], index=['全样本']).append(self.lose_count)
#算胜率
        self.win_ratio = self.win_count/self.trade_count
#算单次平均收益率
        self.trade_return = self.trade_return-1
        self.ave_return = self.trade_return.groupby(level=0).mean()
        self.tot_ave_return = self.trade_return.mean()
        self.ave_return = pd.Series([self.tot_ave_return], index=['全样本']).append(self.ave_return)
#算单次获胜收益率和失败收益率
        self.win_ave_return = (self.trade_return[self.trade_return>=0]).groupby(level=0).mean()
        self.tot_win_ave_ret = (self.trade_return[self.trade_return>=0]).mean()
        self.lose_ave_return = (self.trade_return[self.trade_return<0]).groupby(level=0).mean()
        self.tot_lose_ave_ret = (self.trade_return[self.trade_return<0]).mean()
        self.win_ave_return = pd.Series([self.tot_win_ave_ret], index=['全样本']).append(self.win_ave_return)
        self.lose_ave_return = pd.Series([self.tot_lose_ave_ret], index=['全样本']).append(self.lose_ave_return)
#算赔率
        self.odds = self.win_ave_return/self.lose_ave_return.abs()
#算最大回撤，连续累计收益率，间断累计收益率
        start = []
        end = []
        value = [np.nan]*len(self.clo)                         #资产总值
        self.draw_down = [np.nan]*len(self.clo)                #最大回撤
        self.cum_return1 = [np.nan]*len(self.clo)              #连续累计收益率
        self.cum_return2 = [np.nan] * len(self.clo)            #间断累计收益率
        next_start = 0
        next_end = 0
        value[0] = 1
        state = 0
        at = pd.Series([-1],index=['2017'])
        self.start_trade = self.start_trade.append(at)
        self.stop_trade = self.stop_trade.append(at)
        #最大回撤函数
        def drawdown(ser):
            max_draw_down = 0
            temp_max_value = 0
            for i in ser.index.get_level_values(level=1):
                temp_max_value = max(temp_max_value, value[i - 1])
                max_draw_down = min(max_draw_down, value[i] / temp_max_value - 1)
            return max_draw_down
        #算累计收益率
        for i in range(1, len(self.clo)):
            if (i == self.start_trade.iloc[next_start]):
                state = 1
                value[i] = value[i-1] * (self.clo[i]/self.clo[i-1])
                self.cum_return1[i] = self.clo[i]/self.clo[i-1]
                next_start = next_start+1
                continue
            if (i == self.stop_trade.iloc[next_end]):
                state = 0
                value[i] = value[i-1] * (self.clo[i] / self.clo[i-1])
                self.cum_return1[i] = self.clo[i]/self.clo[i-1]
                continue
            if (state == 0):
                value[i] = value[i-1]
                self.cum_return1[i] = 1
            if (state == 1):
                value[i] = value[i-1] * (self.clo[i] / self.clo[i-1])
                self.cum_return1[i] = self.clo[i]/self.clo[i-1]
        self.cum_return1 = pd.Series(self.cum_return1,index = self.clo.index)
        self.cum_return1 = self.cum_return1.cumprod()
        self.cum_return1 = self.cum_return1-1
        w = self.start_trade.iloc[0]
        for i in range(0,w):
            self.cum_return2[i] = 0
        next_start = 0
        for i in range(w,len(self.clo)):
            if (i == self.start_trade.iloc[next_start]):
                self.cum_return2[i] = self.cum_trade_return.iloc[next_start]
                next_start = next_start+1
            else:
                self.cum_return2[i]=self.cum_return2[i-1]
        self.cum_return2 = pd.Series(self.cum_return2, index=self.clo.index)
        #算最大回撤
        value = pd.Series(value,index = self.clo.index)               #资产总值
        self.ann_draw_down = value.groupby(level=0).agg(drawdown)
        next_start = 0
        for i in range(0,w):
            self.draw_down[i] = 0
        for i in range(w,len(self.clo)):
            if (state == 1):
                a = self.start_trade.iloc[next_start-1]
                b = value.iloc[a:i]
                self.draw_down[i] = b.agg(drawdown)
            if (i == self.start_trade.iloc[next_start]):
                state = 1
                self.draw_down[i]=0
                next_start = next_start+1
            if (i == self.stop_trade.iloc[next_end]):
                a = self.start_trade.iloc[0]
                b = value.iloc[a:i]
                state = 0
                self.draw_down[i] = b.agg(drawdown)
            if (state == 0):
                self.draw_down[i] = 0
        self.draw_down = pd.Series(self.draw_down, index=self.clo.index)
        self.tot_ann_draw_down_null = self.clo.agg(drawdown)
        self.tot_ann_draw_down = value.agg(drawdown)
        self.ann_draw_down = pd.Series([self.tot_ann_draw_down], index=['全样本']).append(self.ann_draw_down)
        #self.ann_draw_down.index = self.cum_return.index

#算连续获胜和失败次数
        self.trade_01 = self.trade_01.astype('int')
        self.str_trade_01 = self.trade_01.groupby(level=0).agg(lambda x: ''.join(x.astype('str')))
        self.continuous_win = self.str_trade_01.copy()
        self.continuous_lose = self.str_trade_01.copy()
        for i in range(0,len(self.str_trade_01)):
            a = self.str_trade_01.iloc[i].split('1')
            b = self.str_trade_01.iloc[i].split('2')
            self.continuous_win.iloc[i] = max([len(x) for x in a])
            self.continuous_lose.iloc[i] = max([len(x) for x in b])
        a = ''.join(self.trade_01.astype('str'))
        b = a.split('1')
        self.tot_continuous_win = max([len(x) for x in b])
        c = a.split('2')
        self.tot_continuous_lose = max([len(x) for x in c])
        self.continuous_win = pd.Series([self.tot_continuous_win], index=['全样本']).append(self.continuous_win)
        self.continuous_lose = pd.Series([self.tot_continuous_lose], index=['全样本']).append(self.continuous_lose)

#对每个不同的n1,n2,n3取值创建不同的文件夹
        path = "C:\\DELL\\quantitative research\\second_test\\result\\"
        n_123 = ''.join([str(n) for n in (n1, n2, n3)])
        path = path + FACname + '\\_' + n_123
        if os.path.exists(path):
            pass
        else:
            os.mkdir(path)

#画图
        #画出第一种连续的累计收益率
        cumret = pd.DataFrame([self.clo, self.cum_return1])
        cumret = cumret.T
        cumret.index = cumret.index.get_level_values(0)
        cumret = cumret.rename(columns={cumret.columns[1]: u'cumret1'})
        cumret.plot(kind='line',secondary_y=[u'cumret1'])
        plt.title(r'cum_return1')
        plt.savefig(path + "\\cum_ret1.png")
        plt.close()
        print "cum_return1 is done!"
        #画出最大回撤与大盘指数的图
        drawdown = pd.DataFrame([self.clo,self.draw_down])
        drawdown.index = drawdown.index.get_level_values(0)
        drawdown = drawdown.rename(index={drawdown.index[1]: u'maxdrawdown'})
        drawdown = drawdown.T
        drawdown.plot(kind='line', secondary_y=[u'maxdrawdown'])
        plt.title(r'maximum_drawdown')
        plt.savefig(path + "\\max_drawdown.png")
        plt.close()
        print "max drawdown is done!"
        #画出第二种间断的累计收益率
        cumret2 = pd.DataFrame([self.clo, self.cum_return2])
        cumret2 = cumret2.T
        cumret2.index = cumret2.index.get_level_values(0)
        cumret2 = cumret2.rename(columns={cumret2.columns[1]: u'cumret2'})
        cumret2.plot(kind='line', secondary_y=[u'cumret2'])
        plt.title(r'cum_return2')
        plt.savefig(path + "\\cum_ret2.png")
        plt.close()
        print "cum_return1 is done!"

#建立dataframe
        self.df = pd.DataFrame(index=self.cum_return.index)
        self.df[u'累计收益率'] = self.cum_return
        self.df[u'年化收益率'] = self.ann_return
        self.df[u'交易次数'] = self.trade_count
        self.df[u'获胜次数'] = self.win_count
        self.df[u'失败次数'] = self.lose_count
        self.df[u'胜率'] = self.win_ratio
        self.df[u'单次均收益率'] = self.ave_return
        self.df[u'单次获胜均收益率'] = self.win_ave_return
        self.df[u'单次失败均收益率'] = self.lose_ave_return
        self.df[u'赔率'] = self.odds
        self.df[u'最大回撤'] = self.ann_draw_down
        self.df[u'最大连胜次数'] = self.continuous_win
        self.df[u'最大连败次数'] = self.continuous_lose
        self.df = self.df.T
        self.df.to_csv(path + "Assess.csv",encoding = 'gbk')
        print "Get the access csv successfully!"

#注意：k反应了投资者的偏好，如果对高收益更偏好，则k值越小，反之k值越大
    def assess_func(self,k):                                                       #无量纲化与无策略时相比较
        x1 = (self.tot_ann_return+1)/(self.clo.ix[-1]/self.clo.ix[0])
        x2 = self.tot_ann_draw_down / self.tot_ann_draw_down_null
        self.assess = x1 + x2 * k

    def cal_all(self,n1,n2,n3,FACname,k):
        self.get_ud(n1,n2)
        self.start_buyandsell(n3)
        self.stop_buyandsell()
        self.assess_model(n1,n2,n3,FACname)
        self.assess_func(k)
        print "Finished!"

#数据预处理部分
df = pd.ExcelFile(r'C:\\DELL\\quantitative research\\second_test\\indices_ohlc.xlsx')

#收盘价数据
df_close = pd.read_excel(df,'close')
df_close[u'DATE'] = df_close[u'DATE'].astype('str')
df_close[u'DATE'] = pd.DatetimeIndex(df_close[u'DATE']).year
# in-sample selection
df_close_in = df_close[df_close[u'DATE'].isin(range(2006,2015))]
df_close_in[u'DATE'] = df_close_in[u'DATE'].astype('str')
in_date = df_close_in[u'DATE']
df_close_in['id'] = range(len(df_close_in))
df_close_in = df_close_in.set_index([u'DATE', 'id'])
# out-sample selection
df_close_out = df_close[df_close[u'DATE'].isin(range(2015,2017))]
df_close_out[u'DATE'] = df_close_out[u'DATE'].astype('str')
out_date = df_close_out[u'DATE']
df_close_out['id'] = range(len(df_close_out))
df_close_out = df_close_out.set_index([u'DATE', 'id'])

#最高价数据
df_high = pd.read_excel(df,'high')
df_high[u'DATE'] = df_high[u'DATE'].astype('str')
df_high[u'DATE'] = pd.DatetimeIndex(df_high[u'DATE']).year
# in-sample selection
df_high_in = df_high[df_high[u'DATE'].isin(range(2006,2015))]
df_high_in[u'DATE'] = df_high_in[u'DATE'].astype('str')
df_high_in['id'] = range(len(df_high_in))
df_high_in = df_high_in.set_index([u'DATE', 'id'])
# out-sample selection
df_high_out = df_high[df_high[u'DATE'].isin(range(2015,2017))]
df_high_out[u'DATE'] = df_high_out[u'DATE'].astype('str')
df_high_out['id'] = range(len(df_high_out))
df_high_out = df_high_out.set_index([u'DATE', 'id'])

#最低价数据
df_low = pd.read_excel(df,'low')
df_low[u'DATE'] = df_low[u'DATE'].astype(u'str')
df_low[u'DATE'] = pd.DatetimeIndex(df_low[u'DATE']).year
# in-sample selection
df_low_in = df_low[df_low[u'DATE'].isin(range(2006,2015))]
df_low_in[u'DATE'] = df_low_in[u'DATE'].astype('str')
df_low_in['id'] = range(len(df_low_in))
df_low_in = df_low_in.set_index([u'DATE', 'id'])
# out-sample selection
df_low_out = df_low[df_low[u'DATE'].isin(range(2015,2017))]
df_low_out[u'DATE'] = df_low_out[u'DATE'].astype('str')
df_low_out['id'] = range(len(df_low_out))
df_low_out = df_low_out.set_index([u'DATE', 'id'])


close_50_in = df_close_in.ix[:,0]
close_300_in = df_close_in.ix[:,1]
close_500_in = df_close_in.ix[:,2]
close_800_in = df_close_in.ix[:,3]
high_50_in = df_high_in.ix[:,0]
high_300_in = df_high_in.ix[:,1]
high_500_in = df_high_in.ix[:,2]
high_800_in = df_high_in.ix[:,3]
low_50_in = df_low_in.ix[:,0]
low_300_in = df_low_in.ix[:,1]
low_500_in = df_low_in.ix[:,2]
low_800_in = df_low_in.ix[:,3]

close_50_out = df_close_out.ix[:,0]
close_300_out = df_close_out.ix[:,1]
close_500_out = df_close_out.ix[:,2]
close_800_out = df_close_out.ix[:,3]
high_50_out = df_high_out.ix[:,0]
high_300_out = df_high_out.ix[:,1]
high_500_out = df_high_out.ix[:,2]
high_800_out = df_high_out.ix[:,3]
low_50_out = df_low_out.ix[:,0]
low_300_out = df_low_out.ix[:,1]
low_500_out = df_low_out.ix[:,2]
low_800_out = df_low_out.ix[:,3]

SZ_50 = SelectTime(close_50_in, high_50_in, low_50_in, in_date)
for n in ('300', '500', '800'):
    globals()['HS_' + n] = eval('SelectTime(close_%s_in,high_%s_in,low_%s_in,in_date)' % (n, n, n))

#用排序算法找出in-sample中最大assess的分数及相应的n1,n2,n3
a_ans = []
a_n1 = []
a_n2 = []
a_n3 = []

for s in ('SZ_50','HS_300','HS_500','HS_800'):
    for n1 in range(2,7):
        for n2 in range(2,7):
            for n3 in range(2,7):
                eval(s).cal_all(n1,n2,n3,s,1)
                a_ans.append(eval(s).assess)                                #a_ans存储assess function
                a_n1.append(n1)                                             #a_n1,n2,n3分别存储
                a_n2.append(n2)
                a_n3.append(n3)
    length = len(a_ans)
    for i in range(0,length):
        for j in range(i+1,length):
            if(a_ans[i]<a_ans[j]):
                a_ans[i],a_ans[j] = a_ans[j],a_ans[i]
                a_n1[i],a_n1[j] = a_n1[j],a_n1[i]
                a_n2[i],a_n2[j] = a_n2[j],a_n2[i]
                a_n3[i],a_n3[j] = a_n3[j],a_n3[i]
    print a_ans
    print a_n1
    print a_n2
    print a_n3
    a_ans = []
    a_n1 = []
    a_n2 = []
    a_n3 = []
    
#检验各个out-sample的情况
#SZ_50_out = SelectTime(close_50_out, high_50_out, low_50_out, out_date)
#SZ_50_out.cal_all(4,2,5,'SZ_50',1)
#print SZ_50_out.assess
#HS_300_out = SelectTime(close_300_out, high_300_out, low_300_out, out_date)
#HS_300_out.cal_all(4,3,4,'HS_300',1)
#print HS_300_out.assess
#HS_500_out = SelectTime(close_500_out, high_500_out, low_500_out, out_date)
#HS_500_out.cal_all(5,3,4,'HS_500',1)
#print HS_500_out.assess
#HS_800_out = SelectTime(close_800_out, high_800_out, low_800_out, out_date)
#HS_800_out.cal_all(3,2,5,'HS_800',1)
#print HS_800_out.assess
