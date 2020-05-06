#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 23:58:11 2020

@author: kishor
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from random import random

class Airfoil:
    def __init__(self):
        self.theta=None
    def cost_f(self,pre,label):
        sum=0
        for i in range(label.shape[0]):
         sum = sum +(pre[i]-label[i])*(pre[i]-label[i])
        return sum/(2* pre.shape[0])
    def grad(self,theta,pre,label,data):
        alpha=0.1
        error=(pre-label)
        theta=theta - alpha * (np.dot(np.transpose(error),data)/label.shape[0])
        return theta
    def train(self,path):
        df=pd.read_csv(path,header=None)
        label=df.values[:,-1]
        data=df.values[:,:-1]
        scaler = MinMaxScaler()
        scaler.fit(data)
        data=scaler.transform(data)
        df1 = pd.DataFrame(data)
        df1.insert(0,5,1)
        data=df1.values
        self.theta=[random(), random(),random(),random(),random(),random()]
        self.theta=np.array(self.theta).reshape([1,6])
        label=label.reshape([label.shape[0],1])
        k=True
        old=0
        for i in range(5000):
          value=np.dot(self.theta,np.transpose(data))
          value=value.reshape([label.shape[0],1])
          self.theta=self.grad(self.theta,value,label,data)
          value=np.dot(self.theta,np.transpose(data))
          value=value.reshape([label.shape[0],1])
          new = self.cost_f(value,label)
          if(k):
              old=new
              k=False
          elif(old<=new):
              break
          else:
              old=new
    def predict(self,path):
        df=pd.read_csv(path,header=None)
        data=None
        if(df.shape[1]==6):
            data=data=df.values[:,:-1]
        else:
            data=df.values
        scaler = MinMaxScaler()
        scaler.fit(data)
        data=scaler.transform(data)
        df1 = pd.DataFrame(data)
        df1.insert(0,5,1)
        data=df1.values
        value=np.dot(self.theta,np.transpose(data))
        return value
