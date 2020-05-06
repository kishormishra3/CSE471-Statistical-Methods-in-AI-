#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 00:27:45 2020

@author: kishor
"""
import pandas as pd
import numpy as np
from random import random
from sklearn.preprocessing import MinMaxScaler

class Weather:
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
        df = pd.read_csv(path)
        df = df.rename({"Apparent Temperature (C)": "output"}, axis=1)
        label=df['output'].values
        df=df.drop(["Formatted Date","Daily Summary","output"],axis=1)
        mode_masType = df['Precip Type'].mode()[0]
        df = df.fillna({'Precip Type':mode_masType})
        
        mode_masType = df['Summary'].mode()[0]
        df = df.fillna({'Summary':mode_masType})
        
        
        mode_masType = df['Temperature (C)'].mean()
        df = df.fillna({'Temperature (C)':mode_masType})
        
        mode_masType = df['Humidity'].mean()
        df = df.fillna({'Humidity':mode_masType})
        
        mode_masType = df['Wind Speed (km/h)'].mean()
        df = df.fillna({'Wind Speed (km/h)':mode_masType})
        
        mode_masType = df['Wind Bearing (degrees)'].mean()
        df = df.fillna({'Wind Bearing (degrees)':mode_masType})
        
        mode_masType = df['Visibility (km)'].mean()
        df = df.fillna({'Visibility (km)':mode_masType})
        
        mode_masType = df['Pressure (millibars)'].mean()
        df = df.fillna({'Pressure (millibars)':mode_masType})
        
        
        df = pd.concat([df,pd.get_dummies(df['Summary'], prefix='Summary')],axis=1)
        df = pd.concat([df,pd.get_dummies(df['Precip Type'], prefix='Precip Type')],axis=1)
        df.drop(['Summary','Precip Type'],axis=1, inplace=True)
        scaler = MinMaxScaler()
        data=df.values
        scaler.fit(data)
        data=scaler.transform(data)
        df=pd.DataFrame(data)
        df.insert(0,'One_Padding',1)
        data=df.values
        for i in range(df.shape[1]):
            self.theta.append(random())
        self.theta=np.array(self.theta).reshape([1,len(self.theta)])
        label=label.reshape([label.shape[0],1])
        k=True
        old=0
        for i in range(6000):
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
        df=pd.read_csv(path)
        df=df.drop(["Formatted Date","Daily Summary"],axis=1)
        mode_masType = df['Precip Type'].mode()[0]
        df = df.fillna({'Precip Type':mode_masType})
        
        mode_masType = df['Summary'].mode()[0]
        df = df.fillna({'Summary':mode_masType})
        
        
        mode_masType = df['Temperature (C)'].mean()
        df = df.fillna({'Temperature (C)':mode_masType})
        
        mode_masType = df['Humidity'].mean()
        df = df.fillna({'Humidity':mode_masType})
        
        mode_masType = df['Wind Speed (km/h)'].mean()
        df = df.fillna({'Wind Speed (km/h)':mode_masType})
        
        mode_masType = df['Wind Bearing (degrees)'].mean()
        df = df.fillna({'Wind Bearing (degrees)':mode_masType})
        
        mode_masType = df['Visibility (km)'].mean()
        df = df.fillna({'Visibility (km)':mode_masType})
        
        mode_masType = df['Pressure (millibars)'].mean()
        df = df.fillna({'Pressure (millibars)':mode_masType})
        
        
        df = pd.concat([df,pd.get_dummies(df['Summary'], prefix='Summary')],axis=1)
        df = pd.concat([df,pd.get_dummies(df['Precip Type'], prefix='Precip Type')],axis=1)
        df.drop(['Summary','Precip Type'],axis=1, inplace=True)
        scaler = MinMaxScaler()
        data=df.values
        scaler.fit(data)
        data=scaler.transform(data)
        df=pd.DataFrame(data)
        df.insert(0,'One_Padding',1)
        data=df.values
        value=np.dot(self.theta,np.transpose(data))
        return value
