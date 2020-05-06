#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 16:13:17 2020

@author: kishor
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import numpy as np
import string
import os
from scipy.spatial import distance
class Cluster:
    def __init__(self):
        self.X_train=None
        self.convt={}
    def cluster(self,file_path):
        list1=[]
        counter=0
        for _, _, files in os.walk(file_path):
          for ii in files:
            i=file_path+ii
            f=open(i,"rb")
            k=f.read().decode(errors='replace')
            k = k.replace('\n', ' ')
            k = k.strip('\t')
            f.close()
            k = k.translate(str.maketrans('', '', string.punctuation))
            k = k.lower()
            self.convt[counter]=ii
            counter+=1
            list1.append(k)
        train_d=np.array(list1)
        tf_idf_vectorizor = TfidfVectorizer(stop_words = 'english',strip_accents='unicode')
        tf_idf=tf_idf_vectorizor.fit_transform(train_d)
        tf_idf_norm = normalize(tf_idf)
        self.X_train=tf_idf_norm.toarray()
        self.X_train=np.asarray(self.X_train,dtype=np.float64)
        self.X_train=np.asarray(self.X_train,dtype=np.float64)
        
        k_clusters = 5
        k_means = np.random.uniform(size=(k_clusters,self.X_train.shape[1]))
        for row in range(k_clusters):
            k_means[row,:] /= np.linalg.norm(k_means[row,:])
        k_means.shape
        np.linalg.norm(k_means,axis=1)
        mean=k_means.copy()
        mean=np.asarray(mean,dtype=np.float64)
        old_mean=np.zeros([self.X_train.shape[1]])
        flag=True
        start = 0
        end = 30
        list2=[]
        list1={}
        while start < end :
          list2=[]
          list1=[]
          for i in range(5):
            list2.append([])
            list1.append([])
          kk=1
          for i in range(self.X_train.shape[0]):
            temp=[]
            for j in range(mean.shape[0]):
              temp.append(distance.euclidean(self.X_train[i],mean[j]))
            index= temp.index(min(temp))
            list2[index].append(self.X_train[i])
            list1[index].append(i)
          kk+=1
          for p in range(mean.shape[0]):
            sum=np.zeros([self.X_train.shape[1]])
            sum=np.asarray(sum,dtype=np.float64)
            for ind in list2[p]:
              sum=sum+ind
            mean[p]=sum/len(list2[p])
          start+=1
          if(flag):
            flag = False
          else:
            count=0
            for i in range(5):
              if(np.sum(mean[i])==old_mean[i]):
                count+=1
              old_mean[i]=np.sum(mean[i])
            if(count==5):
              break
        l={}
        for i in range(len(list1)):
          for j in list1[i]:
            name=self.convt[j]
            l[name]=str(i)
        return l
    