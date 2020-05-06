#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 01:09:40 2020

@author: kishor
"""
from sklearn import linear_model
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
class AuthorClassifier:
    def __init__(self):
        self.theta=None
        self.clf=None
    def train(self,path):
        df = pd.read_csv(path)
        text=df['text']
        label=df['author']
        text=text.values
        label=label.values
        tf_idf_vectorizor = TfidfVectorizer(stop_words = 'english',strip_accents='unicode')
        tf_idf=tf_idf_vectorizor.fit_transform(text)
        X_train=tf_idf
        self.clf=linear_model.SGDClassifier(max_iter=1000, tol=1e-3,n_jobs=-1)
        self.clf.fit(X_train,label)
    def predict(self,path):
        df = pd.read_csv(path)
        text=df['text']
        text=text.values
        tf_idf_vectorizor = TfidfVectorizer(stop_words = 'english',strip_accents='unicode')
        tf_idf=tf_idf_vectorizor.fit_transform(text)
        X_test=tf_idf
        prediction=self.clf.predict(X_test)
        return prediction
