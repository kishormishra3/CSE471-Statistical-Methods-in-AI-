import numpy as np
import os
import sys
import cv2
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
def flattern(data):
  data_f=[]
  for i in range(data.shape[0]):
    data_f.append(data[i].flatten())
  return np.array(data_f)
class PCA_:
    def __init__(self,C=100):
        self.C=C
    def transform(self,data):
        cov=np.cov(data.T)
        eig_val, eig_vec= np.linalg.eig(cov)
        idx = eig_val.argsort()[::-1]   
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:,idx]
        m= np.real(eig_vec[:,0:self.C])
        XX=np.dot(data,m)
        return XX

data=[]
label=[]
dim=(100,100)
file_path=sys.argv[1]
file=open(file_path)
s=file.readline()
while(s):
  s=s.split()
  img=cv2.imread(s[0],cv2.IMREAD_GRAYSCALE)
  resized = cv2.resize(img,dim, interpolation = cv2.INTER_AREA)
  data.append(resized)
  label.append((s[1]))
  s=file.readline()
data=np.array(data)
Y_train=np.array(label)
data=flattern(data)
data=PCA_(100).transform(data)
one=np.ones(data.shape[0])
data=preprocessing.scale(data)
one=one.reshape([data.shape[0],1])
X_train=np.append(data,one,axis=1)
scaler = MinMaxScaler()


def sigmoid(Z):
  scaler.fit(Z)
  Z=scaler.transform(Z)
  return 1/(1+np.exp(-Z))
def loss(h,y):
  m=len(y)
  cost = 1 / m * np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h))
  return cost


def grad(theta,pre,label,X,al):
  alpha=al
  error=(pre-label)
  theta=theta - alpha * (np.dot((X.T),error)/label.shape[0])
  return theta

classes=np.unique(Y_train)
d={}
for cl in classes:
  k=True
  theta=np.random.rand(X_train.shape[1],1)
  #theta=np.ones([X_train.shape[1],1])
  Z = np.dot(X_train,theta)
  value=sigmoid(Z)
  ty=[]
  new_theta=0
  for i in Y_train:
    if i ==cl:
      ty.append(1)
    else:
      ty.append(0)
  ty=np.array(ty)
  ty=ty.reshape([len(Y_train),1])
  for i in range(5000):
    theta=grad(theta,value,ty,X_train,0.1)
    Z = np.dot(X_train,theta)
    value=sigmoid(Z)
    if k:
      cost=loss(value,ty)
      new_theta=theta.copy()
      k=False
    else:
      new=loss(value,ty)
      if(new > cost):
        break
      new_theta=theta.copy()
      cost=new
  d[cl]=new_theta



data=[]
label=[]
dim=(100,100)
file_path=sys.argv[2]
file=open(file_path)
s=file.readline()
while(s):
  s=s[:-1]
  img=cv2.imread(s,cv2.IMREAD_GRAYSCALE)
  resized = cv2.resize(img,dim, interpolation = cv2.INTER_AREA)
  data.append(resized)
  s=file.readline()
data=np.array(data)
data=flattern(data)
data=PCA_(100).transform(data)
one=np.ones(data.shape[0])
data=preprocessing.scale(data)
one=one.reshape([data.shape[0],1])
X_test=np.append(data,one,axis=1)

index=list(d.keys())
for i in X_test:
  v=[]
  for j in d.values():
    v.append(np.dot(i.T,j))
  print(index[v.index(max(v))])