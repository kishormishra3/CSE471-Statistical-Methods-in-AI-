import pandas as pd
import numpy as np
from queue import Queue

class Tree:
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None
        
        
class DecisionTree:
    def __init__(self):
        self.feature=[]
        self.data=[]
        self.data2=[]
        self.feature2=[]
        self.data_top=[]
        
    def printt(self,root):
      q = Queue()
      q.put(root)
      while(not q.empty()):
          temp=q.get()
          print(temp.data)
          if(temp.left!=None):
            q.put(temp.left)
          if(temp.right!=None):
            q.put(temp.right)
    
    def feature_compute(self,data):
      unique=20
      feature=[]
      for i in data.columns:
        if i!="label":
          uni=data[i].unique()
          ex=uni[0]
          if(isinstance(ex,str) or len(uni)<=unique):
            feature.append("c")
          else:
            feature.append("r")
      return feature
         
        
    def get_split(self,data):
      d={}
      _,col=data.shape
      for i in range(0,col-1):
        val=data[:,i]
        val=np.unique(val)
        if(self.feature[i]=='c'):
          d[i]=val
        else:
          value=[]
          val=sorted(val)
          for j in range(1,len(val)):
            value.append((val[j]+val[j-1])/2)
          d[i]=value
      return d
    
    def data_split(self,data,col,val):
      col_val=data[:,col]
      if(self.feature[col]=='c'):
        low=data[col_val==val]
        high=data[col_val!=val]
      else:
        low=data[col_val<=val]
        high=data[col_val>val]
      return low,high
    
    def mse(self,data):
      label=data[:,-1]
      if(len(label)==0):
        mse=0
      else:
        p=np.mean(label)
        mse=np.mean((label-p)**2)
      return mse
    
    def getEntropy(self,low,high):
      l=len(low)+len(high)
      total=((len(low)/l)*self.mse(low)+(len(high)/l)*self.mse(high))/l
      return total
    
    def best_split(self,data,ind):
      k = True
      for i in ind:
          for j in ind[i]:
            low,high= self.data_split(data,i,j)
            c_m=self.getEntropy(low,high)
            if(k==True or c_m<b_m):
              b_m=c_m
              col=i
              val=j
              k=False
      return col,val
    
    
    def DecisionTree1(self,data,root,maxdepth=15,counter=0):
      root =Tree()
      if(len(data)<=20 or counter >maxdepth) :
        label=data[:,-1]
        root.data=np.mean(label)
        root.left=None
        root.right=None
        return root
      counter=counter+1
      ind=self.get_split(data)
      col,val= self.best_split(data,ind)
      low,high= self.data_split(data,col,val)
      if(len(low)==0 or len(high)==0):
        label=data[:,-1]
        root.data=np.mean(label)
        root.left=None
        root.right=None
        return root
      if(self.feature[col]=='r'):
        root.data = str(self.data_top[col]) + " <= " + str(val)
      else:
        root.data = str(self.data_top[col]) + " = " + str(val)
      root.left = self.DecisionTree1(low,root.left,maxdepth,counter)
      root.right = self.DecisionTree1(high,root.right,maxdepth,counter)  
      return root
    
    def predict_tree(self,tree,test):
      temp=tree
      if(isinstance(temp.data,float)):
        return temp.data
      if(len(temp.data.split(" "))==4):
        fet,op,val,val2 = temp.data.split(" ")
        val=val+val2
      else:  
        fet,op,val = temp.data.split(" ")
      if(op == "="):
        if(test[fet] == val):
          return self.predict_tree(tree.left,test)
        return self.predict_tree(tree.right,test)
      else:
        if(test[fet] <= float(val)):
          return self.predict_tree(tree.left,test)
      return self.predict_tree(tree.right,test)
    
    def predict(self,path):
        self.data2= pd.read_csv(path)
        self.data2=self.data2.drop(["Id","Alley","PoolQC","MiscFeature","Fence","FireplaceQu"],axis=1)
        self.feature2=self.feature_compute(self.data2)
        m=0
        for i in self.data2.columns:
          if(self.data2[i].isnull().any()==True):
            if(self.feature2[m]=='c'):
              mode_masType = self.data2[i].mode()[0]
              self.data2 = self.data2.fillna({i:mode_masType})
            else:
              median_lot = self.data2[i].mean()
              self.data2 = self.data2.fillna({i:median_lot})
        self.data2=self.data2.replace("C (all)","C(all)")
        p=[]
        for i in range(len(self.data2)):
          p.append(self.predict_tree(self.tree,self.data2.iloc[i]))
        return p
    
    def train(self,path):
        self.data = pd.read_csv(path)
        self.data=self.data.drop(["Id","Alley","PoolQC","MiscFeature","Fence","FireplaceQu"],axis=1)
        self.data = self.data.rename({"SalePrice": "label"}, axis=1)
        m=0
        self.feature=self.feature_compute(self.data)
        for i in self.data.columns:
          if(self.data[i].isnull().any()==True and i != "label"):
            if(self.feature[m]=='c'):
              mode_masType = self.data[i].mode()[0]
              self.data = self.data.fillna({i:mode_masType})
            else:
              median_lot = self.data[i].mean()
              self.data = self.data.fillna({i:median_lot})
        self.data=self.data.replace("C (all)","C(all)")
        self.data
        self.data_top = self.data.columns
        root=0
        self.tree=self.DecisionTree1(self.data.values,root)



