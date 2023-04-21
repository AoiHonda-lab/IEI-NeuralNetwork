import numpy as np
import math
import pandas as pd
import torch
import torch.nn as nn


######################################################################################################################
class PreprocessingLayerPercentile(nn.Module):#5%~95%
    def __init__(self, data, num):
        super().__init__()
        #dataframe creating
        X_train_data = data[:][0].numpy()
        y_train_data = data[:][1].numpy()
        X=pd.DataFrame(data=X_train_data)
        y=pd.DataFrame(data=y_train_data, columns=["target"]) 
        df = pd.concat([X, y], axis=1) 
        df_corr = df.corr() #calculation of correlation coefficients

        X_sort = X.iloc[:,num].sort_values().reset_index()
        alpha = X_sort[num][math.floor(len(X)*0.05)]
        beta = X_sort[num][math.floor(len(X)*0.95)]
             
        if df_corr.loc['target',num] >= 0:# correlation coefficient is positive
            # definition of a matrix to store weights
            weight = torch.FloatTensor([[5.88 / (beta - alpha)]])
            self.weight = nn.Parameter(weight)
            # definition of a vector to store the bias
            bias = torch.FloatTensor([-5.88 * (beta + alpha) / (2 * (beta - alpha))])
            self.bias = nn.Parameter(bias)
        elif df_corr.loc['target',num] < 0:# correlation coefficient is negative
            # definition of a matrix to store weights
            weight = torch.FloatTensor([[-5.88 / (beta - alpha)]])
            self.weight = nn.Parameter(weight)
            # definition of a vector to store the bias
            bias = torch.FloatTensor([5.88 * (beta + alpha) /(2* (beta - alpha))])
            self.bias = nn.Parameter(bias)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)

######################################################################################################################
class PreprocessingLayerStandardDeviation(nn.Module):#StandardDeviation
    def __init__(self, data, num):
        super().__init__()
        #dataframe creating
        X_train_data = data[:][0].numpy()
        y_train_data = data[:][1].numpy()
        X=pd.DataFrame(data=X_train_data)
        y=pd.DataFrame(data=y_train_data, columns=["target"]) 
        df = pd.concat([X, y], axis=1) 
        df_corr = df.corr() #calculation of correlation coefficients

        mean = X.mean()[num]
        std = X.std()[num]
        
        if df_corr.loc['target',num] >= 0:# correlation coefficient is positive
            # definition of a matrix to store weights
            weight = torch.FloatTensor([[2 * 3.75 / 4 * std]])
            self.weight = nn.Parameter(weight)
            # definition of a vector to store the bias
            bias = torch.FloatTensor([-3.75 * 2 * mean / 4 * std])
            self.bias = nn.Parameter(bias)
        elif df_corr.loc['target',num] < 0:# correlation coefficient is negative
            # definition of a matrix to store weights
            weight = torch.FloatTensor([[-2 * 3.75 / 4 * std]])
            self.weight = nn.Parameter(weight)
            # definition of a vector to store the bias
            bias = torch.FloatTensor([3.75 * 2 * mean / 4 * std])
            self.bias = nn.Parameter(bias)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)

######################################################################################################################
class PreprocessingLayerMaxMin(nn.Module):#minmax
    def __init__(self, data, num):
        super().__init__()
        #dataframe creating
        X_train_data = data[:][0].numpy()
        y_train_data = data[:][1].numpy()
        X=pd.DataFrame(data=X_train_data)
        y=pd.DataFrame(data=y_train_data, columns=["target"]) 
        df = pd.concat([X, y], axis=1) 
        df_corr = df.corr() #calculation of correlation coefficients

        mmax = X.max(axis=0) 
        mmin = X.min(axis=0)
        
        
        if df_corr.loc['target',num] >= 0:# correlation coefficient is positive
            # definition of a matrix to store weights
            weight = torch.FloatTensor([[6 / (mmax[num]-mmin[num])]])
            self.weight = nn.Parameter(weight)
            # definition of a vector to store the bias
            bias = torch.FloatTensor([-6 * (mmin[num] + mmax[num]) / (2 * (mmax[num]-mmin[num]))])
            self.bias = nn.Parameter(bias)
        elif df_corr.loc['target',num] < 0:# correlation coefficient is negative
            # definition of a matrix to store weights
            weight = torch.FloatTensor([[-6 / (mmax[num]-mmin[num])]])
            self.weight = nn.Parameter(weight)
            # definition of a vector to store the bias
            bias = torch.FloatTensor([6 * (mmin[num] + mmax[num]) / (2 * (mmax[num]-mmin[num]))])
            self.bias = nn.Parameter(bias)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)
