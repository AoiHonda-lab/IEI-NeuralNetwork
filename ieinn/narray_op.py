import torch
import torch.nn as nn
from itertools import combinations

class Algebraic(nn.Module):
    def __init__(self,add):
        super().__init__()
        self.add=add

    def forward(self, x):
        columns_num = x.size()[1]
        self.nodes_tnorm=x
        items = [i for i in range(0, columns_num)]
        for i in range(columns_num+1):
            if i >self.add:#add加法的まで
                break
            for c in combinations(items, i):
                if len(c)>=2:#len(c)=1のときはそのまま出力層へ
                    c=list(c)#tuple to list
                    subset= x[:,c]
                    result=torch.prod(subset,1,keepdim=True)
                    self.nodes_tnorm=torch.cat((self.nodes_tnorm,result),dim=1)
        return self.nodes_tnorm
################################################################################################################
class Min(nn.Module):
    def __init__(self,add):
        super().__init__()
        self.add=add

    def forward(self, x):
        columns_num = x.size()[1]
        self.nodes_tnorm=x
        items = [i for i in range(0, columns_num)]
        for i in range(columns_num+1):
            if i >self.add:#add加法的まで
                break
            for c in combinations(items, i):
                if len(c)>=2:#len(c)=1のときはそのまま出力層へ
                    c=list(c)
                    subset= x[:,c]#xからcの説明変数群を獲得
                    result=torch.min(subset,1,keepdim=True)[0]#subset最小値を獲得
                    self.nodes_tnorm=torch.cat((self.nodes_tnorm,result),dim=1)#一点集合が格納済みのself.nodes_tnormに二点集合以上の結果を追加
        return self.nodes_tnorm

################################################################################################################

class Lukasiewicz(nn.Module):
    def __init__(self,add):
        super().__init__()
        self.add=add

    def forward(self, x):
        columns_num = x.size()[1]
        instance_num = x.size()[0]
        self.nodes_tnorm=x

        items = [i for i in range(0, columns_num)]
        for i in range(columns_num+1):
            if i >self.add:#add加法的まで
                break
            for c in combinations(items, i):
                if len(c)>=2:#len(c)=1のときはそのまま出力層へ
                    c=list(c)
                    subset = x[:,c]
                    subset_sum = torch.sum(subset,1,keepdim=True)##subset総和
                    value = subset_sum -len(c)+1
                    zeros = torch.zeros(instance_num,1)
                    value_zeros_list = torch.cat((value,zeros),dim=1)##value,zerosを一つの配列とし,torch.maxが行えるようにする
                    result = torch.max(value_zeros_list,1,keepdim=True)[0]
                    self.nodes_tnorm=torch.cat((self.nodes_tnorm,result),dim=1)

        return self.nodes_tnorm
