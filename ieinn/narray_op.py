import torch
import torch.nn as nn
from itertools import combinations

##############################################################################
class Algebraic(nn.Module):
    def __init__(self,add):
        super().__init__()
        self.add=add

    def forward(self, x):
        columns_num = x.size()[1]
        self.nodes_tnorm=x
        items = [i for i in range(0, columns_num)]
        for i in range(2, self.add+1):
            for c in combinations(items, i):
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
        for i in range(2, self.add+1):
            for c in combinations(items, i):
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
        for i in range(2, self.add+1):
            for c in combinations(items, i):
                c=list(c)
                subset = x[:,c]
                subset_sum = torch.sum(subset,1,keepdim=True)##subset総和
                value = subset_sum -len(c)+1
                zeros = torch.zeros(instance_num,1)
                value_zeros_list = torch.cat((value,zeros),dim=1)##value,zerosを一つの配列とし,torch.maxが行えるようにする
                result = torch.max(value_zeros_list,1,keepdim=True)[0]
                self.nodes_tnorm=torch.cat((self.nodes_tnorm,result),dim=1)
        return self.nodes_tnorm

#################################################################################################################################
class Drastic(nn.Module):
    def __init__(self,add):
        super().__init__()
        self.add=add

    def forward(self, x):
        columns_num = x.size()[1]
        self.nodes_tnorm=x

        items = [i for i in range(0, columns_num)]
        for i in range(2, self.add+1):
            for c in combinations(items, i):
                c=list(c)#tuple to list
                subset= x[:,c]
                hidden=subset[:,[0]]
                result=subset[:,[0]]
                for i in range(1,len(c)):
                    x_ones = hidden==1.0
                    y_ones = subset[:,[i]]==1.0
                    result[x_ones]=subset[:,[i]][x_ones]
                    result[y_ones]=hidden[y_ones]
                    otherwise = x_ones+y_ones
                    otherwise = list(map(lambda x: not x, otherwise))
                    otherwise=torch.tensor(otherwise)
                    otherwise=torch.reshape(otherwise,x_ones.size())
                    result[otherwise]=0

                    hidden=result
                            
                self.nodes_tnorm=torch.cat((self.nodes_tnorm,hidden),dim=1)
        return self.nodes_tnorm
################################################################################################################

class Dubois(nn.Module):
    def __init__(self,add,lambda_=0.3):
        super().__init__()
        self.add=add
        lambda_ = torch.tensor(lambda_)
        self.lambda_ = nn.Parameter(lambda_)

    def forward(self, x):
        columns_num = x.size()[1]
        self.nodes_tnorm=x

        items = [i for i in range(0, columns_num)]
        for i in range(2, self.add+1):
            for c in combinations(items, i):
                c=list(c)
                subset = x[:,c]
                hidden=subset[:,[0]]
                for i in range(1,len(c)):
                    hidden=torch.mul(hidden, subset[:,[i]])/torch.max(torch.max(hidden, subset[:,[i]]),self.lambda_)
                            
                self.nodes_tnorm=torch.cat((self.nodes_tnorm,hidden),dim=1)

        return self.nodes_tnorm
    
##################################################################################################################################

class Hamacher(nn.Module):
    def __init__(self,add,lambda_=1.0):
        super().__init__()
        self.add=add
        lambda_ = torch.tensor(lambda_)
        self.lambda_ = nn.Parameter(lambda_)

    def forward(self, x):
        columns_num = x.size()[1]
        self.nodes_tnorm=x

        items = [i for i in range(0, columns_num)]
        for i in range(2, self.add+1):
            for c in combinations(items, i):
                c=list(c)
                subset = x[:,c]
                hidden=subset[:,[0]]
                for i in range(1,len(c)):
                    hidden=torch.mul(hidden, subset[:,[i]])/(self.lambda_+torch.mul((1-self.lambda_),(hidden+subset[:,[i]]-torch.mul(hidden, subset[:,[i]]))))
                            
                self.nodes_tnorm=torch.cat((self.nodes_tnorm,hidden),dim=1)

        return self.nodes_tnorm
    
#################################################################################################################################

class Schweizer(nn.Module):
    def __init__(self,add,lambda_=0.3):
        super().__init__()
        self.add=add
        lambda_ = torch.tensor(lambda_)
        self.lambda_ = nn.Parameter(lambda_)

    def forward(self, x):
        columns_num = x.size()[1]
        self.nodes_tnorm=x

        items = [i for i in range(0, columns_num)]
        for i in range(2, self.add+1):
            for c in combinations(items, i):
                c=list(c)
                subset = x[:,c]
                hidden=subset[:,[0]]
                for i in range(1,len(c)):
                    hidden=1-((1-hidden)**self.lambda_+(1-subset[:,[i]])**self.lambda_-torch.mul((1-hidden)**self.lambda_,(1-subset[:,[i]])**self.lambda_))**(1/self.lambda_)
                            
                self.nodes_tnorm=torch.cat((self.nodes_tnorm,hidden),dim=1)

        return self.nodes_tnorm
    
#################################################################################################################################

class Yager(nn.Module):
    def __init__(self,add,lambda_=0.3):
        super().__init__()
        self.add=add
        lambda_ = torch.tensor(lambda_)
        self.lambda_ = nn.Parameter(lambda_)

    def forward(self, x):
        columns_num = x.size()[1]
        self.nodes_tnorm=x

        items = [i for i in range(0, columns_num)]
        for i in range(2, self.add+1):
            for c in combinations(items, i):
                c=list(c)
                subset = x[:,c]
                hidden=subset[:,[0]]
                for i in range(1,len(c)):
                    tmp=(1-hidden)**self.lambda_+(1-subset[:,[i]])**self.lambda_
                    hidden=1-torch.min(torch.ones(tmp.size()),tmp)**(1/self.lambda_)
                            
                self.nodes_tnorm=torch.cat((self.nodes_tnorm,hidden),dim=1)

        return self.nodes_tnorm

#################################################################################################################################

class Dombi(nn.Module):
    def __init__(self,add,lambda_=0.3):
        super().__init__()
        self.add=add
        lambda_ = torch.tensor(lambda_)
        self.lambda_ = nn.Parameter(lambda_)

    def forward(self, x):
        columns_num = x.size()[1]
        self.nodes_tnorm=x
        items = [i for i in range(0, columns_num)]

        if self.lambda_==0:
            return Min(self.add)

        else:
            for i in range(2, self.add+1):
                for c in combinations(items, i):
                    c=list(c)
                    subset = x[:,c]
                    hidden=subset[:,[0]]
                    for i in range(1,len(c)):
                        hidden=1/(1+((1/hidden-1)**self.lambda_+(1/subset[:,[i]]-1)**self.lambda_)**(1/self.lambda_))
                                
                    self.nodes_tnorm=torch.cat((self.nodes_tnorm,hidden),dim=1)

            return self.nodes_tnorm

################################################################################################################
class AlgebraicSum(nn.Module):
    def __init__(self,add):
        super().__init__()
        self.add=add

    def forward(self, x):
        columns_num = x.size()[1]
        self.nodes_tnorm=x

        items = [i for i in range(0, columns_num)]
        for i in range(2, self.add+1):
            for c in combinations(items, i):
                c=list(c)#tuple to list
                subset= x[:,c]
                hidden=subset[:,[0]]
                for i in range(1,len(c)):
                    hidden=hidden + subset[:,[i]] - torch.mul(hidden, subset[:,[i]])
                            
                self.nodes_tnorm=torch.cat((self.nodes_tnorm,hidden),dim=1)
        return self.nodes_tnorm

################################################################################################################
class Max(nn.Module):
    def __init__(self,add):
        super().__init__()
        self.add=add

    def forward(self, x):
        columns_num = x.size()[1]
        self.nodes_tnorm=x
        items = [i for i in range(0, columns_num)]
        for i in range(2, self.add+1):
            for c in combinations(items, i):
                c=list(c)
                subset= x[:,c]#xからcの説明変数群を獲得
                result=torch.max(subset,1,keepdim=True)[0]#subset最小値を獲得
                self.nodes_tnorm=torch.cat((self.nodes_tnorm,result),dim=1)#一点集合が格納済みのself.nodes_tnormに二点集合以上の結果を追加
        return self.nodes_tnorm
    
################################################################################################################
class LukasiewiczSum(nn.Module):
    def __init__(self,add):
        super().__init__()
        self.add=add

    def forward(self, x):
        columns_num = x.size()[1]
        self.nodes_tnorm=x

        items = [i for i in range(0, columns_num)]
        for i in range(2, self.add+1):
            for c in combinations(items, i):
                c=list(c)#tuple to list
                subset= x[:,c]
                hidden=subset[:,[0]]
                for i in range(1,len(c)):
                    hidden=torch.min(hidden + subset[:,[i]], torch.tensor(1.0))
                            
                self.nodes_tnorm=torch.cat((self.nodes_tnorm,hidden),dim=1)
        return self.nodes_tnorm

################################################################################################################
class DrasticSum(nn.Module):
    def __init__(self,add):
        super().__init__()
        self.add=add

    def forward(self, x):
        columns_num = x.size()[1]
        self.nodes_tnorm=x

        items = [i for i in range(0, columns_num)]
        for i in range(2, self.add+1):
            for c in combinations(items, i):
                c=list(c)#tuple to list
                subset= x[:,c]
                hidden=subset[:,[0]]
                result=subset[:,[0]]
                for i in range(1,len(c)):
                    x_ones = hidden==0
                    y_ones = subset[:,[i]]==0
                    result[x_ones]=subset[:,[i]][x_ones]
                    result[y_ones]=hidden[y_ones]
                    otherwise = x_ones+y_ones
                    otherwise = list(map(lambda x: not x, otherwise))
                    otherwise=torch.tensor(otherwise)
                    otherwise=torch.reshape(otherwise,x_ones.size())
                    result[otherwise]=1.0

                    hidden=result
                            
                self.nodes_tnorm=torch.cat((self.nodes_tnorm,hidden),dim=1)
        return self.nodes_tnorm

################################################################################################################
class ArithmeticMean(nn.Module):
    def __init__(self,add):
        super().__init__()
        self.add=add

    def forward(self, x):
        columns_num = x.size()[1]
        self.nodes_tnorm=x
        items = [i for i in range(0, columns_num)]
        for i in range(2, self.add+1):
            for c in combinations(items, i):
                c=list(c)#tuple to list
                subset= x[:,c]
                result=torch.sum(subset,1,keepdim=True)
                result=result/subset.size()[1]
                self.nodes_tnorm=torch.cat((self.nodes_tnorm,result),dim=1)
        return self.nodes_tnorm