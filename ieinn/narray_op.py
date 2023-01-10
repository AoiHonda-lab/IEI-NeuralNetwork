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
        t=0
        for i in range(columns_num+1):
            if i >self.add:#add加法的まで
                break
            for c in combinations(items, i):
                if len(c)>=2:#len(c)=1のときはそのまま出力層へ
                    t+=1
                    c=list(c)
                    exec("subset"+str(t)+"= x[:,c]")
                    exec("result"+str(t)+"=torch.prod(subset"+str(t)+",1,keepdim=True)")
                    
                    exec("self.nodes_tnorm=torch.cat((self.nodes_tnorm,result"+str(t)+"),dim=1)")

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
        t=0
        for i in range(columns_num+1):
            if i >self.add:#add加法的まで
                break
            for c in combinations(items, i):
                if len(c)>=2:#len(c)=1のときはそのまま出力層へ
                    t+=1
                    c=list(c)
                    exec("subset"+str(t)+"= x[:,c]")
                    exec("result"+str(t)+"=torch.min(subset"+str(t)+",1,keepdim=True)")
                    
                    exec("self.nodes_tnorm=torch.cat((self.nodes_tnorm,result"+str(t)+"[0]),dim=1)")

        return self.nodes_tnorm