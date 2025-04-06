import torch
import torch.nn as nn



class OutputLayer(nn.Module):
    def __init__(self, columns_num, all_num):#
        super().__init__()
        self.columns_num=columns_num
        self.all_num=all_num
        
        # definition of a matrix to store weights
        weight = torch.zeros(1, self.all_num)
        weight[0,:self.columns_num]=1/self.columns_num
        self.weight = nn.Parameter(weight)
        # definition of a vector to store the bias
        bias = torch.FloatTensor([0])
        self.bias = nn.Parameter(bias)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)
