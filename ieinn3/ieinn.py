import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.special import comb
import time
from . import narray_op as n_op
from . import preprocessing_layer
from . import output_layer


class IE(nn.Module):
    def __init__(self, dataloader, additivity_order=None, narray_op='Algebraic', preprocessing='PreprocessingLayerPercentile'):
        super(IE,self).__init__()
        self.X_data = dataloader.dataset[:][0]
        self.add = additivity_order
        self.narray_op = narray_op
        self.preprocessing = preprocessing
        self.columns_num = self.X_data.size()[1]
                    
        if self.add == None:
            self.add = self.columns_num

        if self.add > self.columns_num:
            raise IndexError('"additivity_order" must be less than the "number of features"')
        if self.narray_op not in ['Algebraic', 'Min', 'Lukasiewicz']:#, 'Dubois', 'Hamacher']:
            raise ValueError('narray_op / Algebraic, Min, Lukasiewicz')#, Dubois, Hamacher')
        if self.preprocessing not in ['PreprocessingLayerPercentile', 'PreprocessingLayerStandardDeviation', 'PreprocessingLayerMaxMin']:
            raise ValueError('preprocessing / PreprocessingLayerPercentile, PreprocessingLayerStandardDeviation, PreprocessingLayerMaxMin')
    
        # PreprocessingLayer
        preprocessing_cls = getattr(preprocessing_layer, self.preprocessing)
        for i in range(0,self.columns_num):
            exec("self.preprocessing" + str(i+1) + "= preprocessing_cls(dataloader.dataset, "+str(i)+")")
        #IEILayer
        iei_cls = getattr(n_op, self.narray_op)
        self.iei = iei_cls(self.add)
        # OutputLayer
        output_nodesSum = 0
        for i in range(1,self.add+1):
            output_nodesSum += comb(self.columns_num, i, exact=True)
        self.output = output_layer.OutputLayer(self.columns_num, output_nodesSum)
            
    def forward(self, x):
        columns_num = x.size()[1]
        self.nodes=torch.tensor(())

        #preprocessing
        for i in range(0, columns_num):
            exec("x" + str(i+1) + "= self.preprocessing" + str(i+1) + "(x[:," + str(i) + ":" + str(i+1) + "])")
            exec("x_sig" + str(i+1) + "= torch.sigmoid(x" + str(i+1) + ")")
            exec("self.nodes=torch.cat((self.nodes,x_sig" + str(i+1) + "),dim=1)")
        #iei
        hidden=self.iei(self.nodes)
        #output
        return self.output(hidden)

    def fit_and_valid(self, train_Loader, test_Loader, criterion, optimizer, device='cpu', epochs=100):
        start = time.time()
        self.train_loss_list = []
        self.val_loss_list = []
        self.lrs_list=[]

        for epoch in range(epochs):
            self.train_loss = 0
            self.val_loss = 0

            #train
            self.train()
            for i, (images, labels) in enumerate(train_Loader):
                images, labels = images.to(device), labels.to(device)

                # Zero your gradients for every batch
                optimizer.zero_grad()
                # Make predictions for this batch
                outputs = self(images)
                # Compute the loss
                loss = criterion(outputs, labels)
                self.train_loss += loss.item()*len(labels)
                # Compute the its gradients
                loss.backward()
                # Adjust learning weights
                optimizer.step()
                self.lrs=optimizer.param_groups[0]["lr"]
                self.lrs_list.append(optimizer.param_groups[0]["lr"])

            avg_train_loss = self.train_loss / len(train_Loader.dataset)

            #val
            self.eval()
            with torch.no_grad():
                for images, labels in test_Loader:
                    images = images.to(device)
                    labels = labels.to(device)
            
                    outputs = self(images)
                    loss = criterion(outputs, labels)
                    self.val_loss += loss.item()*len(labels)
            avg_val_loss = self.val_loss / len(test_Loader.dataset)

            print ('Epoch [{}/{}], train_loss: {loss:.8f} val_loss: {val_loss:.8f}' 
                        .format(epoch+1, epochs, i+1, loss=avg_train_loss, val_loss=avg_val_loss))
            self.train_loss_list.append(avg_train_loss)
            self.val_loss_list.append(avg_val_loss)
        print("time")
        print(time.time() - start)

        return self.val_loss

    def plot_train(self):
        plt.figure(figsize=(8,6))
        plt.plot(self.train_loss_list,label='train data', lw=3, c='b')
        plt.xticks(size=14)
        plt.yticks(size=14)
        plt.grid(lw=2)
        plt.legend(fontsize=14)
        plt.show()
        return 0

    def plot_test(self):
        plt.figure(figsize=(8,6))
        plt.plot(self.val_loss_list,label='test data', lw=3, c='b')
        plt.xticks(size=14)
        plt.yticks(size=14)
        plt.grid(lw=2)
        plt.legend(fontsize=14)
        plt.show()
        return 0
        
    def plot(self):
        plt.figure(figsize=(8,6))
        plt.plot(self.train_loss_list,label='train data', lw=3, c='b')
        plt.plot(self.val_loss_list,label='test data', lw=3, c='r')
        plt.xticks(size=14)
        plt.yticks(size=14)
        plt.grid(lw=2)
        plt.legend(fontsize=14)
        plt.show()
        return 0
    
    def r2_score(self,test_Loader,device='cpu'):
        test_label = test_Loader.dataset[:][1].to(device)
        test_mean=torch.mean(test_label)
        test_mean_list=torch.ones(len(test_label),1)*test_mean
        mse_loss = torch.nn.MSELoss() 
        mean_loss=mse_loss(test_label,test_mean_list.to(device))*len(test_label)
        return 1-self.val_loss/mean_loss.item()

 #　組み合わせを作る関数
    def power_sets(self):
        items = [i for i in range(1, self.columns_num+1)]
        sub_sets=[]
        for i in range(len(items) + 1):
            if i > self.columns_num:
                break
            for c in combinations(items, i):
                sub_sets.append(c)
        all_sets = sub_sets
        return all_sets

    # メビウス変換・逆変換を行う関数
    def mobius(self, mobius_transformation=False):
        fuzzy_ = []
        all_sets = (self.power_sets()) 
        self.np_houjo_w = (self.output.weight).to('cpu').detach().numpy().copy() #最後の層の重みをnumpyに変換
        l_data = self.np_houjo_w[0].tolist()
        l_data.insert(0, 0)
        data = np.array(l_data)

        #メビウス変換
        if mobius_transformation==True:
            for i in range(0, len(data)):
                f = 0
                for j in range(0, len(data)):
                    A = all_sets[i]
                    B = all_sets[j]
                    if  set(B) <= set(A): #"<="は部分集合か、"<"にしたら真部分集合か判定できる
                        f += data[j]*(-1)**(len(all_sets[i])-len(all_sets[j]))
                fuzzy_.append(f)
            d_fuzzy = dict(zip(all_sets, fuzzy_))

        #メビウス逆変換
        else:
            for i in range(0, len(data)):
                f = 0
                for j in range(0, len(data)):
                    A = all_sets[i]
                    B = all_sets[j]
                    if  set(B) <= set(A): #"<="は部分集合か、"<"にしたら真部分集合か判定できる
                        f += data[j]
                fuzzy_.append(f)
            d_fuzzy = dict(zip(all_sets, fuzzy_))

        return d_fuzzy #部分集合の組み合わせと、そのファジィ測度の値を辞書の形にしたものを返す

    #　シャプレイ値を返す関数
    def shapley(self, value_type='mobius'):
        all_sets = (self.power_sets()) 
        self.np_houjo_w = (self.output.weight).to('cpu').detach().numpy().copy() #最後の層の重みをnumpyに変換

        #メビウス変換値でなければメビウス変換
        if value_type=='fuzzy':
            d_fuzzy = self.mobius(mobius_transformation=True)
        #メビウス変換値ならそのまま
        if value_type=='mobius':
            l_data = self.np_houjo_w[0].tolist()
            l_data.insert(0, 0)
            data = np.array(l_data)
            d_fuzzy = dict(zip(all_sets, data))

        shapley = []
        for i in range(1, self.columns_num+1):
            shapley_val = 0
            A = all_sets[i]
            for B in all_sets:
                if set(A) <= set(B):
                    shapley_val += d_fuzzy[B] / len(B)
            shapley.append(shapley_val) 
        shapley = dict(zip(all_sets[1:], shapley))
        return shapley