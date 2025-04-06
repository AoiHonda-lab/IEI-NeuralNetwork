import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.special import comb
import time
from . import narray_op as n_op
from . import preprocessing_layer
from . import output_layer
from itertools import combinations
import numpy as np
import csv

class IE(nn.Module):
    def __init__(self, dataloader, additivity_order=None, narray_op='Algebraic', preprocessing='PreprocessingLayerPercentile', device='cpu'):
        super(IE,self).__init__()
        self.X_data = dataloader.dataset[:][0]
        self.additivity_order =  additivity_order   # the additivity order of the fuzzy measure
        self.narray_op = narray_op                  # interaction operator
        self.preprocessing = preprocessing          # preprocessing method
        self.columns_num = self.X_data.size()[1]
        self.device = device
                
                    
        if self. additivity_order == None:
            self. additivity_order = self.columns_num
        self.set_list = self.power_sets_additivity_order()[1:] # Subset up to additivity specification

        if self. additivity_order > self.columns_num:
            raise IndexError('" additivity_order" must be less than the "number of features"')

        # Error handling for arguments
        t_norm = ["Algebraic", "Min", "Lukasiewicz", "Drastic", "Dubois", "Hamacher", "Schweizer", "Yager", "Dombi"]
        t_conorm = ["AlgebraicSum", "Max", "LukasiewiczSum", "DrasticSum"]
        others = ["ArithmeticMean"]
        valid_narray_op = t_norm + t_conorm + others

        if self.narray_op not in valid_narray_op:
            # Create a categorized error message
            error_message = (
                f"Input Error: '{self.narray_op}' is not a valid input.\n\n"
                "Valid inputs are categorized as follows:\n"
                "▶ t-norm:\n" + ", ".join(t_norm) + "\n\n"
                "▶ t-conorm:\n" + ", ".join(t_conorm) + "\n\n"
                "▶ Others:\n" + ", ".join(others)
            )
            raise ValueError(error_message)

        if self.preprocessing not in ['PreprocessingLayerPercentile', 'PreprocessingLayerStandardDeviation', 'PreprocessingLayerMaxMin', 'Random']:
            raise ValueError('Valid inputs are as follows / PreprocessingLayerPercentile, PreprocessingLayerStandardDeviation, PreprocessingLayerMaxMin, Random')
    
        # Layer Definition
        # PreprocessingLayer
        preprocessing_cls = getattr(preprocessing_layer, self.preprocessing)
        for i in range(0,self.columns_num):
            exec("self.preprocessing" + str(i+1) + "= preprocessing_cls(dataloader.dataset, "+str(i)+")")
        #IEILayer
        iei_cls = getattr(n_op, self.narray_op)
        self.iei = iei_cls(self.additivity_order)
        # OutputLayer
        output_nodes_sum = 0
        for i in range(1,self.additivity_order+1):
            output_nodes_sum += comb(self.columns_num, i, exact=True)
        self.output = output_layer.OutputLayer(self.columns_num, output_nodes_sum)
            
    def forward(self, x):
        columns_num = x.size()[1]
        self.nodes=torch.tensor(()).to(self.device)

        #preprocessing
        for i in range(0, columns_num):
            exec("x" + str(i+1) + "= self.preprocessing" + str(i+1) + "(x[:," + str(i) + "].view(x.size()[0],1))")
            exec("x_sig" + str(i+1) + "= torch.sigmoid(x" + str(i+1) + ")")
            exec("self.nodes=torch.cat((self.nodes,x_sig" + str(i+1) + "),dim=1)")
        #iei
        hidden=self.iei(self.nodes)
        #output
        return self.output(hidden)

    def fit_and_valid(self, train_loader, test_loader, criterion, optimizer, epochs=100, regularization=True, mono_lambda=0.2):
        start = time.time()
        self.train_loss_list = []
        self.val_loss_list = []
        self.not_mono_num_list = []
        self.not_mono_num_percentage = []
    
        for epoch in range(epochs):
            self.train_loss = 0
            self.val_loss = 0
            
            #train
            self.train()
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                # Zero your gradients for every batch
                optimizer.zero_grad()
                # Make predictions for this batch
                outputs = self(images)
                
                # regularization term
                reg_sum = torch.tensor(0., requires_grad=True) 
                # Weights of output layers (up to additivity_order)
                self.dict_weight = dict(zip(self.set_list, self.output.weight[0]))
                                              
                if regularization == True:
                    for A in self.set_list:
                        i_sum = torch.tensor(0., requires_grad=True)
                        for i in A:
                            sum = 0
                            for B in self.set_list:
                                # B contains element i and is a true subset of A
                                if (set({i})<=(set(B))) and (set(B)<(set(A))):
                                    sum = sum - self.dict_weight[B]
                            # If the monotonicity condition is not met
                            if self.dict_weight[A] < (sum):
                                if i_sum == 0:
                                    i_sum = torch.norm(self.dict_weight[A]-sum)
                                elif i_sum < torch.norm(self.dict_weight[A]-sum):
                                    i_sum = torch.norm(self.dict_weight[A]-sum)
                                        
                        reg_sum = reg_sum + torch.norm(i_sum)
                else:
                    pass 
    
                # Compute the loss
                loss = criterion(outputs, labels) #Loss without regularization term (for output)
                self.train_loss += loss.item()*len(labels)
                
                loss = loss + mono_lambda * reg_sum # Loss with regularization term
                
                # Compute the its gradients
                loss.backward()
                # Adjust learning weights
                optimizer.step()

            avg_train_loss = self.train_loss / len(train_loader.dataset)

            #val
            self.eval()
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
            
                    outputs = self(images)
                    loss = criterion(outputs, labels)
                    self.val_loss += loss.item()*len(labels)
            avg_val_loss = self.val_loss / len(test_loader.dataset)

            print ('Epoch [{}/{}], loss: {loss:.8f} val_loss: {val_loss:.8f}' 
                        .format(epoch+1, epochs, i+1, loss=avg_train_loss, val_loss=avg_val_loss))
            self.train_loss_list.append(avg_train_loss)
            self.val_loss_list.append(avg_val_loss)

        print("time")
        print(f"{time.time() - start:.2f} sec")

        return self.val_loss

    # Get the weights of the output layer with numpy
    def get_weight(self):
        np_weight = self.output.weight.to('cpu').detach().numpy().copy()
        return np.ravel(np_weight)
    
    #　Obtain a power set
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

    # Get subset up to additivity specification
    def power_sets_additivity_order(self):
        items = [i for i in range(1, self.columns_num+1)]
        sub_sets=[]
        for i in range(len(items) + 1):
            if i > self.additivity_order:
                break
            for c in combinations(items, i):
                sub_sets.append(c)
        all_sets = sub_sets
        return all_sets

    # Learning curve for training data
    def plot_train(self):
        plt.figure(figsize=(8,6))
        plt.plot(self.train_loss_list,label='train data', lw=3, c='b')
        plt.xticks(size=14)
        plt.yticks(size=14)
        plt.grid(lw=2)
        plt.legend(fontsize=14)
        plt.show()
        return 0

    # Learning curve for test data
    def plot_test(self):
        plt.figure(figsize=(8,6))
        plt.plot(self.val_loss_list,label='test data', lw=3, c='b')
        plt.xticks(size=14)
        plt.yticks(size=14)
        plt.grid(lw=2)
        plt.legend(fontsize=14)
        plt.show()
        return 0
        
    # Learning curves for training and test data
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

    # coefficient of determination
    def r2_score(self,test_loader):
        mse_loss = torch.nn.MSELoss()
        test_image = test_loader.dataset[:][0].to(self.device)
        test_label = test_loader.dataset[:][1].to(self.device)

        # average_predict
        test_mean=torch.mean(test_label)
        test_mean_list=torch.ones(len(test_label),1,device=self.device)*test_mean
        mean_loss=mse_loss(test_label,test_mean_list)

        # model_predict
        outputs=self(test_image)
        test_loss=mse_loss(test_label,outputs)
        return 1-test_loss.item()/mean_loss.item()
    
    # Output sets that do not satisfy monotonicity
    def test_monotone(self):
        ng_list = [] # List of sets not satisfying monotonicity
        self.dict_weight_np = dict(zip(self.set_list, self.get_weight()))

        for A in self.set_list:
            for i in A:
                sum = 0
                for B in self.set_list:
                    if (set({i})<=(set(B))) and (set(B)<(set(A))):
                        sum = sum - self.dict_weight_np[B]
                if self.dict_weight_np[A] < (sum):
                    ng_list.append(A)
                    break
        return ng_list
    
    # Calculate monotone mesures from weights of output layer
    def mobius_to_fuzzy(self, mobius_transformation=False):
        np.set_printoptions(suppress=True)
        fuzzy_ = []
        all_sets = (self.power_sets()) 
        output_weight = (self.output.weight).to('cpu').detach().numpy().copy() 
        output_weight_list = output_weight[0].tolist()
        output_weight_list.insert(0, 0)
        data = np.array(output_weight_list)
        values=np.zeros(2**self.columns_num)
        values[:len(data)]=data
        values.tolist()

        # Möbius reverse transform
        for i in range(0, len(values)):
            f = 0
            for j in range(0, len(values)):
                A = all_sets[i]
                B = all_sets[j]
                if  set(B) <= set(A):
                    f += values[j]
            fuzzy_.append(f)
        dict_fuzzy = dict(zip(all_sets, fuzzy_))

        return dict_fuzzy

    def shapley(self):
        additivity_sets = (self.power_sets_additivity_order()) 
        output_weight = (self.output.weight).to('cpu').detach().numpy().copy()
        
        output_weight_list = output_weight[0].tolist()
        output_weight_list.insert(0, 0)
        data = np.array(output_weight_list)
        dict_fuzzy = dict(zip(additivity_sets, data.tolist()))

        shapley = []
        for i in range(1, self.columns_num+1):
            shapley_value = 0
            A = additivity_sets[i]
            for B in additivity_sets:
                if set(A) <= set(B):
                    shapley_value += dict_fuzzy[B] / len(B)
            shapley.append(shapley_value) 
        shapley = dict(zip(additivity_sets[1:], shapley))
        return shapley

    def shapley_plot(self,labels=None):
        shapley=self.shapley()
        if labels is None:
            labels=[str(k) for k in shapley.keys()]

        labels=list(labels)
        values = list(shapley.values())

        plt.figure(figsize=(10, 6))
        plt.barh(labels, values, color='skyblue')
        plt.xlabel('Shapley value')
        plt.ylabel('Columns')
        plt.title('Shapley Value')
        plt.gca().invert_yaxis() 
        plt.show()
        return 0
    
    # Export output layer weights
    def weight_exp(self):
        with open('output_weight.csv', 'w',newline='') as f:
            writer = csv.writer(f)
            for weight in self.output.weight:
                for i in weight:
                    writer.writerow([i.tolist()])
    
    # Export all parameters
    def param_exp(self):
        with open('param.csv', 'w',newline='') as f:
            writer = csv.writer(f)
            for p in self.parameters():
                for i in p:
                    if i.ndim == 1:
                        writer.writerow(i.tolist())
                    else:
                        writer.writerow([i.tolist()])