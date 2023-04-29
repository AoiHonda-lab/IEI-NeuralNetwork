import pandas as pd
import torch
import torch.nn as nn
import csv
from sklearn.model_selection import train_test_split
from ieinn import ieinn
import pprint

def generate_data():
    df=pd.read_csv('CarEvaluation20221207.csv',encoding="shift-jis")
    df=df.drop(0,axis=0)
    df=df.astype(float)
        
    y=pd.DataFrame(df.iloc[:,0])
    X=pd.DataFrame(df.iloc[:,1:])

    return train_test_split(X, y, test_size=0.2, random_state=42)

def main():
    # data Generating
    X_train, X_test, y_train, y_test = generate_data()

    # Extract as a numpy array with value and convert to tensor
    X_train = torch.FloatTensor(X_train.values)
    y_train = torch.FloatTensor(y_train.values)
    X_test = torch.FloatTensor(X_test.values)
    y_test = torch.FloatTensor(y_test.values)

    # Dataset creating
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    # DataLoade creating
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=128, 
                                            shuffle=True, 
                                            num_workers=2)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=128, 
                                            shuffle=False, 
                                            num_workers=2)

    # check GPU or CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    model=ieinn.IE(train_loader,additivity_order=2,narray_op='Algebraic', preprocessing='PreprocessingLayerStandardDeviation').to(device)
    criterion = nn.MSELoss() #loss function
    optimizer = torch.optim.Adam(model.parameters()) #Optimization method
    print(model)
    print()

    #Check the initial parameters before training
    print('initial parameters before training')
    pprint.pprint(model.state_dict())
    print()

    history=model.fit_and_valid(train_loader, test_loader, criterion, optimizer, epochs=100, device=device)
    print()

    #Check the parameters after training
    print('parameters after training')
    pprint.pprint(model.state_dict())
    print()

    #model.plot_test()

    #the decision coefficients
    print('display the decision coefficients')
    print('r2_score:%f' % model.r2_score(test_loader))

if __name__ == '__main__':
    main()
