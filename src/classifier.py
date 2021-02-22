import numpy as np
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support,roc_auc_score
from sklearn.model_selection  import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


def svmClassifier(data,label,kernel="rbf",gamma="scale",C=10):
    x,test_x,y,test_y = train_test_split(data,label,test_size=0.25,random_state=40)
    model = svm.SVC(C=C,kernel=kernel,gamma=gamma)
    model.fit(x, y)
    pred_y = model.predict(test_x)
    p,r,f,_=precision_recall_fscore_support(test_y,pred_y,labels=range(15),average="weighted")
    print(f"precision: {p}\nrecall: {r}\nf-score: {f}")
    print(f"")
    return p,r,f

def rfClassifier(data,label):
    x,test_x,y,test_y = train_test_split(data,label,test_size=0.25,random_state=40)
    model = RandomForestClassifier()
    model.fit(x,y)
    pred_y = model.predict(test_x)
    p,r,f,_=precision_recall_fscore_support(test_y,pred_y,labels=range(15),average="weighted")
    print(f"precision: {p}\nrecall: {r}\nf-score: {f}")

def gbdtClassifier(data,label):
    x,test_x,y,test_y = train_test_split(data,label,test_size=0.25,random_state=40)
    model = GradientBoostingClassifier()
    model.fit(x,y)
    pred_y = model.predict(test_x)
    p,r,f,_=precision_recall_fscore_support(test_y,pred_y,labels=range(15),average="weighted")
    print(f"precision: {p}\nrecall: {r}\nf-score: {f}")



class DataSetFC(Dataset):
    def __init__(self, data,label):
        assert(data.shape[0]==label.shape[0])
        self.datas = data
        self.label=label
    def __len__(self):
        return self.datas.shape[0]
    
    def __getitem__(self, i):
        return torch.tensor(self.datas[i],dtype=torch.float),torch.tensor(self.label[i],dtype=torch.long)

class DataSetCNN(Dataset):
    def __init__(self, data,label):
        assert(data.shape[0]==label.shape[0])
        self.datas = data
        self.label=label
    def __len__(self):
        return self.datas.shape[0]
    
    def __getitem__(self, i):
        return torch.tensor(self.datas[i].reshape(-1,30),dtype=torch.float),torch.tensor(self.label[i],dtype=torch.long)

class FullyConnected(nn.Module):
    def __init__(self,classes=15):
        super(FullyConnected,self).__init__()
        self.linear1=nn.Sequential(nn.Linear(30,200),nn.BatchNorm1d(200),nn.ReLU())
        self.tmp1=nn.Sequential(nn.Linear(200,50),nn.BatchNorm1d(50),nn.ReLU())
        self.linear2=nn.Linear(50,classes)
        self.dropout = nn.Dropout(0.5)
    def forward(self,x):
        x=self.dropout(x)
        x=self.linear1(x)
        x=self.dropout(x)
        x=self.tmp1(x)
        x=self.linear2(x)
        x=nn.Softmax(dim=-1)(x)
        return x

class CNN(nn.Module):
    def __init__(self,classes=50):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  
            nn.Conv1d(1, 16, 3,1,1),  
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), 
        )
        self.conv2 = nn.Sequential( 
            nn.Conv1d(16, 32, 3, 1, 1), 
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2), 
        )
        # self.conv3 = nn.Sequential( 
        #     nn.Conv2d(32, 64, 3, 1, 1), 
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2), 
        # )
        self.flatten = nn.Sequential(nn.Linear(224, 80),nn.BatchNorm1d(80),nn.ReLU())   
        self.out = nn.Linear(80, classes)   
        self.dropout=nn.Dropout(0.5)

    def forward(self, x):
        # print(x)
        x = self.conv1(x)
        x = self.conv2(x)        
        # x = self.conv3(x)        
        x = x.view(x.size(0), -1) 
        x=self.dropout(x)
        x = self.flatten(x)
        x=self.out(x)
        x=nn.LogSoftmax(dim=-1)(x)
        return x



if __name__=="__main__":
    data=np.load("dataset.npy")
    label=np.load("label.npy")
    # svmClassifier(data,label)
    # rfClassifier(data,label)
    # gbdtClassifier(data,label)

    classes=15
    epochs = 200

    initial_lr = 1e-4
    batch_size=32
    criterion=nn.CrossEntropyLoss()
    best_score=0
    epoch_list,train_acc_list,test_acc_list,best_acc_list=[],[],[],[]
    data=np.load("dataset.npy")
    label=np.load("label.npy")

    # load data
    train_image,test_image ,train_label, test_label = train_test_split(data,label,test_size=0.25,random_state=40)
    # note: you should use train_image, train_label for training, apply the model to test_image to get predictions and use test_label to evaluate the predictions 

    train_set=DataSetCNN(train_image,train_label)
    test_set=DataSetCNN(test_image,test_label)
    model=CNN(classes)
    optimizer=torch.optim.Adam(model.parameters(), lr=initial_lr)
    is_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if is_gpu else 'cpu')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    if is_gpu:
        model.cuda()
        criterion.cuda()
    # train model using train_image and train_label
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
    for epoch in range(epochs):
        acc_list=[]
        model.train()
        bar=tqdm(train_loader)
        for b in bar:
            optimizer.zero_grad()
            if is_gpu:
                b[0]=b[0].cuda()
                b[1]=b[1].cuda()
            out=model(b[0])
            batch_pred = out.data.max(1)[1]
            acc = batch_pred.eq(b[1]).float().mean() 
            acc=acc.cpu().detach().numpy()
            acc_list.append(acc)
            loss = criterion(out,b[1])
            loss.backward()
            optimizer.step()
        scheduler.step()
        train_acc=np.mean(acc_list)
        
        acc_list=[]
        with torch.no_grad():
        ### Your Code Here ###
            model.eval()
            # target_num = torch.zeros((1,classes))
            # predict_num = torch.zeros((1,classes))
            # acc_num = torch.zeros((1,classes))
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)
            # bar = tqdm(test_loader)
            # pred = []
            for step,b in enumerate(test_loader):
                if is_gpu:
                    b[0]=b[0].cuda()
                    b[1]=b[1].cuda()
                out=model(b[0])
                # print(out)
                batch_pred = out.data.max(1)[1]
                acc = batch_pred.eq(b[1]).float().mean() 
                acc=acc.cpu().detach().numpy()
                acc_list.append(acc)
                # pred.append(batch_pred)
            
            test_acc=np.mean(acc_list)
            if test_acc>best_score:
                best_score=test_acc
            print('Epoch:{}, Loss:{:.5f}, train_acc:{:.5f}, test_acc:{:.5f}, best:{:.5f}'.format(epoch+1, loss.item(),train_acc,test_acc,best_score))