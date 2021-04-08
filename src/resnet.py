import numpy as np
from numpy.core.numeric import identity
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support,roc_auc_score
from sklearn.model_selection  import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from statistical_based.exp import feature_extractor
from sklearn.model_selection import StratifiedKFold
from classifier import dataAugmentation1

class DataSetCNN(Dataset):
    def __init__(self, data,label):
        assert(data.shape[0]==label.shape[0])
        self.datas = data
        self.label=label
    def __len__(self):
        return self.datas.shape[0]
    
    def __getitem__(self, i):

        return torch.tensor(self.datas[i].reshape(-1,30),dtype=torch.float),torch.tensor(self.label[i],dtype=torch.long)

class BasicBlock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(BasicBlock, self).__init__()
        self.conv1=nn.Conv1d(in_channel,out_channel,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1=nn.BatchNorm1d(out_channel)
        self.relu=nn.ReLU()
        self.conv2=nn.Conv1d(out_channel,out_channel,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm1d(out_channel)
        self.downsample=nn.Sequential(nn.Conv1d(in_channel,out_channel,3,1,1,bias=False))

    def forward(self,x):
        identity=x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if out.size!=identity.size:
            identity=self.downsample(identity)
        out += identity
        out = self.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self,block,numclass=13):
        super(ResNet,self).__init__()
        self.conv1 = nn.Sequential(  
            nn.Conv1d(1, 64, 5,1,1,bias=False),  
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), 
        )
        self.layer1=nn.Sequential(block(64,64),block(64,64))
        self.layer2=nn.Sequential(block(64,128),block(128,128))
        self.layer3=nn.Sequential(block(128,256),block(256,256))
        self.layer4=nn.Sequential(block(256,512),block(512,512))
        self.avgpool=nn.AdaptiveAvgPool1d(1)
        self.fc=nn.Linear(512,numclass)

    def forward(self,x):
        x=self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def train(train_image,test_image ,train_label, test_label):
    classes=13
    epochs = 100
    initial_lr = 1e-4
    batch_size = 32
    criterion = nn.CrossEntropyLoss()
    best_score = 0
    # epoch_list,train_acc_list,test_acc_list,best_acc_list=[],[],[],[]

    train_set=DataSetCNN(train_image,train_label)
    test_set=DataSetCNN(test_image,test_label)
    model=ResNet(BasicBlock,classes)
    optimizer=torch.optim.Adam(model.parameters(), lr=initial_lr)
    is_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if is_gpu else 'cpu')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    if is_gpu:
        model.cuda()
        criterion.cuda()
    # train model using train_image and train_label
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
    max_f,best_p,best_r=0,0,0
    for epoch in range(epochs):
        y_true,y_pred=np.array([],dtype=np.int),np.array([],dtype=np.int)
        acc_list=[]
        model.train()
        bar=tqdm(train_loader)
        for b in bar:
            optimizer.zero_grad()
            if is_gpu:
                b[0]=b[0].cuda()
                b[1]=b[1].cuda()
                # b[2]=b[2].cuda()
            out=model(b[0])
            # out=out.squeeze(dim=-1)
            batch_pred = out.data.max(1)[1]
            y_true=np.append(y_true,b[1].numpy())
            y_pred=np.append(y_pred,batch_pred.numpy())
            acc = batch_pred.eq(b[1]).float().mean() 
            acc=acc.cpu().detach().numpy()
            acc_list.append(acc)
            loss = criterion(out,b[1])
            loss.backward()
            optimizer.step()
        p1,r1,f1,_=precision_recall_fscore_support(y_true,y_pred,labels=np.unique(y_pred),average="macro")
        # print(f"precision: {p}\nrecall: {r}\nf-score: {f}")
        scheduler.step()
        train_acc=np.mean(acc_list)
        # x=set(y_true)-set(y_pred)
        acc_list=[]
        with torch.no_grad():
            model.eval()
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)
            y_true,y_pred=np.array([],dtype=np.int),np.array([],dtype=np.int)
            for step,b in enumerate(test_loader):
                if is_gpu:
                    b[0]=b[0].cuda()
                    b[1]=b[1].cuda()
                    # b[2]=b[2].cuda()
                out=model(b[0])
                batch_pred = out.data.max(1)[1]
                y_true=np.append(y_true,b[1].numpy())
                y_pred=np.append(y_pred,batch_pred.numpy())
                acc = batch_pred.eq(b[1]).float().mean() 
                acc=acc.cpu().detach().numpy()
                acc_list.append(acc)
                # pred.append(batch_pred)
            p2,r2,f2,_=precision_recall_fscore_support(y_true,y_pred,labels=np.unique(y_pred),average="macro")
            test_acc=np.mean(acc_list)
            if test_acc>best_score:
                best_score=test_acc
            if f2>max_f:
                best_p=p2
                best_r=r2
                max_f=f2
            # print('Epoch:{}, Loss:{:.5f}, train_acc:{:.5f}, test_acc:{:.5f}, best:{:.5f}'.format(epoch+1, loss.item(),train_acc,test_acc,best_score))
            print("Epoch:{}, Loss:{:.5f}\ntrain_p:{:.5f}, train_r:{:.5f}, train_f:{:.5f},test_p:{:.5f}, test_r:{:.5f}, test_f:{:.5f},best_p:{:.5f}, best_r:{:.5f}, best_f:{:.5f}".format(epoch+1,loss.item(),p1,r1,f1,p2,r2,f2,best_p,best_r,max_f))
    return max_f

if __name__=="__main__":
    data=np.load("data/dataset.npy")
    label=np.load("data/label.npy")
    # svmClassifier(data,label)
    # rfClassifier(data,label)
    # gbdtClassifier(data,label)

    index=np.where((label==12) | (label==14))[0]
    data=np.delete(data,index,axis=0)
    label=np.delete(label,index)
    idx=np.where(label==13)
    label[idx]=12
    train_image,test_image ,train_label, test_label = train_test_split(data,label,test_size=0.25,random_state=40)
    # train_image,train_label=dataAugmentation1(train_image,train_label,mode=["reverse"])
    train(train_image,test_image ,train_label, test_label)