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
from sklearn.ensemble import RandomForestClassifier
from statistical_based.exp import feature_extractor
from sklearn.model_selection import StratifiedKFold
import time

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
    def __init__(self, data,label,features):
        assert(data.shape[0]==label.shape[0])
        self.datas = data
        self.label=label
    def __len__(self):
        return self.datas.shape[0]
    
    def __getitem__(self, i):
        return torch.tensor(self.datas[i],dtype=torch.float),torch.tensor(self.label[i],dtype=torch.long)

class DataSetCNN(Dataset):
    def __init__(self, data,label,features):
        assert(data.shape[0]==label.shape[0])
        self.datas = data
        self.label=label
        self.features=features
    def __len__(self):
        return self.datas.shape[0]
    
    def __getitem__(self, i):

        return torch.tensor(self.datas[i].reshape(-1,30),dtype=torch.float),torch.tensor(self.label[i],dtype=torch.long),torch.tensor(self.features[i],dtype=torch.float)

class FullyConnected(nn.Module):
    def __init__(self,classes=15):
        super(FullyConnected,self).__init__()
        self.linear1=nn.Sequential(nn.Linear(30,200),nn.BatchNorm1d(200),nn.ReLU())
        self.tmp1=nn.Sequential(nn.Linear(200,500),nn.BatchNorm1d(500),nn.ReLU())
        # self.tmp2=nn.Sequential(nn.Linear(500,1000),nn.BatchNorm1d(1000),nn.ReLU())
        self.tmp3=nn.Sequential(nn.Linear(500,200),nn.BatchNorm1d(200),nn.ReLU())
        self.tmp4=nn.Sequential(nn.Linear(200,50),nn.BatchNorm1d(50),nn.ReLU())
        self.linear2=nn.Linear(50,classes)
        self.dropout = nn.Dropout(0.5)
    def forward(self,x):
        x=self.dropout(x)
        x=self.linear1(x)
        
        x=self.tmp1(x)
        # x=self.tmp2(x)
        x=self.tmp3(x)
        x=self.tmp4(x)
        x=self.linear2(x)
        x=self.dropout(x)
        x=nn.Softmax(dim=-1)(x)
        return x

class CNN(nn.Module):
    def __init__(self,classes=50):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  
            nn.Conv1d(1, 64, 5,1,1),  
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2), 
        )
        self.conv2 = nn.Sequential( 
            nn.Conv1d(64, 128, 5, 1, 1,bias=False), 
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2), 
        )
        self.conv3 = nn.Sequential( 
            nn.Conv1d(128, 256, 7, 1, 1,bias=False), 
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2), 
        )
        self.flatten = nn.Sequential(nn.Linear(1024, 128),nn.BatchNorm1d(128),nn.ReLU())
        # self.out = nn.Linear(256, 64)   
        self.out = nn.Linear(128, classes)   
        self.dropout=nn.Dropout(0.5)

    def forward(self, x):
        # print(x)
        x = self.conv1(x)
        x = self.conv2(x)        
        x = self.conv3(x)        
        x = x.view(x.size(0), -1) 
        x=self.dropout(x)
        x = self.flatten(x)
        # x=torch.cat((x,y),1)
        x=self.out(x)
        # x=self.out1(x)
        x=nn.LogSoftmax(dim=-1)(x)
        # x=nn.Softmax(dim=-1)(x)
        return x


def train(train_image,test_image ,train_label, test_label,train_feature,test_feature):
    classes=13
    epochs = 100
    initial_lr = 1e-4
    batch_size = 32
    criterion = nn.CrossEntropyLoss()
    best_score = 0
    # epoch_list,train_acc_list,test_acc_list,best_acc_list=[],[],[],[]

    train_set=DataSetCNN(train_image,train_label,train_feature)
    test_set=DataSetCNN(test_image,test_label,test_feature)
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

def dataAugmentation(train,label,n=100):
    for i in np.unique(label):
        idx=np.where(label==i)[0]
        if idx.shape[0]>=n:
            continue
        augNum=n-idx.shape[0]
        try:
            idx=np.random.choice(idx,size=augNum,replace=False)
        except:
            idx=np.random.choice(idx,size=augNum,replace=True)
        rData=train[idx]
        noisy=np.random.randn(augNum,30)*0.01
        rData=rData+noisy
        rData=(rData.T-np.min(rData,axis=1))/(np.max(rData,axis=1)-np.min(rData,axis=1))
        rData=rData.T
        # if not i in [0,5,6,11]:
        #     idx1=np.random.choice(range(rData.shape[0]),size=20)
        #     rData1=np.roll(rData[idx1],15,axis=1)
        #     train=np.concatenate((train,rData1))
        #     label=np.concatenate((label,np.array([i]*rData1.shape[0])))
        train=np.concatenate((train,rData))
        label=np.concatenate((label,np.array([i]*augNum)))
    return train,label

def dataAugmentation1(train,label,mode=['reverse','gauss','roll']):
    # 0翻转成6,  1翻转成7, ....
    newdata=[]
    newlabel=[]
    np.random.seed(40)
    for i in range(6):
        j=i+6
        idx1=np.where(label==i)[0]
        idx2=np.where(label==j)[0]
        if "reverse" in mode:
            data1=train[idx1]
            data2=train[idx2]
            data1=-data1
            data2=-data2
            data1=(data1.T-np.min(data1,axis=1))/(np.max(data1,axis=1)-np.min(data1,axis=1))
            data1=data1.T
            data2=(data2.T-np.min(data2,axis=1))/(np.max(data2,axis=1)-np.min(data2,axis=1))
            data2=data2.T
            newdata.append(data1)
            newlabel.append(np.array([j]*data1.shape[0]))
            newdata.append(data2)
            newlabel.append(np.array([i]*data2.shape[0]))
        if "gauss" in mode:
            data1=train[idx1]
            data2=train[idx2]
            data1=data1+np.random.randn(data1.shape[0],30)*0.002
            data2=data2+np.random.randn(data2.shape[0],30)*0.002
            data1=(data1.T-np.min(data1,axis=1))/(np.max(data1,axis=1)-np.min(data1,axis=1))
            data1=data1.T
            data2=(data2.T-np.min(data2,axis=1))/(np.max(data2,axis=1)-np.min(data2,axis=1))
            data2=data2.T
            newdata.append(data1)
            newlabel.append(np.array([i]*data1.shape[0]))
            newdata.append(data2)
            newlabel.append(np.array([j]*data2.shape[0]))
        if "roll" in mode:
            if i in [1,4]:
                data1=train[idx1]
                data2=train[idx2]
                for step in [-5,-3,-1,1,3,5]:
                    newlabel.append(np.array([i]*data1.shape[0]))
                    newlabel.append(np.array([j]*data2.shape[0]))
                    newdata.append(np.roll(data1,step,axis=1))
                    newdata.append(np.roll(data2,step,axis=1))
    for i in range(len(newdata)):
        train=np.concatenate((train,newdata[i]))
        label=np.concatenate((label,newlabel[i]))
    # 剧烈波动可以左右翻转，上下翻转
    idx=np.where(label==12)
    if "reverse" in mode:
        rData=train[idx]
        rData=np.roll(rData,15,axis=1)
        train=np.concatenate((train,rData))
        label=np.concatenate((label,np.array([12]*rData.shape[0])))
        rData=train[idx]
        rData=-rData
        rData=(rData.T-np.min(rData,axis=1))/(np.max(rData,axis=1)-np.min(rData,axis=1))
        rData=rData.T
        train=np.concatenate((train,rData))
        label=np.concatenate((label,np.array([12]*rData.shape[0])))
    if "gauss" in mode:
        rData=train[idx]
        noisy=np.random.randn(rData.shape[0],30)*0.002
        rData=rData+noisy
        rData=(rData.T-np.min(rData,axis=1))/(np.max(rData,axis=1)-np.min(rData,axis=1))
        rData=rData.T
        train=np.concatenate((train,rData))
        label=np.concatenate((label,np.array([12]*rData.shape[0])))
    if "roll" in mode:
        rData=train[idx]
        for i in range(1,9,2):
            train=np.concatenate((train,np.roll(rData,i,axis=1)))
            label=np.concatenate((label,np.array([12]*rData.shape[0])))
    return train,label



def kFoldTrain(train_image,test_image ,train_label, test_label,train_image_feature,test_image_feature,k=5):
    classes=13
    epochs = 100
    initial_lr = 1e-4
    batch_size = 32
    criterion = nn.CrossEntropyLoss()
    best_score = 0
    # epoch_list,train_acc_list,test_acc_list,best_acc_list=[],[],[],[]
    sfk=StratifiedKFold(k,random_state=42,shuffle=True)
    # train_set=DataSetCNN(train_image,train_label)
    test_set=DataSetCNN(test_image,test_label,test_image_feature)

    is_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if is_gpu else 'cpu')
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    # if is_gpu:
    #     model.cuda()
    #     criterion.cuda()
    # train model using train_image and train_label
    bestModel=None
    max_f,best_p,best_r,vf=0,0,0,0
    for fold,(subtrain_idx,subtest_idx) in enumerate(sfk.split(train_image,train_label)):
        subtrain,subtrain_label,subtrain_image_feature,subtest,subtest_label,subtest_image_feature=train_image[subtrain_idx],train_label[subtrain_idx],train_image_feature[subtrain_idx],train_image[subtest_idx],train_label[subtest_idx],train_image_feature[subtest_idx]
        model=CNN(classes)
        optimizer=torch.optim.Adam(model.parameters(), lr=initial_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
        subtrain_set=DataSetCNN(subtrain,subtrain_label,subtrain_image_feature)
        train_loader = DataLoader(subtrain_set, batch_size=batch_size, shuffle=True, num_workers=4)
        subtest_set=DataSetCNN(subtest,subtest_label,subtest_image_feature)
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
                out=model(b[0],b[2])
                # out=out.squeeze(dim=-1)
                batch_pred = out.data.max(1)[1]
                y_true=np.append(y_true,b[1].numpy())
                y_pred=np.append(y_pred,batch_pred.numpy())
                loss = criterion(out,b[1])
                loss.backward()
                optimizer.step()
            p1,r1,f1,_=precision_recall_fscore_support(y_true,y_pred,labels=np.unique(y_pred),average="macro")
            # print(f"precision: {p}\nrecall: {r}\nf-score: {f}")
            scheduler.step()
            # x=set(y_true)-set(y_pred)

            # validate
            with torch.no_grad():
                model.eval()
                test_loader = DataLoader(subtest_set, batch_size=batch_size, shuffle=True, num_workers=4)
                y_true,y_pred=np.array([],dtype=np.int),np.array([],dtype=np.int)
                for step,b in enumerate(test_loader):
                    if is_gpu:
                        b[0]=b[0].cuda()
                        b[1]=b[1].cuda()
                    out=model(b[0],b[2])
                    batch_pred = out.data.max(1)[1]
                    y_true=np.append(y_true,b[1].numpy())
                    y_pred=np.append(y_pred,batch_pred.numpy())
                p2,r2,f2,_=precision_recall_fscore_support(y_true,y_pred,labels=np.unique(y_pred),average="macro")
                if f2>=vf:
                    vf=f2
                    bestModel=model
                print(f"fold:{fold+1}, epoch:{epoch},vf:{round(f2,5)},best_vf:{round(vf,5)}")








        # test
        with torch.no_grad():
            bestModel.eval()
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)
            y_true,y_pred=np.array([],dtype=np.int),np.array([],dtype=np.int)
            for step,b in enumerate(test_loader):
                if is_gpu:
                    b[0]=b[0].cuda()
                    b[1]=b[1].cuda()
                out=bestModel(b[0],b[2])
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

if __name__=="__main__":
    # 根目录下运行
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
    
    # data,label=dataAugmentation(data,label,n=150)
    # np.save("data_aug",data)
    # np.save("label_aug",label)
    
# k fold
    # k=5
    # max_f=0
    # sfk=StratifiedKFold(k,random_state=40,shuffle=True)
    # # load data
    # for fold,(trainidx,testidx) in enumerate(sfk.split(data,label)):
    train_image,test_image ,train_label, test_label = train_test_split(data,label,test_size=0.25,random_state=40)
        # train_image,test_image ,train_label, test_label = data[trainidx],data[testidx],label[trainidx],label[testidx]
        # note: you should use train_image, train_label for training, apply the model to test_image to get predictions and use test_label to evaluate the predictions 
    # train_image,train_label=dataAugmentation(train_image,train_label,60)
    # train_image,train_label=dataAugmentation1(train_image,train_label,mode=["reverse"])
    # test_image,test_label=dataAugmentation(test_image,test_label,25)
    # for i in np.unique(train_label):
    #     idx=np.where(train_label==i)[0]
    #     print(idx.shape)
    train_image_feature=[]
    test_image_feature=[]
    s=time.time()
    for i in train_image:
        train_image_feature.append(feature_extractor(i))
    for i in test_image:
        test_image_feature.append(feature_extractor(i))
    train_image_feature=np.array(train_image_feature)
    test_image_feature=np.array(test_image_feature)
    model = RandomForestClassifier(n_estimators=100, random_state=10)
    model.fit(train_image_feature, train_label)
    print(time.time()-s)
    time.sleep(2)
    pred_y = model.predict(test_image_feature)
    p, r, f, _ = precision_recall_fscore_support(
        test_label, pred_y, labels=range(13), average="macro")
    print(f"precision: {p}\nrecall: {r}\nf-score: {f}")
    s=time.time()
    f=train(train_image,test_image ,train_label, test_label,train_image_feature,test_image_feature)
    print(time.time()-s)
        # max_f=max(max_f,f)
        # print(f"fold: {fold+1},f: {round(f,5)},max_f: {round(max_f,5)}")
    # train=np.concatenate((train_image,train_image_feature),axis=1)
    #print(np.array(train_data).shape, np.array(train_label).shape)



    # time.sleep(10)
    # kFoldTrain(train_image,test_image ,train_label, test_label,train_image_feature,test_image_feature,k=10)
    