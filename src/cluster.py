import numpy as np
import pandas as pd
import random
import time
from tqdm import tqdm

# def kMeans(dataSet,n=5):
#     centers=random.sample(range(dataSet.shape[0]),n)
#     clusters={}
#     for i in centers:
#         clusters[i]=[i]
#     # 直到收敛
#     epoch=1
#     while True:
#         start=time.time()
#         # 计算每个非中心点到中心点的距离，选一个加入
#         pbar=tqdm(range(dataSet.shape[0]))
#         for i in pbar:
#             # 中心点不计算
#             if i in centers:
#                 continue
#             index=join(centers,dataSet[centers],dataSet[i])
#             clusters[index].append(i)
        
#         # 寻找新的中心点
#         newCenters=getNewCenters(clusters,dataSet)
#         if sorted(centers)==sorted(newCenters):
#             break
#         else:
#             centers=newCenters
#             clusters={}
#             for i in centers:
#                 clusters[i]=[i]
#         print(f"完成第{epoch}轮聚类, 耗时{time.time()-start}s")
#         epoch+=1
#     return clusters


# def join(centers,centerData,point):
#     # 计算这个点与每个center的距离，选择一个最近的加入
#     dist=[]
#     for center in centerData:
#         dist.append(dtw(center,point))
#     return centers[np.argmin(dist)]

# def getNewCenters(clusters,dataSet):
#     centers=[]
#     for cluster in clusters.values():
#         centers.append(getCenter(cluster,dataSet))
#     return centers


# def getCenter(cluster,dataSet):
#     # 计算每个点到其他各点的距离和，最小的为新center
#     dist=[]
#     # 记录在dataSet里的index
#     index=[]
#     clusterData=dataSet[cluster]
#     for i in range(len(cluster)):
#         index.append(cluster[i])
#         sum=0
#         for j in range(len(cluster)):
#             sum+=dtw(clusterData[i],clusterData[j])
#         dist.append(sum)
#     center=index[np.argmin(dist)]
#     return center





# ------------------------------------------------------------

def dtw(s1,s2):
    s1=(s1-np.min(s1))/(np.max(s1)-np.min(s1))
    s2=(s2-np.min(s2))/(np.max(s2)-np.min(s2))
    r, c = len(s1), len(s2)
    D0 = np.zeros((r+1,c+1))
    D0[0,1:] = np.inf
    D0[1:,0] = np.inf
    D1 = D0[1:,1:]

    for i in range(r):
        for j in range(c):
            D1[i,j] = np.abs(s1[i]-s2[j])

    M = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i,j] += min(D0[i,j],D0[i,j+1],D0[i+1,j])


    i,j = np.array(D0.shape) - 2
    p,q = [i],[j]
    while(i>0 or j>0):
        tb = np.argmin((D0[i,j],D0[i,j+1],D0[i+1,j]))
        if tb==0 :
            i-=1
            j-=1
        elif tb==1 :
            i-=1
        else:
            j-=1
        p.insert(0,i)
        q.insert(0,j)
    # print(D1[-1,-1])
    return D1[-1,-1]


class kMeansCluster:
    def __init__(self) -> None:
        self.centers=[]
        self.dataSet=[]
        self.distMap={}

    def miniBatch(self,dataSet,n=5):
        centers=random.sample(range(dataSet.shape[0]),n)
        clusters={}
        for i in centers:
            clusters[i]=[i]
        # 直到收敛
        epoch=1
        while True:
            start=time.time()
            # 计算每个非中心点到中心点的距离，选一个加入
            pbar=tqdm(range(dataSet.shape[0]))
            for i in pbar:
                # 中心点不计算
                if i in centers:
                    continue
                index=self.join(centers,i,dataSet)
                clusters[index].append(i)
            
            # 寻找新的中心点
            newCenters=self.getNewCenters(clusters,dataSet)
            if sorted(centers)==sorted(newCenters):
                break
            else:
                centers=newCenters
                clusters={}
                for i in centers:
                    clusters[i]=[i]
            print(f"完成第{epoch}轮聚类, 耗时{time.time()-start}s")
            epoch+=1
        return centers


    def join(self,centers,pointIndex,dataSet):
        # 计算这个点与每个center的距离，选择一个最近的加入
        dist=[]
        for i in range(len(centers)):
            keyList=sorted([centers[i],pointIndex])
            key=f"{keyList[0]}--{keyList[1]}"
            if self.distMap.__contains__(key):
                d=self.distMap[key]
            else:
                d=dtw(dataSet[centers[i]],dataSet[pointIndex])
                self.distMap[key]=d
            dist.append(d)
        return centers[np.argmin(dist)]

    def getNewCenters(self,clusters,dataSet):
        centers=[]
        for cluster in clusters.values():
            centers.append(self.getCenter(cluster,dataSet))
        return centers


    def getCenter(self,cluster,dataSet):
        # 计算每个点到其他各点的距离和，最小的为新center
        dist=[]
        # 记录在dataSet里的index
        index=[]
        # clusterData=dataSet[cluster]
        print("计算新center")
        pbar=tqdm(range(len(cluster)))
        for i in pbar:
            index.append(cluster[i])
            sum=0
            for j in range(len(cluster)):
                keyList=sorted([cluster[i],cluster[j]])
                key=f"{keyList[0]}--{keyList[1]}"
                if self.distMap.__contains__(key):
                    d=self.distMap[key]
                else:
                    d=dtw(dataSet[cluster[i]],dataSet[cluster[j]])
                    self.distMap[key]=d
                sum+=d
            dist.append(sum)
        center=index[np.argmin(dist)]
        return center

    def fit(self,dataSet,n=5,miniBatchPercent=None):
        # minibatch 找出中心点
        if miniBatchPercent:
            m=np.random.choice(range(dataSet.shape[0]),int(miniBatchPercent*dataSet.shape[0]),replace=False)
            miniDataset=dataSet[m]
        else:
            miniDataset=dataSet
        centers=self.miniBatch(miniDataset,n)
        # print(centers)
        # 其余点加入cluster
        clusters={}
        for i in centers:
            clusters[i]=[i]
        pbar=tqdm(range(dataSet.shape[0]))
        for i in pbar:
            index=self.join(centers,i,dataSet)
            clusters[index].append(i)
        return clusters