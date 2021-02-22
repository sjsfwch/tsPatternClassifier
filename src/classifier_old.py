import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import warnings

SIGMA=3
MULTISPIKETHRESHOLD=3
SPIKETHRESHOLD=3

"""
pattern 预定义
-3: 分类出错
-2: 平稳
-1: 未知
0:  突增
1:  向上突刺
2:  突增然后保持
3:  凸型
4:  密集向上突刺
5:  缓慢上升
6:  突降
7:  向下突刺
8:  突降然后保持
9:  凹形
10: 密集向下突刺
11: 缓慢下降
12: 模式渐变
13: 剧烈波动
14: 突增突降
"""

class StatClassifier:
    def __init__(self,windowSize,step,threshold) -> None:
        """
        windowSize: 观察pattern时滑动窗口的大小
        step: 观察pattern时滑动窗口移动的速度
        threshold: 数据做归一化或者过滤时的阈值区间 e.g. (20,80)表示取数据的20%分位点为下限, 80%分位点为上限进行归一化或者过滤
        """
        self.windowSize=windowSize
        self.step=step
        self.threshold=threshold
        self.patterns={}


    def setPattern(self,patternFilePath=None) -> None:
        if not patternFilePath:
            raise ValueError("pattern file path could not be None")
        if not os.path.exists(patternFilePath) and os.path.isfile(patternFilePath):
            raise FileNotFoundError("could not find pattern file: {}".format(patternFilePath))
        try:
            with open(patternFilePath,'r') as f:
                self.patterns=json.load(f)
        except:
            raise Exception("pattern file must be json")


    def classifyData(self,data:np.ndarray,patternFilePath,normalize=False,filter=False)->list:
        """
        data:需要分类的整体数据
        filter: 按数据的分位点进行过滤
        """
        saves={}
        count=0
        lower=np.percentile(data,self.threshold[0])
        upper=np.percentile(data,self.threshold[1])
        # windowSize 必须大于5
        if self.windowSize<5:
            warnings.warn("window size must larger than 5, now is {}".format(self.windowSize))
        # 设置pattern
        self.setPattern(patternFilePath)
        # 数据归一化
        if normalize:
            data=(data-lower)/(upper-lower)
        # 分窗口分类
        result=[]
        for i in range(0,len(data),self.step):
            if i+self.windowSize<len(data):
                subData=data[i:i+self.windowSize]
                flag=np.where((subData<lower)&(subData>upper))[0]
                if filter:
                    if not np.any(flag):
                        # 平稳
                        p=-2
                    else:
                        p=self.classify(subData,data,i)
                else:
                    p=self.classify(subData,data,i)
                result.append((i,i+self.windowSize,p))
                saves[count]={"start":i,"end":i+self.windowSize,"pattern":p,"data":list(subData)}
            else:
                # 越界直接从最后一个数据取一个窗口
                subData=data[-1*self.windowSize:]
                flag=np.where((subData<lower)&(subData>upper))[0]
                if filter:
                    if not np.any(flag):
                        # 平稳
                        p=-2
                    else:
                        p=self.classify(subData,data,len(data)-self.windowSize)
                else:
                    p=self.classify(subData,data,len(data)-self.windowSize)
                # p=self.classify(subData,data,len(data)-self.windowSize)
                result.append((len(data)-self.windowSize,len(data),p))
                saves[count]={"start":len(data)-self.windowSize,"end":len(data),"pattern":p,"data":list(subData)}
            count+=1
        with open("testdata","w") as f:
            json.dump(saves,f,indent=2,ensure_ascii=False)
        return result

    # def classify(self,data)->int:
    #     if len(data)!=self.windowSize:
    #         raise Exception("data lenth({}) must equal to classifier.windowSize({})".format(len(data),self.windowSize))
    #     if len(data)<5:
    #         warnings.warn("windowSize must larger than 5")
    #         return -3
    #     data=np.array(data)
    #     # 先通过mean，std过滤一下平稳的数据，提高效率
    #     mean,std=np.mean(data),np.std(data)
    #     if std<mean*0.2:
    #         return -2
    #     # 判断缓慢变化
    #     t=self.checkSlowChange(data)
    #     if t>0:
    #         return t
    #     # 判断上升类、下降类、波动类
    #     subMean=np.mean(sorted(data[1:-1]))
    #     subStd=np.std(sorted(data[1:-1]))
    #     ksigmas=(data-subMean)/subStd
    #     ups=np.where(ksigmas>=SIGMA)[0]
    #     downs=np.where(ksigmas<=-SIGMA)[0]
    #     anomalyType=2 # 0: 上升类，1: 下降类， 2:波动类
    #     if len(ups)>0:
    #         if len(downs)==0:
    #             anomalyType=0
    #     else:
    #         if len(downs)>0:
    #             anomalyType=1
        
    #     # 分别处理三个类别
    #     if anomalyType==0:
    #         t=self.processUpType(data)
    #     elif anomalyType==1:
    #         t=self.processDownType(data)
    #     else:
    #         t=self.processWaveType(data)
    #     return t

    

    def processUpType(self,data,ksigmas=None) -> int:
        # 若ksigmas没被传进来，计算ksigmas
        if not ksigmas:
            subMean=np.mean(sorted(data[1:-1]))
            subStd=np.std(sorted(data[1:-1]))
            ksigmas=(data-subMean)/subStd
        if self.checkMultiSpikeUp(ksigmas):
            # 密集尖刺
            return 4
        ups=np.where(ksigmas>=SIGMA)[0]
        ksigmas[ksigmas<0]=0
        # 判断是否还处于高位
        if ksigmas[-1]>=SIGMA:
            # 处于高位，判断是平缓还是在上升
            if ksigmas[-2]<ksigmas[-1]*0.75:
                # 突增
                return 0 
            else:
                # 突增然后保持
                return 2
        else:
            if len(ups)>=MULTISPIKETHRESHOLD:
                # 凸起
                return 3
            else:
                # spike
                return 1


    def processDownType(self,data,ksigmas=None) ->int:
        # 若ksigmas没被传进来，计算ksigmas
        if not ksigmas:
            subMean=np.mean(sorted(data[1:-1]))
            subStd=np.std(sorted(data[1:-1]))
            ksigmas=(data-subMean)/subStd
        if self.checkMultiSpikeDown(ksigmas):
            # 密集尖刺
            return 10
        ups=np.where(ksigmas<=-SIGMA)[0]
        ksigmas[ksigmas>0]=0
        # 判断是否还处于低位
        if ksigmas[-1]<=-SIGMA:
            # 处于低位，判断是平缓还是在下降
            if ksigmas[-2]>ksigmas[-1]*0.75:
                # 突降
                return 6
            else:
                # 突降然后保持
                return 8
        else:
            if len(ups)>=MULTISPIKETHRESHOLD:
                # 凸起
                return 9
            else:
                # spike
                return 7

    def processWaveType(self,data) ->int:
        # 滑动窗口计算subMean,subStd
        subMeans,subStds=[],[]
        wndSize=5
        count=0
        for i in range(0,len(data)-wndSize):
            subMean=np.mean(data[i:i+wndSize])
            subStd=np.std(data[i:i+wndSize])
            if subMean<0.2:
                count+=1
            subMeans.append(subMean)
            subStds.append(subStd)
        
        if count>3:
            # 模式渐变
            return 12
        if np.std(np.array(subMeans)) < 0.1:
            # 突增突降
            return 14
        # 剧烈波动
        return 13


    def checkMultiSpikeUp(self,ksigmas) -> bool:
        # spike最多持续3个点, endIdx-startIdx<4
        count,interval=0,0
        flag=False
        for i in range(len(ksigmas)):
            if (ksigmas[i]>=SIGMA):
                flag=True
                interval+=1
            else:
                # 结束spike
                if flag:
                    if interval<=SPIKETHRESHOLD:
                        count+=1
                    flag=False
                    interval=0
                else:
                    continue
        if count>=MULTISPIKETHRESHOLD:
            return True
        else:
            return False


    def checkMultiSpikeDown(self,ksigmas) -> bool:
        # spike最多持续3个点, endIdx-startIdx<4
        count,interval=0,0
        flag=False
        for i in range(len(ksigmas)):
            if (ksigmas[i]<=-SIGMA):
                flag=True
                interval+=1
            else:
                # 结束spike
                if flag:
                    if interval<=SPIKETHRESHOLD:
                        count+=1
                    flag=False
                    interval=0
                else:
                    continue
        # 收尾
        if interval>0:
            count+=1
        if count>=MULTISPIKETHRESHOLD:
            return True
        else:
            return False


    def classify(self,data,all,start)->int:
        if len(data)!=self.windowSize:
            raise Exception("data lenth({}) must equal to classifier.windowSize({})".format(len(data),self.windowSize))
        if len(data)<5:
            warnings.warn("windowSize must larger than 5")
            return -3
        data=np.array(data)
        all=np.array(all)
        statStart=max(0,start-12*60)
        # 先通过mean，std过滤一下平稳的数据，提高效率
        mean,meanAll,stdAll=np.mean(data),np.mean(all[statStart:start]),np.std(all[statStart:start])
        if np.isnan(meanAll):
            meanAll=np.mean(sorted(data[1:-1]))
        if np.isnan(stdAll):
            stdAll=np.std(sorted(data[1:-1]))
        if np.abs(mean-meanAll)/meanAll<0.2:
            return -2
        # 判断缓慢变化
        t=self.checkPattern5(data)
        if t>=0:
            return t
        t=self.checkPattern11(data)
        if t>=0:
            return t
        # 判断上升类、下降类、波动类
        # subMean=np.mean(sorted(data[1:-1]))
        # subStd=np.std(sorted(data[1:-1]))
        ksigmas=(data-meanAll)/stdAll
        ups=np.where(ksigmas>=SIGMA)[0]
        downs=np.where(ksigmas<=-SIGMA)[0]
        if len(ups)==0 and len(downs)==0:
            return -2
        anomalyType=2 # 0: 上升类，1: 下降类， 2:波动类
        if len(ups)>0:
            if len(downs)==0:
                anomalyType=0
        else:
            if len(downs)>0:
                anomalyType=1
        
        # 分别处理三个类别
        if anomalyType==0:
            t=self.checkPattern024(data,ksigmas)
            if t>=0:
                return t
            t=self.checkPattern134(data,ksigmas)
            if t>=0:
                return t
            return -1
        elif anomalyType==1:
            t=self.checkPattern6810(data,ksigmas)
            if t>=0:
                return t
            t=self.checkPattern7910(data,ksigmas)
            if t>=0:
                return t
            return -1
        else:
            t=self.checkPattern1314(data,ksigmas)
            if t>=0:
                return t
            t=self.checkPattern12(data)
            return t
        return t

    
    def checkSlowChange(self,data) -> bool:
        upCount,downCount=0,0
        for i in range(1,len(data)):
            if data[i]-data[i-1]>0:
                upCount+=1
            elif data[i]-data[i-1]<0:
                downCount+=1
        if upCount>2*downCount and data[-1]/data[0]>1.5:
            # 缓慢上升
            return 5
        
        if downCount>2*upCount and data[-1]/data[0]<0.66:
            # 缓慢下降
            return 11

        return -1


    def checkPattern024(self, data, ksigmas) -> int:
        """
        突增类型，数据仍维持在高位，但相较于前几个点，仍在增长
        突增然后保持，数据仍维持在高位，但较于前几个点，变化不大
        """

        # 数据最后一个点未维持在高位，不是突增
        if ksigmas[-1] < SIGMA:
            return -1
        if data[-1]/np.max(data)<0.8:
            return -1
        if data[-1]==np.max(data):
            return 0
        anomalyIndex=np.where(ksigmas>=SIGMA)[0]
        spikeCount=0
        for i in range(1,len(anomalyIndex)):
            if anomalyIndex[i]-anomalyIndex[i-1]>1:
                spikeCount+=1
        # 说明有两个以上的突刺/凸起
        if spikeCount>0:
            return 4
        if len(anomalyIndex)>2:
            return 2
        else:
            return 0
        

    def checkPattern134(self, data, ksigmas) -> int:
        """
        突刺：全局仅有不超过2个的连续的异常点
        凸起：全局仅有一个连续的异常段
        密集突刺：全局有多个不连续的异常点/段
        """
        # 数据最后一个点维持在高位，不是这3种类别
        if ksigmas[-1] > SIGMA and data[-1]>np.max(data)*0.6:
            return -1
        anomalyIndex=np.where(ksigmas>=SIGMA)[0]
        spikeCount=0
        for i in range(1,len(anomalyIndex)):
            if anomalyIndex[i]-anomalyIndex[i-1]>1:
                spikeCount+=1
        # 说明有两个以上的突刺/凸起
        if spikeCount>1:
            return 4
        if len(anomalyIndex)>2:
            return 3
        else:
            return 1

    # def checkPattern2(self, data) -> bool:
    #     pass

    # def checkPattern3(self, data) -> bool:
    #     pass

    # def checkPattern4(self, data) -> bool:
    #     pass

    def checkPattern5(self, data) -> int:
        """
        缓慢上升：数据一直在上升，可以有下降，但不能连续下降超过3次且幅度不能超过5%
        """
        downcount=0
        upcount=0
        for i in range(1,len(data)):
            if data[i]-data[i-1]<0:
                change=np.abs(data[i]-data[i-1])/data[i-1]
                if change<0.2:
                    downcount+=1
                    if downcount>3:
                        break
                else:
                    if upcount>10:
                        return -1
                    else:
                        upcount=0
            else:
                downcount=0
                upcount+=1

        if downcount>3:
            return -1
        if upcount>10:
            return 5
        return -1



    def checkPattern6810(self, data, ksigmas) -> int:
        """
        同上升
        """

        # 数据最后一个点未维持在高位，不是突增
        if ksigmas[-1] < -SIGMA:
            return -1
        anomalyIndex=np.where(ksigmas<=-SIGMA)[0]
        spikeCount=0
        for i in range(1,len(anomalyIndex)):
            if anomalyIndex[i]-anomalyIndex[i-1]>1:
                spikeCount+=1
        # 说明有两个以上的突刺/下凹
        if spikeCount>0:
            return 10
        if len(anomalyIndex)>2:
            return 8
        else:
            return 6

    def checkPattern7910(self, data, ksigmas) -> int:
        """
        同上升
        """
        if ksigmas[-1] > -SIGMA:
            return -1
        anomalyIndex=np.where(ksigmas<=-SIGMA)[0]
        spikeCount=0
        for i in range(1,len(anomalyIndex)):
            if anomalyIndex[i]-anomalyIndex[i-1]>1:
                spikeCount+=1
        # 说明有两个以上的突刺/下凹
        if spikeCount>0:
            return 10
        if len(anomalyIndex)>2:
            return 9
        else:
            return 7

    # def checkPattern8(self, data) -> bool:
    #     pass

    # def checkPattern9(self, data) -> bool:
    #     pass

    # def checkPattern10(self, data) -> bool:
    #     pass

    def checkPattern11(self, data) -> bool:
        """
        同下降
        """
        count=0
        for i in range(1,len(data)):
            if data[i]-data[i-1]>0:
                if np.abs(data[i]-data[i-1])/data[i-1]<0.2 :
                    count+=1
                    if count>3:
                        break
                else:
                    return -1
            else:
                count=0

        if count>3:
            return -1
        return 11

    def checkPattern12(self, data) -> bool:
        """
        模式渐变：数据从一种均值，渐变到另一种均值

        """
        mean1,std1=np.mean(data[:5]),np.std(data[5])
        mean2,std2=np.mean(data[-5:]),np.std(data[-5:])
        if np.abs(mean1-mean2)/np.abs(mean1)>=0.2:
            return 12
        return -1

    # def checkPattern13(self, data) -> bool:
    #     pass

    def checkPattern1314(self, data, ksigmas) -> bool:
        """
        突增突降：窗口内必有一个突增和一个突降
        剧烈波动：窗口内有多个突增和突降
        """
        upIndex=np.where(ksigmas>=SIGMA)[0]
        downIndex=np.where(ksigmas<=-SIGMA)[0]
        if len(upIndex)<1 or len(downIndex)<1:
            return -1
        
        upSpike,downSpike=0,0
        for i in range(1,len(upIndex)):
            if upIndex[i]-upIndex[i-1]>1:
                upSpike+=1
        
        for i in range(1,len(downIndex)):
            if downIndex[i]-downIndex[i-1]>1:
                downIndex+=1
        if upSpike==0 and downSpike==0:
            return 14
        else:
            return 13





# --------------------分割线-------------------------

def plotByRes(res,data,normalizeThreshold,saveDir=None,patterns={}):
    count=0
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    # lower=np.percentile(data,normalizeThreshold[0])
    # upper=np.percentile(data,normalizeThreshold[1])
    # data=(data-lower)/(upper-lower)
    for r in res:
        plt.figure()
        if patterns:
            plt.title(patterns[str(r[2])]["中文名称"])
        else:
            plt.title(r[2])
        # plt.ylim(np.min(data),np.max(data))
        plt.plot(data[r[0]:r[1]])
        if saveDir:
            if not (os.path.exists(saveDir) and os.path.isdir(saveDir)):
                os.mkdir(saveDir)
            plt.savefig(f"{saveDir}/{count}.jpg")
        plt.close()
        count+=1
        if count>400:
            break

def getDiff(data)->list:
    diff=[]
    for i in range(1,len(data)):
        diff.append(data[i]-data[i-1])
    return diff


def debug(cls:StatClassifier,data):
    index="20"
    with open("testdata","r") as f:
        res=json.load(f)
    p=cls.classify(res[index]["data"],data,res[index]["start"])
    print(cls.patterns[str(p)]["中文名称"])

if __name__=="__main__":
    # data=pd.read_csv("/Users/sjsfwch/Desktop/异常检测/src/db12/metric.platform.by18qdfm1db1002")["AAS_TOTAL"].dropna().values
    # np.save("AAS_TOTAL",data)
    data=np.load("AAS_TOTAL.npy")
    cls=StatClassifier(30,5,(0,99.5))
    cls.setPattern("pattern.json")
    debug(cls,data) 
    res=cls.classifyData(data,"pattern.json")
    # print(res)
    plotByRes(res,data,(1,99.5),"test",cls.patterns)

