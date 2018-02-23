import matplotlib.pyplot as plt
import math
from scipy.stats import multivariate_normal
from State import State
import numpy as np

class HiddenMarkovModel:
    
    def __init__ (self, statesNum, modelName, data):
        self.allStates = list()
        self.statesNum = statesNum
        self.name = modelName
        self.normalizedData = self.normalize_data(data['X'])
        self.trainData = self.normalizedData[0:70]
        self.gaussian_init(self.trainData)
        self.testData = self.normalizedData[70:100]
        self.init_trans()   
                
    def normalize_data (self, data):
        meanX = list()
        meanY = list()
        for i in range(len(data)):
            sumX = 0.0
            sumY = 0.0
            for j in range(len(data[i])):    
                sumX = sumX + data[i][j][0]                  
                sumY = sumY + data[i][j][1]                         
            meanX.append(sumX / len(data[i]))
            meanY.append(sumY / len(data[i]))
        devX = list()
        devY = list()
        for i in range(len(data)):
            diffX = 0
            diffY = 0
            for j in range(len(data[i])):
                diffX = diffX + ((data[i][j][0] - meanX[i])**2)
                diffY = diffY + ((data[i][j][1] - meanY[i])**2)
            devX.append(math.sqrt(diffX / len(data[i])))
            devY.append(math.sqrt(diffY / len(data[i])))
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j][0] = (data[i][j][0] - meanX[i]) / devX[i]
                data[i][j][1] = (data[i][j][1] - meanY[i]) / devY[i]
        return data
    
    def init_trans(self):
        self.transMat = np.zeros((self.statesNum, self.statesNum))
        self.transMat = self.transMat.astype(dtype = np.float64)
        for i in range(self.statesNum-1):
            self.transMat[i][i] = 0.75
            self.transMat[i][i+1] = 0.25
        self.transMat[self.statesNum-1][self.statesNum-1] = 1
        
        for i in range(len(self.trainData)):
            plt.plot(*zip(*self.trainData[i]), marker = '.', color = 'r', ls = '')
        for s in self.allStates:
            x1, y1 = np.random.multivariate_normal(s.mean, s.covariance, 5000).T
            plt.plot(x1, y1, 'x')
        
        plt.show()
        
    
    def learn (self):
        
       
        #Expectaion Maximization for Gaussians
        #_____________________________________________
        zer = False
        prevCov = list()
        prevMean = list()
        prevWeight = list()
        prev = self.logLikelihood(self.trainData)
        for i in range(self.statesNum):
            prevCov.append(self.allStates[i].covariance) 
            prevMean.append(self.allStates[i].mean)
            prevWeight.append(self.allStates[i].weight)
        while True :
            #compute ric, mc and weight
            print(self.logLikelihood(self.trainData))
            for i in range(len(self.trainData)):
                plt.plot(*zip(*self.trainData[i]), marker = '.', color = 'r', ls = '')
            ric = list()
            mc = list()
            for obs in self.trainData:
                for point in obs:
                    norm = 0
                    pdfs = list()
                    for s in self.allStates:
                        cPdf = multivariate_normal.pdf(point, s.mean, s.covariance)  
                        norm = norm + s.weight*cPdf
                        pdfs.append(cPdf)            #instead of computing pdfs again, we save them in a list
                        
                    for i in range(len(self.allStates)):
                        r = (self.allStates[i].weight*pdfs[i])/norm   #assignment of ric 
                        ric.append(r)
                        if len(mc) in range(0, self.statesNum): 
                            mc.append(r)
                        else:
                            mc[i] = mc[i] + r
                    pdfs.clear()
                    
            wSum = sum(mc)
            for i in range(self.statesNum):
                self.allStates[i].weight = mc[i] / wSum
                
            #compute the mean
            muX = list()
            muY = list()
            j = 0
            for obs in self.trainData:
                for point in obs:
                    for i in range(self.statesNum):
                        if len(muX) in range(0, self.statesNum):
                            muX.append(point[0]*ric[j])
                            muY.append(point[1]*ric[j])
                        else:
                            muX[i] = muX[i] + ric[j]*point[0]
                            muY[i] = muY[i] + ric[j]*point[1]
                        j = j + 1
                        
            for i in range(self.statesNum):
                muX[i] = muX[i] / mc[i]
                muY[i] = muY[i] / mc[i]
                self.allStates[i].mean[0] = muX[i]
                self.allStates[i].mean[1] = muY[i]
                
            #compute the covariance
            cov = list()
            j = 0
            for obs in self.trainData:
                for point in obs:
                    for i in range(self.statesNum):
                        arr = np.arange(4).reshape(2,2)
                        arr[0] = np.subtract(point[0], self.allStates[i].mean[0])
                        arr[1] =  np.subtract(point[1], self.allStates[i].mean[1])
                        c = np.multiply(ric[j], np.multiply(arr, np.transpose(arr)))
                        if len(cov) in range(0, self.statesNum):
                            cov.append(c)
                        else:
                            cov[i] = np.add(cov[i], c)
                        j = j + 1
                        
            for i in range(self.statesNum):
                cov[i] = np.divide(cov[i], mc[i])
                self.allStates[i].covariance = cov[i]
                
            ric = None
            mc = None
            muX = None
            muY = None
            cov = None
            
            x1, y1 = np.random.multivariate_normal(self.allStates[0].mean, self.allStates[0].covariance, 5000).T
            plt.plot(x1, y1, 'x')
            x, y = np.random.multivariate_normal(self.allStates[1].mean, self.allStates[1].covariance, 5000).T
            plt.plot(x, y, 'x')
            print(self.logLikelihood(self.trainData))
            x2, y2 = np.random.multivariate_normal(self.allStates[2].mean, self.allStates[2].covariance, 5000).T
            plt.plot(x2, y2, 'x')
            x3, y3 = np.random.multivariate_normal(self.allStates[3].mean, self.allStates[3].covariance, 5000).T
            plt.plot(x3, y3, 'x')
        
            plt.show()
            if zer is True:
                print("zeroo")
                zer = False
            try:
                ll = self.logLikelihood(self.trainData)
                if ll < prev:
                    self.allStates[i].covariance = prevCov[i]
                    self.allStates[i].mean = prevMean[i]
                    self.allStates[i].weight = prevWeight[i]
                    break
                else:
                    prev = ll
                    for i in range(self.statesNum):
                        prevCov[i] = self.allStates[i].covariance
                        prevMean[i] = self.allStates[i].mean
                        prevWeight[i] = self.allStates[i].weight
            except:
                print("except")
                for i in range(self.statesNum):
                    self.allStates[i].covariance = prevCov[i]
                    self.allStates[i].mean = prevMean[i]
                    self.allStates[i].weight = prevWeight[i]
                break
            
            
            
                    
    def evaluate(self, obs):
        
        #forward-backward algorithm
        obs = self.normalize_data([obs])
        plt.show()        
        T = len(obs[0])
        self.alpha = np.dtype(np.float64)
        self.alpha = np.arange(T*self.statesNum, dtype=np.float64).reshape(self.statesNum, T)
        self.alpha[0][0] = 1.0
        for s in range(1,self.statesNum):
            self.alpha[s][0] = 0.0
        self.beta = np.dtype(np.float64)
        self.beta = np.arange(T*self.statesNum, dtype=np.float64).reshape(self.statesNum, T)
        for s in range(0,self.statesNum):
            self.beta[s][T-1] = 1.0
        for t in range(0,T-1):
            for i in range(self.statesNum):
                sumAlpha = 0.0
                for j in range(self.statesNum):                        
                    sumAlpha = sumAlpha + self.alpha[j][t]*(self.transMat[j][i])
                
                self.alpha[i][t+1] = float(sumAlpha*((multivariate_normal.pdf(obs[0][t+1], 
                     self.allStates[i].mean, self.allStates[i].covariance))))

        for t in range(0,T-1):
            for i in range(self.statesNum):
                sumBeta = 0
                for j in range(self.statesNum):
                        
                    sumBeta = sumBeta + (self.transMat[i][j])*(multivariate_normal.pdf(obs[0][T-t-1], 
                     self.allStates[j].mean, self.allStates[j].covariance))*self.beta[j][T-t-1]
                self.beta[i][T-t-2] = sumBeta

        prob = 0
        for t in range(T):
            for i in range(self.statesNum):
                prob = prob + self.alpha[i][t]*self.beta[i][t]
                #print (prob)
        return prob
    
    def logLikelihood(self, data):
        
        llh = list()
        for obs in data:
            sumLog = 0
            for point in obs:
                sumC = 0
                for s in self.allStates:
                    sumC = sumC + s.weight*multivariate_normal.pdf(point, s.mean, s.covariance)
                if sumC == 0.0:
                    sumC = 0.0000001
                sumLog = sumLog +  math.log(sumC)
            llh.append(sumLog)
        prob = sum(llh) / len(llh)
        return prob
    
    def gaussian_init(self, data):
        pnts = 0
        for obs in data:
            pnts = pnts + len(obs)
        pnts = int(pnts/self.statesNum)
        part = int(pnts/len(data))
        for s in range(self.statesNum):
            xp = list()
            yp = list()
            for obs in data:
                for i in range(s*part, part+s*part):
                    if (i >= len(obs)):
                        break
                    xp.append(obs[i][0])
                    yp.append(obs[i][1])
            self.allStates.append(State([np.mean(xp), np.mean(yp)], np.cov(xp, yp), 1/self.statesNum))
            xp.clear()
            yp.clear()