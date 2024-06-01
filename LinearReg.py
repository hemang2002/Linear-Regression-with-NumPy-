# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 22:40:33 2024

@author: hhmso
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import sys
import time

from sklearn.linear_model import LinearRegression

class LinearReg:
            
    # define hyperparameters
    def __init__(self, epoch = 10000, lr_rate = 0.0001, display_rate = 50, default_rate = 0.001, display_Epoch = False):
        
        # hyperparameters
        self.lr_rate = lr_rate
        self.display_rate = display_rate
        self.epoch = epoch
        self.default_rate = default_rate
        self.display_rate = display_rate
        self.display_Epoch = display_Epoch
        
    # Define data
    def dataRead(self, x, y):
        
        try:
            
            self.m, self.n = x.shape
            
            self.x = x.T
            self.y = y.T
        
            # Weights
            self.w = np.random.uniform(low = 0, high = 15, size = (1, self.n))
            self.b = np.random.uniform(low = 0, high = 15, size = (1, ))
        
        except:
            print("Error in fedding data into model")
        
        
    # Getting linear equation 
    def lineEq(self, x):
        return np.dot(self.w, self.x) + self.b 
    
    
    # Loss
    def getLoss(self, y_pred, y_actual):
        return np.square(np.subtract(y_actual, y_pred))/self.m


    def gradientDecent(self):
        
        try:
            
            temp = np.sum(np.sqrt(self.getLoss(self.lineEq(self.x), self.y).mean()))
            self.b = self.b - self.lr_rate * 2 * temp
            self.w = self.w - self.lr_rate * 2 * self.w * temp
                        
        except:
            print("Error in gradient decent")
            
    
    # Display Epoch and loss
    def displayEpoch(self, e, loss):
        
        progress = int((e + 1) / self.epoch * 100)
        num_progress_equals = int((e + 1) / self.epoch * 30)
    
        sys.stdout.write('\n')
        sys.stdout.write("[%-30s] %d%%" % ('=' * num_progress_equals, progress))
        sys.stdout.flush()
        
        sys.stdout.write(", epoch %d, Loss: %.3f" % (e + 1, loss))
        sys.stdout.flush()
    
    
    # main function to fit
    def fit(self, x, y):
        
        try:
            
            self.dataRead(x, y)
            self.loss = np.array(0.0)
            self.dis_ep = np.array(0.0)
            
            for e in range(0, self.epoch):
                
                self.gradientDecent()
                
                if e % self.display_rate == 0:
                    
                    self.loss = np.append(self.loss, [np.sqrt(np.sum(self.getLoss(self.lineEq(self.x), self.y)))])
                    self.dis_ep = np.append(self.dis_ep,[e])
                    if self.display_Epoch == True:
                        self.displayEpoch(e, self.loss[-1])
                    
                    if abs(self.loss[-1] - self.loss[-2]) < self.default_rate:
                        break
                    
            self.plotModel(self.loss[1:], self.dis_ep[1:])
            return self
            
        except:
            print("Error in fitting model")


    # plot loss vs epoch
    def plotModel(self, loss, dis_ep):
        
        plt.rcParams['font.size'] = 50
        plt.figure(figsize = (20, 20))
        plt.plot(dis_ep, loss, color = "blue", alpha = 0.7, marker = "o", label = "Loss")
        plt.xlabel("Epoch")
        plt.xlabel("Loss per epoch")
        plt.title("Relation Between Loss with resepct to each epoch")
        plt.legend()
        plt.grid()
        plt.show()
        
        
if __name__ == "__main__":
    
    df = pd.read_csv("Dataset\forestfires.csv")
    df.drop(["month", "day"], axis = 1, inplace = True)
    x, y = df.iloc[:400, :-1], df.iloc[:400, -1:]
    
    s_time = time.time()
    model = LinearReg(display_Epoch = True).fit(x.to_numpy(), y.to_numpy())
    print(model.loss[-1])
    print(s_time - time.time())
    pred = model.lineEq(df.iloc[400:, :-1])
    print(np.sqrt(np.square(np.subtract(pred, df.iloc[400:, -1:].to_numpy())).mean()))
    
    s_time = time.time()
    a = LinearRegression().fit(x.to_numpy(), y.to_numpy())
    pred2 = a.predict(x.to_numpy())
    print(np.sqrt(np.square(np.subtract(a.predict(x.to_numpy()), y.to_numpy())).mean()))
    print(s_time - time.time())
