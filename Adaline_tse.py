import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron(object):

    def init_(self , eta= 0.01 , n_iter=50 , random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def net_input(self,X):
        return np.dot(X,self.w_[1:]) + self.w_[0]

    def predict(self,X):
        return np.where(self.net_input(X) >= 0.0 , 1 , -1 )

    def fit(self , X , y):  
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0 , scale=0.01 , size = 1 + X.shape[1])

        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi , target in zip(X,y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self




df = pd.read_csv('iris.data', header=None)
df.tail()

y = df.iloc[0:100 , 4].values
y = np.where(y== 'Iris-setosa', -1 , 1)

X = df.iloc[0:100 , [0,2]].values

ppn = Perceptron(eta=0.1,n_iter = 10)
ppn.fit(X,y)
plt.plot(range(1,len(ppn.errors_) +1 , ppn.errors_,marker='o')
plt.show()