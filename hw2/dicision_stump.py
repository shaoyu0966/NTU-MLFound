import numpy as np

class DicisionStump:

    def init(self, size, noise, seed=0):
        self.size = size
        self.noise = noise

    def generateData(self):
        self.x = np.random.rand(self.size)
        self.x = self.x * 2 - 1
        self.y = list(map(lambda y_i: 1 if y_i > 0 else -1, self.x))
        self.y = np.array(list(map(lambda y_i: y_i if np.random.random_sample() > self.noise else -y_i, self.y)))
    
    def Ein(self, theta, s):
        result = np.array(list(map(lambda larger: 1 if larger else -1, self.x > theta))) * s
        correct = (result == self.y)
        return (self.size - correct.sum())

    def dicisionStump(self):
        theta_set = np.append(self.x, -1)
        s_set = [1, -1]
        self.best_theta_s = (theta_set[0], s_set[0])
        self.best_Ein = self.size
        for theta in theta_set:
            for s in s_set:
                ein = self.Ein(theta, s)
                if ein < self.best_Ein:
                    self.best_theta_s = (theta, s)
                    self.best_Ein = ein

                
        
ds = DicisionStump()
ds.init(20, 0.2)
ds.generateData()
ds.dicisionStump()
