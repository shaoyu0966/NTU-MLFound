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
        error = self.size - correct.sum()
        return error/self.size
    
    def Eout(self, theta, s):
        return 0.5 + 0.3 * s * (abs(theta) - 1)

    def dicisionStump(self):
        theta_set = np.append(self.x, -1)
        s_set = [1, -1]
        self.best_theta_s = (theta_set[0], s_set[0])
        self.best_Ein = self.size
        for theta in theta_set:
            for s in s_set:
                e_in = self.Ein(theta, s)
                if e_in < self.best_Ein:
                    self.best_theta_s = (theta, s)
                    self.best_Ein = e_in
        self.best_Eout = self.Eout(self.best_theta_s[0], self.best_theta_s[1])

# ds = DicisionStump()
# ds.init(20, 0.2)
# ds.generateData()
# ds.dicisionStump()

Ein = []
Eout = []
for i in range(5000):
    ds = DicisionStump()
    ds.init(20, 0.2, seed=i)
    ds.generateData()
    ds.dicisionStump()
    Ein.append(ds.best_Ein)
    Eout.append(ds.best_Eout)

print(np.average(Ein))
print(np.average(Eout))