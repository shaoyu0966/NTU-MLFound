import numpy as np

class DicisionStump:

    def loadData(self, x, y):
        self.x = x
        self.y = y
        self.dim = np.shape(y)[0]

    def generateData(self, dim, noise, seed=0):
        np.random.seed(seed)
        self.dim = dim
        self.noise = noise
        self.x = np.random.rand(self.dim)
        self.x = self.x * 2 - 1
        self.y = list(map(lambda y_i: 1 if y_i > 0 else -1, self.x))
        self.y = np.array(list(map(lambda y_i: y_i if np.random.random_sample() > self.noise else -y_i, self.y)))
    
    def Ein(self, theta, s):
        result = np.array(list(map(lambda larger: 1 if larger else -1, self.x > theta))) * s
        correct = (result == self.y)
        error = self.dim - correct.sum()
        return error / self.dim
    
    def Eout(self, theta, s):
        return 0.5 + 0.3 * s * (abs(theta) - 1)

    def dicisionStump(self):
        theta_set = np.append(self.x, -1)
        s_set = [1, -1]
        self.best_theta_s = (theta_set[0], s_set[0])
        self.best_Ein = self.dim
        for theta in theta_set:
            for s in s_set:
                e_in = self.Ein(theta, s)
                if e_in < self.best_Ein:
                    self.best_theta_s = (theta, s)
                    self.best_Ein = e_in
        self.best_Eout = self.Eout(self.best_theta_s[0], self.best_theta_s[1])


class MultiDimDicisionStump:

    def loadData(self, file_name):
        f = open(file_name, 'r')
        data = np.genfromtxt(file_name)
        self.x = data[:, 0:-1]
        self.y = data[:, -1]
        f.close()

    def dicisionStump(self):
        self.best_dim = None
        self.best_theta_s = None
        self.best_Ein = np.shape(self.y)[0]
        for i in range(np.shape(self.x)[1]):
            ds = DicisionStump()
            ds.loadData(self.x[:, i], self.y)
            ds.dicisionStump()
            if ds.best_Ein < self.best_Ein:
                self.best_dim = i
                self.best_theta_s = ds.best_theta_s
                self.best_Ein = ds.best_Ein
    
    def testing(self, test_file):
        f = open(test_file, 'r')
        data = np.genfromtxt(test_file)
        self.test_x = data[:, 0:-1]
        self.test_y = data[:, -1]
        f.close()

        x = self.test_x[:, self.best_dim]
        theta = self.best_theta_s[0]
        s = self.best_theta_s[1]
        y_predict = np.array(list(map(lambda larger: 1 if larger else -1, x > theta))) * s
        err = (self.test_y != y_predict).sum()
        self.best_Eout =  err / np.shape(self.test_y)


mdds = MultiDimDicisionStump()
mdds.loadData('multi_dim_ds_train.dat')
mdds.dicisionStump()
mdds.testing('multi_dim_ds_test.dat')
print(mdds.best_Ein)
print(mdds.best_Eout)



# ds = DicisionStump()
# ds.generateData(20, 0.2)
# ds.dicisionStump()

# Ein = []
# Eout = []
# for i in range(5000):
#     ds = DicisionStump()
#     ds.generateData(20, 0.2, seed=i)
#     ds.dicisionStump()
#     Ein.append(ds.best_Ein)
#     Eout.append(ds.best_Eout)

# print(np.average(Ein))
# print(np.average(Eout))