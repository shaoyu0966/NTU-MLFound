import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:

    def load_data(self, train_data):
        f = open(train_data, 'r')
        data = np.genfromtxt(train_data)
        self.x = np.insert(data[:, 0:-1], 0, 1, axis=1)
        self.y = data[:, -1]
        f.close()

    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))
    
    def gradient(self):
        gradient = np.zeros(np.shape(self.x)[1])
        data_length = np.shape(self.y)[0]
        for i in range(data_length):
            gradient += self.sigmoid(-self.y[i] * np.inner(self.w, self.x[i])) * (-1 * self.y[i] * self.x[i])
        return gradient / data_length
    
    # fixed learning rate gradient descent
    def gradientDescent(self, step_size, n_iter):
        self.w = np.zeros(np.shape(self.x)[1])
        for i in range(n_iter):
            # print('iteration', i)
            self.w -= step_size * self.gradient()
        return self.w
    
    # fixed learning rate stochastic gradient descent with fixed sampling sequence (cyclic)
    def SGD(self, step_size, n_iter):
        self.w = np.zeros(np.shape(self.x)[1])
        data_length = np.shape(self.y)[0]
        for i in range(n_iter):
            # print('iteration', i)
            n = i % data_length
            self.w += step_size * self.sigmoid(-self.y[n] * np.inner(self.w, self.x[n])) * self.y[n] * self.x[n]
        return self.w

    def evaluate(self, test_data):
        f = open(test_data, 'r')
        data = np.genfromtxt(test_data)
        x = np.insert(data[:, 0:-1], 0, 1, axis=1)
        y = data[:, -1]
        f.close()

        n_err = 0
        data_length = np.shape(y)[0]
        sign = lambda prob: 1 if prob > 0.5 else -1
        for i in range(data_length):
            prob = self.sigmoid(np.inner(self.w, x[i]))
            if sign(prob) != y[i]:
                n_err += 1
        return n_err / data_length



def main_18():
    lr = LogisticRegression()
    lr.load_data('train.dat')
    w = lr.gradientDescent(0.001, 2000)
    Eout = lr.evaluate('test.dat')
    print(w)
    print(Eout)

def main_19():
    lr = LogisticRegression()
    lr.load_data('train.dat')
    w = lr.gradientDescent(0.01, 2000)
    Eout = lr.evaluate('test.dat')
    print(w)
    print(Eout)

def main_20():
    lr = LogisticRegression()
    lr.load_data('train.dat')
    w = lr.SGD(0.001, 2000)
    Eout = lr.evaluate('test.dat')
    print(w)
    print(Eout)

if __name__ == '__main__':
    # main_18()
    # main_19()
    main_20()