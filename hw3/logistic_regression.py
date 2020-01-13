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
        data_length = np.shape(self.x)[0]
        for i in range(data_length):
            gradient += self.sigmoid(-1 * self.y[i] * np.inner(self.w, self.x[i])) * (-1 * self.y[i] * self.x[i])
        return gradient / data_length
    
    def logisticRegression(self, step_size, n_iter):
        self.w = np.zeros(np.shape(self.x)[1])
        for i in range(n_iter):
            print('iteration', i)
            self.w -= step_size * self.gradient()
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

if __name__ == '__main__':
    lr = LogisticRegression()
    lr.load_data('train.dat')
    w = lr.logisticRegression(0.01, 2000)
    Eout = lr.evaluate('test.dat')
    print(w)
    print(Eout)