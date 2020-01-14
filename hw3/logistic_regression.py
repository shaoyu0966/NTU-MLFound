import numpy as np
import matplotlib.pyplot as plt
import sys

class LogisticRegression:

    def load_data(self, train_data, test_data):
        f = open(train_data, 'r')
        data = np.genfromtxt(train_data)
        self.train_x = np.insert(data[:, 0:-1], 0, 1, axis=1)
        self.train_y = data[:, -1]
        f.close()

        f = open(test_data, 'r')
        data = np.genfromtxt(test_data)
        self.test_x = np.insert(data[:, 0:-1], 0, 1, axis=1)
        self.test_y = data[:, -1]
        f.close()

    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))
    
    def gradient(self):
        gradient = np.zeros(np.shape(self.train_x)[1])
        data_length = np.shape(self.train_y)[0]
        for i in range(data_length):
            gradient += self.sigmoid(-self.train_y[i] * np.inner(self.w, self.train_x[i])) * (-1 * self.train_y[i] * self.train_x[i])
        return gradient / data_length
    
    # fixed learning rate gradient descent
    def gradientDescent(self, step_size, n_iter):
        self.w = np.zeros(np.shape(self.train_x)[1])
        self.gd_Ein_his = []
        self.gd_Eout_his = []
        for i in range(n_iter):
            print('Gradient Descent Iteration', i)
            self.gd_Ein_his.append(self.Ein())
            self.gd_Eout_his.append(self.Eout())
            self.w -= step_size * self.gradient()
        return self.w
    
    # fixed learning rate stochastic gradient descent with fixed sampling sequence (cyclic)
    def SGD(self, step_size, n_iter):
        self.w = np.zeros(np.shape(self.train_x)[1])
        self.sgd_Ein_his = []
        self.sgd_Eout_his = []
        data_length = np.shape(self.train_y)[0]
        for i in range(n_iter):
            print('SGD Iteration', i)
            n = i % data_length
            self.sgd_Ein_his.append(self.Ein())
            self.sgd_Eout_his.append(self.Eout())
            self.w += step_size * self.sigmoid(-self.train_y[n] * np.inner(self.w, self.train_x[n])) * self.train_y[n] * self.train_x[n]
        return self.w
    
    def Ein(self):
        n_err = 0
        data_length = np.shape(self.train_y)[0]
        sign = lambda prob: 1 if prob > 0.5 else -1
        for i in range(data_length):
            prob = self.sigmoid(np.inner(self.w, self.train_x[i]))
            if sign(prob) != self.train_y[i]:
                n_err += 1
        return n_err / data_length

    def Eout(self):
        n_err = 0
        data_length = np.shape(self.test_y)[0]
        sign = lambda prob: 1 if prob > 0.5 else -1
        for i in range(data_length):
            prob = self.sigmoid(np.inner(self.w, self.test_x[i]))
            if sign(prob) != self.test_y[i]:
                n_err += 1
        return n_err / data_length


def main_18():
    lr = LogisticRegression()
    lr.load_data('train.dat', 'test.dat')
    w = lr.gradientDescent(0.001, 2000)
    Eout = lr.Eout()
    print(w)
    print(Eout)

def main_19():
    lr = LogisticRegression()
    lr.load_data('train.dat', 'test.dat')
    w = lr.gradientDescent(0.01, 2000)
    Eout = lr.Eout()
    print(w)
    print(Eout)

def main_20():
    lr = LogisticRegression()
    lr.load_data('train.dat', 'test.dat')
    w = lr.SGD(0.001, 2000)
    Eout = lr.Eout()
    print(w)
    print(Eout)

def main_7_8(question):
    n_iter = 2000
    lr = LogisticRegression()
    lr.load_data('train.dat', 'test.dat')
    lr.gradientDescent(0.01, n_iter)
    gd_Ein = lr.gd_Ein_his
    gd_Eout = lr.gd_Eout_his

    lr.SGD(0.001, n_iter)
    sgd_Ein = lr.sgd_Ein_his
    sgd_Eout = lr.sgd_Eout_his

    if question == 7:
        plt.plot(range(n_iter), gd_Ein, 'r', label='Gradient Descent')
        plt.plot(range(n_iter), sgd_Ein, 'b', label='SGD')
        plt.title('Ein')
        plt.xlabel('Iteration')
        plt.ylabel('Ein')
        plt.legend()
        plt.show()
    elif question == 8:
        plt.plot(range(n_iter), gd_Eout, 'r', label='Gradient Descent')
        plt.plot(range(n_iter), sgd_Eout, 'b', label='SGD')
        plt.title('Eout')
        plt.xlabel('Iteration')
        plt.ylabel('Eout')
        plt.legend()
        plt.show()
    else:
        plt.plot(range(n_iter), gd_Eout, 'o', label='Gradient Descent - Eout')
        plt.plot(range(n_iter), sgd_Eout, 'g', label='SGD - Eout')
        plt.plot(range(n_iter), gd_Ein, 'r', label='Gradient Descent - Ein')
        plt.plot(range(n_iter), sgd_Ein, 'b', label='SGD - Ein')
        plt.title('Ein')
        plt.xlabel('Iteration')
        plt.ylabel('Ein')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Execute as \'python3 ./logistic_regression.py question#\'')
        sys.exit()
    
    if int(sys.argv[1]) == 7:
        main_7_8(7)
    elif int(sys.argv[1]) == 8:
        main_7_8(8)
    elif int(sys.argv[1]) == 18:
        main_18()
    elif int(sys.argv[1]) == 19:
        main_19()
    elif int(sys.argv[1]) == 20:
        main_20()
