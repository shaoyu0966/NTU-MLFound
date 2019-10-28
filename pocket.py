import numpy as np
import matplotlib.pyplot as plt

class Pocket:

    def load_x_y(self, file_name):
        f = open(file_name, 'r')
        data = np.genfromtxt(file_name)
        x = np.insert(data[:, 0:-1], 0, 1, axis=1)
        y = data[:, -1]
        f.close()
        return x, y

    def load_train(self, file_name):
        self.train_x , self.train_y = self.load_x_y(file_name)
    
    def load_test(self, file_name):
        self.test_x, self.test_y = self.load_x_y(file_name)
    
    def init(self, seed=0):
        self.w = np.zeros(np.shape(self.train_x)[1])
        self.best_w = np.copy(self.w)
        self.best_n_mistake = self.mistake()
        np.random.seed(seed)
    
    def run(self, n_update):
        while n_update > 0:
            idx = np.random.randint(np.shape(self.train_x)[0])
            if (np.inner(self.w, self.train_x[idx]) > 0) != (self.train_y[idx] > 0):
                self.w = self.w + self.train_y[idx] * self.train_x[idx]
                n_update -= 1
                n_mistake = self.mistake()
                if n_mistake < self.best_n_mistake:
                    self.best_w = self.w
                    self.best_n_mistake = n_mistake
        return self.w

    def mistake(self):
        n_mistake = 0
        for i in range(np.shape(self.train_x)[0]):
            n_mistake += 1 if ((np.inner(self.w, self.train_x[i]) > 0) != (self.train_y[i] > 0)) else 0
        return n_mistake
    
    def errorRate(self, best=True):
        n_error, total = 0, np.shape(self.test_x)[0]
        for i in range(total):
            w = self.best_w if best else self.w
            n_error += 1 if ((np.inner(w, self.test_x[i]) > 0) != (self.test_y[i] > 0)) else 0
        return (n_error / total)

    def experiment(self, n_update, n_trial, hist=False, best=True):
        error_rate = []
        for i in range(n_trial):
            self.init(i)
            self.run(n_update)
            error_rate.append(self.errorRate(best=best))

        if hist:
            plt.hist(error_rate)
            plt.title('Pocket: Error Rate')
            plt.xlabel('Error Rate')
            plt.ylabel('Frequency')
            plt.show()

        return np.average(error_rate)


pocket = Pocket()
pocket.load_train('hw1_7_train.dat')
pocket.load_test('hw1_7_test.dat')
# pocket.init()
# print(pocket.run(n_update=50))
print(pocket.experiment(n_update=100, n_trial=1126, hist=True, best=True))

