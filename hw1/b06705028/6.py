import numpy as np
import matplotlib.pyplot as plt

class PLA:

    def load_data(self, file_name):
        f = open(file_name, 'r')
        data = np.genfromtxt(file_name)
        self.x = np.insert(data[:, 0:-1], 0, 1, axis=1)
        self.y = data[:, -1]
        f.close()
    
    def init(self, seed=0):
        self.w = np.zeros(np.shape(self.x)[1])
        self.seq = np.arange(np.shape(self.x)[0])
        self.n_update = 0
        np.random.seed(seed)
        np.random.shuffle(self.seq)
    
    def run(self):
        prev, curr = None, 0
        while prev == None or curr != prev:
            if (np.inner(self.w, self.x[self.seq[curr]]) > 0) != (self.y[self.seq[curr]] > 0):
                self.w = self.w + self.y[self.seq[curr]] * self.x[self.seq[curr]]
                self.n_update += 1
                prev = curr
            curr = (curr + 1) % np.shape(self.x)[0]
        return self.w
    
    def experiment(self, n_trial, hist=False):
        n_update = []
        for i in range(n_trial):
            self.init(i)
            self.run()
            n_update.append(self.n_update)

        if hist:
            plt.hist(n_update)
            plt.title('PLA: Number of Updates')
            plt.xlabel('Number of updates')
            plt.ylabel('Frequency')
            plt.show()

        return np.average(np.array(n_update))

pla = PLA()
pla.load_data('hw1_6_train.dat')
# pla.init()
# print(pla.run())
print('average number of updates:', pla.experiment(1126, hist=True))
