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
        prev = None
        curr = 0
        while prev == None or curr != prev:
            if (np.inner(self.w, self.x[self.seq[curr]]) > 0) != (self.y[self.seq[curr]] > 0):
                self.w = self.w + self.y[self.seq[curr]] * self.x[self.seq[curr]]
                self.n_update += 1
                prev = curr
            curr = (curr + 1) % np.shape(self.x)[0]
        return self.w
    
    def experiment(self, freq):
        n_update = []
        for i in range(freq):
            self.init(i)
            self.run()
            n_update.append(self.n_update)

        print(np.average(np.array(n_update)))
        plt.hist(n_update)
        plt.title('Frequency of Number of Updates')
        plt.xlabel('Number of updates')
        plt.ylabel('Frequency')
        plt.show()

pla = PLA()
pla.load_data('hw1_6_train.dat')
pla.init()
print(pla.run())
# pla.experiment(1126)

