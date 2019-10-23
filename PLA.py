# read training data
# init
# set random sequence
# run
# histogram

import numpy as np
import matplotlib.pyplot as plt

class PLA:

    def load_data(self, file_name):
        f = open(file_name, 'r')
        data = np.genfromtxt(file_name)
        self.x = np.insert(data[:, 0:-1], 0, 1, axis=1)
        self.y = data[:, -1]
        f.close()
    
    def init(self):
        self.w = np.zeros(np.shape(self.x)[1])
        self.seq = np.arange(np.shape(self.x)[0])
        self.random_seq(10)
    
    def random_seq(self, seed):
        np.random.seed(seed)
        np.random.shuffle(self.seq)
    
    def run(self):
        prev = np.shape(self.x)[0] - 1
        curr = 0
        while not curr == prev:
            if not (np.inner(self.w, self.x[self.seq[curr]]) > 0) == (self.y[self.seq[curr]] > 0):
                self.w = self.w + self.y[self.seq[curr]] * self.x[self.seq[curr]]
                prev = curr
            curr = (curr + 1) % np.shape(self.x)[0]
        print(self.w)


pla = PLA()
pla.load_data('testdata.dat')
pla.init()
pla.run()


