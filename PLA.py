# read training data
# init
# set random sequence
# run
# histogram

import numpy as np

class PLA:

    def load_data(self, file_name):
        f = open(file_name, 'r')
        data = np.genfromtxt(file_name)
        self.x = np.insert(data[:, 0:-1], 0, 1, axis=1)
        self.y = data[:, -1]
        f.close()

pla = PLA()
pla.load_data('hw1_6_train.dat')


