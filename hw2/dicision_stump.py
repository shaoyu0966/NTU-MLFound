import numpy as np

class DicisionStump:

    def init(self, size, noise, seed=0):
        self.size = size
        self.noise = noise

    def generateData(self):
        self.x = np.random.rand(self.size)
        self.x = self.x * 2 - 1
        self.y = np.sign(self.x)
        self.y = list(map(lambda y_i: y_i if np.random.random_sample() > self.noise else -y_i, self.y))
    
        
ds = DicisionStump()
ds.init(20, 0.2)
ds.generateData()
