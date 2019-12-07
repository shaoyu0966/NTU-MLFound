import numpy as np

f = open('testdata.dat', 'w')
for i in range(10000):
    x1, x2 = np.random.random() - 0.5, np.random.random() - 0.5
    y = 1 if x1 + x2 > 0 else -1
    f.write(f'{x1} {x2} {y}\n')
f.close()