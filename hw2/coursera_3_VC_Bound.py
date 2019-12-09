import math

epsilon = 0.05
d_vc = 10

def prob(N):
    return (4 * pow(2 * N, d_vc) * math.exp(-0.125 * pow(epsilon, 2) * N))

N = [420000, 440000, 460000, 480000, 500000]
for n in N:
    print(prob(n))

# for n in range(450000, 460000):
#     if abs(prob(n) - 0.05) < 0.00002:
#         print(n, ": ", prob(n))
