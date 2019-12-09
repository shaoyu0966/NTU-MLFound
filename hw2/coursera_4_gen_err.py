import math
import sympy

def m_H(d_vc, N):
    return N ** d_vc

def VC(d_vc, delta, N):
    return math.sqrt(8/N * math.log(4 * m_H(d_vc, 2 * N) / delta))

def Radeacher_Panelty(d_vc, delta, N):
    t1 = math.sqrt(2/N * math.log(2 * N * m_H(d_vc, N)))
    t2 = math.sqrt(2/N * math.log(1 / delta))
    t3 = 1 / N
    return t1 + t2 + t3

def Parrondo_Van_den_Broek(d_vc, delta, N):
    epsilon = sympy.symbols('epsilon')
    return sympy.solve(epsilon - sympy.sqrt(1/N * (2 * epsilon + math.log(6 * m_H(d_vc, 2*N) / delta))), epsilon)

def Devroye(d_vc, delta, N):
    epsilon = sympy.symbols('epsilon')
    return sympy.solve(epsilon - sympy.sqrt(1/(2*N) * (4 * epsilon * (1 + epsilon) + math.log(4) + math.log(m_H(d_vc, N**2)) - math.log(delta))), epsilon)

def Varient_VC(d_vc, delta, N):
    return math.sqrt(16/N * math.log(2 * m_H(d_vc, N) / math.sqrt(delta)))

d_vc = 50
delta = 0.05
# N = 10000
N = 5

print("VC: ", VC(d_vc, delta, N))
print("Radeacher_Panelty: ", Radeacher_Panelty(d_vc, delta, N))
print("Parrondo_Van_den_Broek: ", Parrondo_Van_den_Broek(d_vc, delta, N))
print("Devroye: ", Devroye(d_vc, delta, N))
print("Varient_VC", Varient_VC(d_vc, delta, N))
