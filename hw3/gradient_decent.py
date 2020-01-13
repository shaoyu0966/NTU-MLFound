import numpy as np
import sympy as sp
import math

class GradientDecent:

    def functionE(self, u=0, v=0):
        self.u, self.v = sp.symbols('u v')
        self.E_ex = sp.exp(self.u) + sp.exp(2 * self.v) + sp.exp(self.u * self.v) + self.u ** 2 - 2 * self.u * self.v + 2 * self.v ** 2 - 3 * self.u - 2 * self.v
        E = sp.lambdify([self.u, self.v], self.E_ex, 'math')
        return E(u, v)

    def gradient(self):
        self.dE_du = sp.lambdify([self.u, self.v], sp.diff(self.E_ex, self.u), 'math')
        self.dE_dv = sp.lambdify([self.u, self.v], sp.diff(self.E_ex, self.v), 'math')
    
    def improve(self, step_size, u0, v0, n_iter):
        cur_u, cur_v = u0, v0
        for i in range(n_iter):
            print(i, cur_u, cur_v)
            u = cur_u - step_size * self.dE_du(cur_u, cur_v)
            v = cur_v - step_size * self.dE_dv(cur_u, cur_v)
            cur_u = u
            cur_v = v
        return cur_u, cur_v

if __name__ == '__main__':
    gd = GradientDecent()
    gd.functionE()
    gd.gradient()
    u5, v5 = gd.improve(0.01, 0.0, 0.0, 5)
    print(gd.functionE(u5, v5))