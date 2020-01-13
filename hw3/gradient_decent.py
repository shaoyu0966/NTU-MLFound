import numpy as np
import sympy as sp
import math

class GradientDecent:

    def functionE(self, u=0, v=0):
        self.u, self.v = sp.symbols('u v')
        self.E_ex = sp.exp(self.u) + sp.exp(2 * self.v) + sp.exp(self.u * self.v) + self.u ** 2 - 2 * self.u * self.v + 2 * self.v ** 2 - 3 * self.u - 2 * self.v
        E = sp.lambdify([self.u, self.v], self.E_ex, 'math')
        return E(u, v)

    def gradient(self, u=0, v=0):
        self.dE_du_ex = sp.diff(self.E_ex, self.u)
        self.dE_dv_ex = sp.diff(self.E_ex, self.v)
        self.dE_du = sp.lambdify([self.u, self.v], self.dE_du_ex, 'math')
        self.dE_dv = sp.lambdify([self.u, self.v], self.dE_dv_ex, 'math')
        return self.dE_du(u, v), self.dE_dv(u, v)
    
    def second_order_derivative(self, u=0, v=0):
        self.d2E_duu_ex = sp.diff(self.dE_du_ex, self.u)
        self.d2E_dvv_ex = sp.diff(self.dE_dv_ex, self.v)
        self.d2E_duv_ex = sp.diff(self.dE_du_ex, self.v)
        self.d2E_duu = sp.lambdify([self.u, self.v], self.d2E_duu_ex, 'math')
        self.d2E_dvv = sp.lambdify([self.u, self.v], self.d2E_dvv_ex, 'math')
        self.d2E_duv = sp.lambdify([self.u, self.v], self.d2E_duv_ex, 'math')
        return self.d2E_duu(u, v),  self.d2E_dvv(u, v), self.d2E_duv(u, v)

    def improve(self, step_size, u0, v0, n_iter):
        cur_u, cur_v = u0, v0
        for i in range(n_iter):
            cur_u, cur_v = cur_u - step_size * self.dE_du(cur_u, cur_v), cur_v - step_size * self.dE_dv(cur_u, cur_v)
        return cur_u, cur_v

def main_7():
    gd = GradientDecent()
    gd.functionE()
    gd.gradient()
    

def main_8():
    u, v = 0, 0
    gd = GradientDecent()
    b = gd.functionE(u, v)
    b_u, b_v = gd.gradient(u, v)
    b_uu, b_vv, b_uv = gd.second_order_derivative(u, v)
    print(b_uu/2, b_vv/2, b_uv, b_u, b_v, b)

if __name__ == "__main__":
    main_8()