import numpy as np
from numpy import sin, cos
import control
import sympy as sp

class LinearizedCartpole():
    '''
    Holds dynamics for cartpole linerized about fixed point at theta = 0, upright
    state: [x, x', th, th']
    u: fx
    '''
    def __init__(self) -> None:
        m_c = 1 #0.5  # cart mass
        m_p = 1 #0.2  # pendulum mass
        mu = 0 #0.1 # coeff. friction
        I = 1 #0.006   # MOI of pendulum
        g = 9.8 # gravity
        l = 1 #0.3 # pendulum length
        self.l = l
        self.dt = 0.1

        p = I*(m_c+m_p) + m_c*m_p*(l**2)

        self.A = np.array([
            [0, 1, 0, 0],
            [0, -(I + m_p*l**2)*mu/p, (m_p**2*g*l**2)/p, 0],
            [0, 0, 0, 1],
            [0, -(m_p*l*mu)/p, m_p*g*l*(m_p+m_c)/p, 0]
        ])

        self.B = np.array([
            [0],
            [(I + m_p*l**2)/p],
            [0],
            [m_p*l/p]
        ])

        self.C = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ])

        self.D = np.array([
            [0],
            [0]
        ])

        self.sys = control.StateSpace(self.A, self.B, self.C, self.D)
        self.sys_disc = control.c2d(self.sys, self.dt, method='zoh')
        self.A_zoh = np.array(self.sys_disc.A)
        self.B_zoh = np.array(self.sys_disc.B)

class BetterLinearizedCartpole():
    '''
    Holds A,B,C,D matrices for the cartpole linearized about
    the fixed point at x*=[0, pi, 0, 0], u*=0 (upright) in error
    coordiantes (x̄ = Ax̄ + Bū)
    - A and B are the CT matrices
    - A_zoh and B_zoh are the DT matrices
    discretized with zero-order hold
    '''
    def __init__(self) -> None:
        m_c = 1
        m_p = 1
        l = 1
        g = 9.8
        dt = 0.1
        self.m_c = m_c
        self.m_p = m_p
        self.l = l
        self.g = g
        self.dt = dt
    
        self.A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, g*(m_p/m_c), 0, 0],
            [0, (g*(m_c+m_p))/(l*m_c), 0, 0],
        ])

        self.B = np.array([
            [0],
            [0],
            [1/m_c],
            [1/(l*m_c)]
        ])

        self.C = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        self.D = np.array([
            [0],
            [0]
        ])

        sys = control.StateSpace(self.A, self.B, self.C, self.D)
        sys_disc = control.c2d(sys, dt, method='zoh')
        self.A_zoh = np.array(sys_disc.A)
        self.B_zoh = np.array(sys_disc.B)

    def f(self, x_, u):
        x, th, xd, thd, fx = x_[0], x_[1], x_[2], x_[3], u[0]
        m_c = self.m_c
        m_p = self.m_p
        l = self.l
        g = self.g
        return np.array([
            xd,
            thd,
            1/(m_c+m_p*sin(th)**2) * (fx + m_p*sin(th)*(l*thd + g*cos(th))),
            1/(l*(m_c+m_p*sin(th)**2)) * (-fx*cos(th) - m_p*l*thd**2*cos(th)*sin(th) - (m_c+m_p)*g*sin(th))
        ])
