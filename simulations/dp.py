import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from dp_solver import DPSolver

if __name__ == '__main__':

    # number of independent variable steps
    N = 10

    # number of states
    nx = 2

    # number of inputs
    nu = 2

    # number of points in the discretized state and input spaces
    NX = 10
    NU = 10
    
    # bounds
    X1_MIN = 0.0
    X1_MAX = 1.0

    X2_MIN = 0.0
    X2_MAX = 1.0

    U1_MIN = 0.0
    U1_MAX = 1.0

    U2_MIN = 0.0
    U2_MAX = 1.0

    x_bounds = ((X1_MIN, X1_MAX), (X2_MIN, X2_MAX))
    u_bounds = ((U1_MIN, U1_MAX), (U2_MIN, U2_MAX))


    def dynamics(x,u):
        return 0.1 * x + u
    def stage_cost(x,u):
        return 0.5 * (x.T@x + u.T@u)

    solver = DPSolver(nx, nu, NX, NU, stage_cost, dynamics,\
        x_bounds=x_bounds, u_bounds=u_bounds)
    
    NDX = solver.X.shape[0]
    # optimal value function
    J_opt = np.zeros((NDX, 1))

    # loop over time
    for i in range(N):
        J_opt = solver.DPOperator(J_opt)
        

