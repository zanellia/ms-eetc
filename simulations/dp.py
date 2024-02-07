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
    NX = 20
    NU = 10
    
    # bounds
    X1_MIN = -1.0
    X1_MAX = 1.0

    X2_MIN = -1.0
    X2_MAX = 1.0

    U1_MIN = -1.0
    U1_MAX = 1.0

    U2_MIN = -0.5
    U2_MAX = 0.5

    x_bounds = ((X1_MIN, X1_MAX), (X2_MIN, X2_MAX))
    u_bounds = ((U1_MIN, U1_MAX), (U2_MIN, U2_MAX))


    Q = 100
    R = 0.00
    def dynamics(x,u):
        return np.vstack((x[0] + 0.8 * x[1] + 0.5 * u[0], x[1] + 0.5 * u[1]))
    def stage_cost(x,u):
        return 0.5 * (Q * (x[0]**2 + x[1]**2) + R * (u[0]**2 + u[1]**2))
    def terminal_cost(x):
        return 0.5 * (Q * (x[0]**2 + x[1]**2))
    # def constraints(x,u):
    #     return np.vstack((u[0] - 1.0, -u[0] - 1.0, u[1] - 1.0, -u[1] - 1.0)

    solver = DPSolver(nx, nu, NX, NU, stage_cost, dynamics,\
        x_bounds=x_bounds, u_bounds=u_bounds)
    
    NDX = solver.X.shape[0]
    # optimal value function
    J_opt = np.zeros((NDX, 1))

    # initialize with terminal cost
    for i in range(NDX):
        J_opt[i] = terminal_cost(np.atleast_2d(solver.X[i,:]).T)

    # loop over time
    for i in range(N):
        print('it {}'.format(i))
        J_opt, U_opt = solver.DPOperator(J_opt)

    # plot value function
    X = solver.X
    fig = plt.figure()
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    ax1.scatter(X[:,0], X[:,1], J_opt)
    ax1.set_xlabel(r"$x_1$")
    ax1.set_ylabel(r"$x_2$")
    ax1.set_zlabel(r"$J^*(\bar{x})$")
    ax2.scatter(X[:,0], X[:,1], U_opt[:,0])
    ax2.set_xlabel(r"$x_1$")
    ax2.set_ylabel(r"$x_2$")
    ax2.set_zlabel(r"$u_1^*(\bar{x})$")
    ax3.scatter(X[:,0], X[:,1], U_opt[:,1])
    ax3.set_xlabel(r"$x_1$")
    ax3.set_ylabel(r"$x_2$")
    ax3.set_zlabel(r"$u_2^*(\bar{x})$")
    plt.show()

