import sys
sys.path.append('..')

import numpy as np
import casadi as ca
import progressbar
import matplotlib
matplotlib.use("tkAgg")
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
from dp_solver import DPSolver

if __name__ == '__main__':
    
    # number of independent variable steps
    N = 1

    # number of states
    nx = 2

    # number of inputs
    nu = 2

    # number of points in the discretized state and input spaces
    NX = 10
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
    R = 0.1

    def dynamics(x,u):
        return np.vstack((x[0] + 0.8 * x[1] + 0.5 * u[0], x[1] + 0.5 * u[1]))

    def stage_cost(x,u):
        return 0.5 * (Q * (x[0]**2 + x[1]**2) + R * (u[0]**2 + u[1]**2))

    def terminal_cost(x):
        return 0.5 * (Q * (x[0]**2 + x[1]**2))

    def constraints(x,u):
        return np.vstack((u[0], u[1]))

    solver = DPSolver(nx, nu, NX, NU, [stage_cost]*(N), dynamics,\
        x_bounds=x_bounds, u_bounds=u_bounds, constraints=constraints, lbg=np.vstack((U1_MIN, U2_MIN)), ubg=np.vstack((U1_MAX, U2_MAX)))
    
    NDX = solver.X.shape[0]
    # optimal value function
    J_opt = np.zeros((NDX, 1))

    # initialize with terminal cost
    for i in range(NDX):
        J_opt[i] = terminal_cost(np.atleast_2d(solver.X[i,:]).T)

    print('Solving optimal control problem with DP...')
    # loop over time in reversed order
    for i in range(N-1,-1,-1):
        print('stage = {}'.format(i))
        J_opt, U_opt = solver.DPOperator(J_opt,i)

    # compute reference solution using parametric QP
    # optimization variables
    w = []
    # initial guess
    w0 = []
    # constraints
    g = []
    lbg = []
    ubg = []
    lbw = []
    ubw = []
    # objective function
    f = 0.0
    # initial state
    Xk = ca.SX.sym('X0', nx, 1) 
    w+=[Xk]
    w0+=[np.zeros((nx,1))]
    lbw += [-np.inf*np.ones((nx,1))]
    ubw += [np.inf*np.ones((nx,1))]

    g+=[Xk]
    lbg+=[np.zeros((nx,1))]
    ubg+=[np.zeros((nx,1))]
    for i in range(N):
        Uk = ca.SX.sym('U' + str(i), nu, 1) 
        w+=[Uk]
        w0+=[np.zeros((nu,1))]
        lbw += [-np.inf*np.ones((nu,1))]
        ubw += [np.inf*np.ones((nu,1))]

        g_ = constraints(Xk, Uk)
        nh = g_.shape[0]
        g+=[g_]
        lbg+=[np.vstack((U1_MIN, U2_MIN))]
        ubg+=[np.vstack((U2_MAX, U2_MAX))]

        f+=stage_cost(Xk, Uk)
        # dynamics
        x_next = dynamics(Xk, Uk)

        Xk = ca.SX.sym('X' + str(i+1), nx, 1) 
        w+=[Xk]
        w0+=[np.zeros((nx,1))]
        lbw += [-np.inf*np.ones((nx,1))]
        ubw += [np.inf*np.ones((nx,1))]

        g+=[Xk - x_next]
        lbg+=[np.zeros((nx,1))]
        ubg+=[np.zeros((nx,1))]

    f+=terminal_cost(Xk)
    w0 = ca.vertcat(*w0)
    lbw = ca.vertcat(*lbw)
    ubw = ca.vertcat(*ubw)
    lbg = ca.vertcat(*lbg)
    ubg = ca.vertcat(*ubg)
    prob = {'f': f, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g)}
    opts = {'ipopt.print_level':0, 'print_time':0}
    qp_solver = ca.nlpsol('solver', 'ipopt', prob, opts);

    U_opt_qp = np.full_like(U_opt, np.nan)
    J_opt_qp = np.full_like(J_opt, np.nan)


    # plot value function
    X = solver.X

    print('Solving optimal control problem with multiple shooting')
    with progressbar.ProgressBar(max_value=NDX) as bar:
        for i in range(NDX):
            x_ = X[i,:].T
            lbg[0:nx] = x_
            ubg[0:nx] = x_
            sol = qp_solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
            stats = qp_solver.stats()
            if stats['return_status'] != 'Solve_Succeeded':
                raise Warning('OCP solver returned %s ' % (stats['return_status']))
            J_opt_qp[i] = sol['f'].full()
            U_opt_qp[i,:] = sol['x'].full()[nx:nx+nu].T
            bar.update(i)

    # compute normalized infinity norm of error
    dJ = 0.0
    nJ = 1.0
    dU = 0.0
    nU = 1.0
    for i in range(NDX):
        if ~np.any(np.isinf(J_opt[i])): 
            dJ_ = np.abs(J_opt[i] - J_opt_qp[i])
            if dJ_ > dJ:
                dJ = dJ_
            if J_opt_qp[i] > nJ:
                nJ = J_opt_qp[i]
        if ~np.any(np.isnan(U_opt[i,:])): 
            dU_ = np.linalg.norm(U_opt[i] - U_opt_qp[i], np.inf)
            if dU_ > dU:
                dU = dU_
            if np.linalg.norm(U_opt_qp[i], np.inf) > nU:
                nU = np.linalg.norm(U_opt_qp[i], np.inf)

    dJ = dJ/nJ
    dU = dU/nU
    print("inf. norm: dJ = %f, dU = %f" % (dJ, dU))

    fig = plt.figure()
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    ax1.scatter(X[:,0], X[:,1], J_opt)
    ax1.scatter(X[:,0], X[:,1], J_opt_qp, 'r')
    ax1.set_xlabel(r"$x_1$")
    ax1.set_ylabel(r"$x_2$")
    ax1.set_zlabel(r"$J^*(\bar{x})$")
    ax2.scatter(X[:,0], X[:,1], U_opt[:,0])
    ax2.scatter(X[:,0], X[:,1], U_opt_qp[:,0], 'r')
    ax2.set_xlabel(r"$x_1$")
    ax2.set_ylabel(r"$x_2$")
    ax2.set_zlabel(r"$u_1^*(\bar{x})$")
    ax3.scatter(X[:,0], X[:,1], U_opt[:,1])
    ax3.scatter(X[:,0], X[:,1], U_opt_qp[:,1], 'r')
    ax3.set_xlabel(r"$x_1$")
    ax3.set_ylabel(r"$x_2$")
    ax3.set_zlabel(r"$u_2^*(\bar{x})$")
    plt.show()

