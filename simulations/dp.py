import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from dp_solver import DPSolver

if __name__ == '__main__':

    nx = 2
    nu = 2

    NX = 10
    NU = 10
    
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

    # grid state and input space
    X = np.zeros(tuple([NX] * nx))
    U = np.zeros(tuple([NU] * nu))

    x_values = []
    for i in range(nx):
        x_values.append(np.linspace(x_bounds[i][0], x_bounds[i][1], NX))

    u_values = []
    for i in range(nu):
        u_values.append(np.linspace(u_bounds[i][0], u_bounds[i][1], NU))

    X = np.array(np.meshgrid(*x_values)).T.reshape(-1,nx)
    NDX = X.shape[0]
    U = np.array(np.meshgrid(*u_values)).T.reshape(-1,nu)

    def dynamics(x,u):
        return 0.1 * x + u
    def stage_cost(x,u):
        return 0.5 * (x.T@x + u.T@u)

    solver = DPSolver(nx, nu, NX, NU, x_bounds, u_bounds, stage_cost, dynamics)

    # # create integrator (use options fed to MS solver)
    # opts = self.opts

    # numIntervals = opts.numIntervals
    # velocityMin = opts.minimumVelocity

    # trainModel = train.exportModel()
    # trainIntegrator = TrainIntegrator(trainModel, opts.integrationMethod, opts.integrationOptions.toDict())

    # # gradient and curvature of current index
    # grad = self.points.iloc[i]['Gradient [permil]']/1e3
    # curv = self.points.iloc[i]['Curvature [1/m]']

    # out = integrator.solve(time=time[i], velocitySquared=velSq[i], ds=self.steps[i],
    #     traction=Fel[i], pnBrake=Fpb[i], gradient=grad, curvature=curv)

    # xNxt1 = ca.vertcat(time[i+1], velSq[i+1])
    # xNxt2 = ca.vertcat(out['time'], out['velSquared'])

    # # DP recursion
    # numIntervals = opts.numIntervals
    
    # optimal value function
    J_opt = np.zeros((NDX, 1))

    def f(x,u):
        return 0.1 * x + u

    def l(x,u):
        return x.T@x + u.T@u

    J_new = solver.DPOperator(J_opt, l, x_values, u_values, f)
    import pdb; pdb.set_trace()

    # # loop over time
    # for i in range(numIntervals):
        

