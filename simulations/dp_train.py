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
from train import *
from ocp import *
from track import *

if __name__ == '__main__':
    
    initialTime = 0.0
    minimumTime = 272.4726
    terminalTime = minimumTime*1.2
    initialVelocitySquared = 1.0
    terminalVelocitySquared = 1.0
    # allow for a "small" violation of the terminal constraints
    velSlack = 2.0
    # TODO(andrea): solve instead an OCP with N = 1 with a terminal constraint?

    # number of independent variable steps
    N = 1

    # number of states
    nx = 2

    # number of inputs
    nu = 2

    # number of points in the discretized state and input spaces
    NX = 50
    NU = 10

    train = Train(config={'id':'NL_intercity_VIRM6'}, pathJSON='../trains')
    train.forceMinPn = 0

    track = Track(config={'id':'00_var_speed_limit_100'}, pathJSON='../tracks')
    track.updateLimits(positionEnd=8500)


    # input checking
    track.checkFields()
    train.checkFields()
    optsDict = {}

    opts = OptionsCasadiSolver(optsDict)

    numIntervals = opts.numIntervals
    velocityMin = opts.minimumVelocity

    trainModel = train.exportModel()
    trainIntegrator = TrainIntegrator(trainModel, opts.integrationMethod, opts.integrationOptions.toDict())

    # train parameters

    rho = train.rho
    totalMass = train.mass*rho

    powerLossesTr, powerLossesRgb = train.powerLossesFuns()

    # non-differentiable power losses for DP cost
    powerLossesND = train.powerLossesFuns(split=False)

    withRgBrake = train.forceMin != 0
    withPnBrake = train.forceMinPn != 0

    accInf = 10  # value for acceleration when bound not defined

    forceMax = train.forceMax/totalMass if train.forceMax is not None else accInf
    forceMin = train.forceMin/totalMass if train.forceMin is not None else -accInf
    forceMinPn = train.forceMinPn/totalMass if train.forceMinPn is not None else -accInf

    # force constraints
    forceMin_ = 0
    if withRgBrake:
        forceMin_ = forceMin

    forceMinPn_ = 0
    if withRgBrake:
        forceMinPn_ = forceMinPn

    powerMax = train.powerMax/totalMass if train.powerMax is not None else None
    powerMin = train.powerMin/totalMass if train.powerMin is not None else None

    accMax = min(accInf, train.accMax if train.accMax is not None else accInf)
    accMin = max(-accInf, -abs(train.accMin) if train.accMin is not None else -accInf)

    velocityMax = train.velocityMax

    x_bounds = ((initialTime, terminalTime),\
        (0.0, velocityMax**2))
    u_bounds = ((forceMin_, forceMax), (forceMinPn_, 0.0))


    if opts.integrateLosses:

        trainIntegrator.initLosses(powerLossesTr, powerLossesRgb, totalMass)

    # track parameters

    points = computeDiscretizationPoints(track, numIntervals)
    steps = np.diff(points.index)

    # objective function

    if not opts.energyOptimal:

        eps = 1e-4

        obj = time[-1] + eps*(ca.dot(Fel, Fel) + ca.dot(Fpb, Fpb))

    else:

        obj = 0  # will add energy of each interval in loop

    # stage-wise construction of optimization problem

    lbg = []  # lower bound of inequality constraints
    ubg = []  # upper bound of inequality constraints
    lbu = []  # lower input bound 
    ubu = []  # upper input bound 
    lbx = []  # lower state bound 
    ubx = []  # upper state bound 
    dynamics = []
    stage_cost = [] 
    constraints = []

    for i in range(numIntervals):

        if i < numIntervals:

            # input bounds
            lbu.append(np.vstack((forceMin_, forceMinPn_)))
            ubu.append(np.vstack((forceMax, 0.0)))

            # power constraints
            if powerMax is not None or powerMin is not None:

                upperBound = powerMax if powerMax is not None else forceMax*velocityMax
                lowerBound = 0 if not withRgBrake else powerMin if powerMin is not None else forceMin*velocityMax

                def g_p(x, u, xnext): 
                    return np.vstack((u[0]*ca.sqrt(x[1]), u[0]*ca.sqrt(xnext[1])))

                ubg_ = abs(upperBound)*np.ones((2,1))
                lbg_ = -abs(lowerBound)*np.ones((2,1))
            else:
                ubg_ = np.zeros((0,1))
                lbg_ = np.zeros((0,1))

                def g_p(x, u, xnext): 
                    return [] 

            # gradient and curvature of current index
            grad = points.iloc[i]['Gradient [permil]']/1e3
            curv = points.iloc[i]['Curvature [1/m]']

            # TODO(andrea): handle variable number of inputs in a cleaner way
            # acceleration constraints
            if withPnBrake:
                def g_(x, u, x_next): 
                    return ca.vertcat(g_p(x,u,x_next),\
                    trainModel.accelerationFun(ca.vertcat(x[0], x[1]),\
                    u, grad, curv))
            else:
                def g_(x, u, x_next): 
                    return ca.vertcat(g_p(x,u,x_next),\
                    trainModel.accelerationFun(ca.vertcat(x[0], x[1]),\
                    u[0], grad, curv))
            
            lbg_ = np.vstack((lbg_,accMin))
            ubg_ = np.vstack((ubg_,accMax))

            lbg.append(lbg_)
            ubg.append(ubg_)
            constraints.append(g_)

            # coupling constraints
            def dyn_(x,u):
                x_next = trainIntegrator.solve(time=x[0], velocitySquared=x[1],\
                ds=steps[i], traction=u[0], pnBrake=u[1], gradient=grad,\
                curvature=curv)
                return np.array([x_next['time'].full(), x_next['velSquared'].full()]) 

            dynamics.append(dyn_)

            # objective
            # scaling of objective function (fixes convergence issues when using powerLosses)
            if opts.energyOptimal:

                scalingFactorObjective = 3.6/(1e-6*totalMass)  # convert to kWh

            else:

                scalingFactorObjective = track.length/train.velocityMax  # divide by fastest possible
            if opts.energyOptimal:

                if not opts.integrateLosses:

                    # approximating interval with mid-point rule
                    def stage_cost_(x, u, x_next):
                        vMid = (ca.sqrt(x[1]) + ca.sqrt(x_next[1]))/2
                        return (steps[i]*(u[0] + powerLossesTr(u[0], vMid)/vMid))/\
                            scalingFactorObjective

                    stage_cost.append(stage_cost_)

                else:

                    raise Exception('Not implemented.')

                # TODO(andrea): this would require a state augmentation...

                # if i > 0:  # small penalty on deviations to remove unnecessary spikes
                #     obj += 1e-3*(Fel[i] - Fel[i-1])**2

                Warning('Input rate penalty not implemented!')


        # state constraints
        speedLimit = points.iloc[i]['Speed limit [m/s]']
        speedLimit = min(speedLimit, velocityMax)

        speedLimit = min(speedLimit, points.iloc[i-1]['Speed limit [m/s]'])  # do not accelerate before speed limit increase

        lbx.append(np.vstack((initialTime, velocityMin**2)))
        ubx.append(np.vstack((terminalTime, speedLimit**2)))

    # terminal state constraints
    lbx.append(np.atleast_2d(np.array([initialTime, terminalVelocitySquared - velSlack])).T)
    ubx.append(np.atleast_2d(np.array([terminalTime, terminalVelocitySquared + velSlack])).T)

    # create DP solver
    solver = DPSolver(nx, nu, NX, NU, stage_cost, dynamics,\
        x_bounds=x_bounds, u_bounds=u_bounds, constraints=constraints, lbg=lbg, ubg=ubg,\
        lbx=lbx, ubx=ubx, lbu=lbu, ubu=ubu)
    
    NDX = solver.X.shape[0]
    # optimal value function
    J_opt = np.zeros((NDX, 1))

    # initialize with either 0 or inf (feasible or not)
    for i in range(NDX):
        x_ = np.atleast_2d(solver.X[i,:]).T
        # simple state bound satisfaction
        if np.any(x_ > ubx[numIntervals]) or np.any(x_ < lbx[numIntervals]):
            J_opt[i] = np.inf
        else:
            J_opt[i] = 0.0

    print('Solving optimal control problem with DP...')
    # loop over time in reversed order
    for i in range(N-1,-1,-1):
        print('stage = {}'.format(i))
        J_opt, U_opt = solver.DPOperator(J_opt,i)

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

