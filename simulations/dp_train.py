import sys
sys.path.append('..')

import numpy as np
import casadi as ca
import progressbar
import matplotlib
matplotlib.use("tkAgg")
import matplotlib.pyplot as plt
from matplotlib import cm
plt.rcParams['text.usetex'] = True
from dp_solver import DPSolver
from train import *
from ocp import *
from track import *

SAVE2JSON = True
JSON_NAME = 'temp'

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
    N = 90

    # number of states
    nx = 2

    # number of inputs
    nu = 2

    # number of points in the discretized state and input spaces
    NX = 10
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
    trainIntegrator = TrainIntegrator(trainModel, opts.integrationMethod,\
        opts.integrationOptions.toDict())

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
        (0.0, (velocityMax)**2))
    u_bounds = ((forceMin_, forceMax), (forceMinPn_, 0.0))

    x_values = []
    x_values.append(np.linspace(x_bounds[0][0], x_bounds[0][1], NX))
    # grid in velocity space rather than squared velocity
    x_values.append((np.linspace(np.sqrt(x_bounds[1][0]), np.sqrt(x_bounds[1][1]), NX))**2)
    # # plot loss function
    # npoints = 100
    # fel = np.linspace(forceMin, forceMax, npoints)
    # v2 = np.linspace(velocityMin**2, velocityMax**2, npoints)

    # FEL, V2 = np.meshgrid(fel, v2)
    # Z = powerLossesND(FEL, V2)
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # surf = ax.plot_surface(FEL, V2, Z, cmap=cm.coolwarm,
    #                    linewidth=0, antialiased=False)
    # plt.show()

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
    params = []

    # dynamics
    def dyn_(x,u,p):
        x_next = trainIntegrator.solve(time=x[0], velocitySquared=x[1],\
        ds=steps[i], traction=u[0], pnBrake=u[1], gradient=p[0],\
        curvature=p[1])
        return np.array([x_next['time'].full(), x_next['velSquared'].full()]) 

    dynamics = dyn_

    # acceleration constraints
    if withPnBrake:
        def g_(x, u, x_next, p): 
            return ca.vertcat(g_p(x,u,x_next),\
            trainModel.accelerationFun(ca.vertcat(x[0], x[1]),\
            u, p[0], p[1]))
    else:
        def g_(x, u, x_next, p): 
            return ca.vertcat(g_p(x,u,x_next),\
            trainModel.accelerationFun(ca.vertcat(x[0], x[1]),\
            u[0], p[0], p[1]))

    constraints = g_

    if opts.energyOptimal:

        if not opts.integrateLosses:

            # approximating interval with mid-point rule
            def stage_cost_(x, u, x_next, p):
                vMid = (ca.sqrt(x[1]) + ca.sqrt(x_next[1]))/2
                return (steps[i]*(u[0] + powerLossesND(u[0], vMid)/vMid))/\
                    scalingFactorObjective

            # stage_cost.append(stage_cost_)
            stage_cost = stage_cost_

        else:

            raise Exception('Not implemented.')

        # TODO(andrea): this would require a state augmentation...

        # if i > 0:  # small penalty on deviations to remove unnecessary spikes
        #     obj += 1e-3*(Fel[i] - Fel[i-1])**2

        Warning('Input rate penalty not implemented!')


    for i in range(numIntervals):

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

        params.append([grad,curv])
        
        lbg_ = np.vstack((lbg_,accMin))
        ubg_ = np.vstack((ubg_,accMax))

        lbg.append(lbg_)
        ubg.append(ubg_)


        # objective
        # scaling of objective function (fixes convergence issues when using powerLosses)
        if opts.energyOptimal:

            scalingFactorObjective = 3.6/(1e-6*totalMass)  # convert to kWh

        else:

            scalingFactorObjective = track.length/train.velocityMax  # divide by fastest possible

        # state constraints
        speedLimit = points.iloc[i]['Speed limit [m/s]']
        speedLimit = min(speedLimit, velocityMax)

        speedLimit = min(speedLimit, points.iloc[i-1]['Speed limit [m/s]'])  # do not accelerate before speed limit increase

        lbx.append(np.vstack((initialTime, velocityMin**2)))
        ubx.append(np.vstack((terminalTime, speedLimit**2)))

    # terminal state constraints
    lbx.append(np.atleast_2d(np.array([initialTime, terminalVelocitySquared - velSlack])).T)
    ubx.append(np.atleast_2d(np.array([terminalTime, terminalVelocitySquared + 10])).T)
    # ubx.append(np.atleast_2d(np.array([terminalTime, np.inf])).T)

    # create DP solver
    solver = DPSolver(nx, nu, NX, NU, stage_cost, dynamics,\
        x_values=x_values, u_bounds=u_bounds, constraints=constraints, lbg=lbg, ubg=ubg,\
        lbx=lbx, ubx=ubx, lbu=lbu, ubu=ubu, params=params)
    
    NDX = solver.X.shape[0]
    # optimal value function
    J_opt_ = np.zeros((NDX, 1))

    # initialize with either 0 or inf (feasible or not)
    for i in range(NDX):
        x_ = np.atleast_2d(solver.X[i,:]).T
        # simple state bound satisfaction
        if np.any(x_ > ubx[numIntervals]) or np.any(x_ < lbx[numIntervals]):
            J_opt_[i] = np.inf
            # J_opt[i] = 0.0
        else:
            J_opt_[i] = 0.0

    print('Solving optimal control problem with DP...')
    J_opt = np.nan * np.ones((NDX,1,N + 1))
    U_opt = np.nan * np.ones((NDX,nu,N))
    
    J_opt[:,:,-1] = J_opt_

    fig = plt.figure()
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel(r"$v^2$")
    ax1.set_zlabel(r"$J^*(\bar{x})$")
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"$v^2$")
    ax2.set_zlabel(r"$F_{\rm{el}}^*(\bar{x})$")
    ax3.set_xlabel(r"$t$")
    ax3.set_ylabel(r"$v^2$")
    ax3.set_zlabel(r"$F_{\rm{pb}}^*(\bar{x})$")

    X = solver.X
    plt.show(block=False)
    # loop over time in reversed order
    for i in range(N-1,-1,-1):
        print('stage = {}'.format(i))
        J_opt[:,:,i], U_opt[:,:,i] = solver.DPOperator(J_opt[:,:,i+1],i)

        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax1.scatter(X[:,0], X[:,1], J_opt[:,:,i])
        ax2.scatter(X[:,0], X[:,1], U_opt[:,0,i])
        ax3.scatter(X[:,0], X[:,1], U_opt[:,1,i])
        fig.canvas.draw()
        fig.canvas.flush_events()
        # plt.show()

    if SAVE2JSON:
        np.save(JSON_NAME + '_X.npy', X)
        np.save(JSON_NAME + '_J_opt.npy', J_opt)
        np.save(JSON_NAME + '_U_opt.npy', U_opt)


