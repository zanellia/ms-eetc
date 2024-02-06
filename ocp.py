import numpy as np
import pandas as pd
import casadi as ca

from train import *

from track import computeDiscretizationPoints

from utils import Options, var, postProcessDataFrame, splitLosses


class OptionsCasadiSolver(Options):

    def __init__(self, paramsDict):

        self.numIntervals = 100  # number of shooting intervals with piece-wise constant controls [-]

        self.maxIterations = 1e3  # maximum number of interior-point iterations

        self.energyOptimal = True  # False means time optimal, True energy optimal

        self.minimumVelocity = 1  # minimum value for velocity to avoid numerical issues [m/s]

        self.integrationMethod = 'RK'  # 'RK', 'IRK', 'COLLOCATION', 'CVODES' or 'LINEAR'

        self.integrationOptions = {}  # method-dependent integration options

        self.integrateLosses = False  # integrate losses or take mid-point rule

        super().__init__(paramsDict)


    def overwriteDefaults(self, paramsDict):
        "Extending 'overwriteDefaults' to handle method-dependent integration options."

        super().overwriteDefaults(paramsDict)

        integrationOptions = paramsDict['integrationOptions'] if 'integrationOptions' in paramsDict else {}

        if self.integrationMethod == 'RK':

            self.integrationOptions = OptionsRK(integrationOptions)

        elif self.integrationMethod == 'IRK':

            self.integrationOptions = OptionsIRK(integrationOptions)

        elif self.integrationMethod == 'CVODES':

            self.integrationOptions = OptionsCVODES(integrationOptions)


    def checkValues(self):
        "Checks on 'root' options (nested options treated by submodule)."

        self.checkPositiveInteger(self.numIntervals, 'Number of intervals', allowZero=False)

        self.checkPositiveInteger(self.maxIterations, 'Maximum number of iterations', allowZero=False)

        if not isinstance(self.energyOptimal, bool):

            raise ValueError("'energyOptimal' flag must be a boolean!")

        if type(self.minimumVelocity) not in {int, float} or self.minimumVelocity <= 0:

            raise ValueError("Minimum velocity should be a strictly positive number!")

        if self.integrationMethod not in {'RK', 'IRK', 'CVODES'}:

            raise ValueError("Unknown integration method!")

        if not isinstance(self.integrateLosses, bool):

            raise ValueError("'integrateLosses' flag must be a boolean!")


class casadiSolver():
    "NLP solver object using casadi and ipopt."

    def __init__(self, train, track, optsDict={}):

        # input checking
        track.checkFields()
        train.checkFields()

        opts = OptionsCasadiSolver(optsDict)

        numIntervals = opts.numIntervals
        velocityMin = opts.minimumVelocity

        trainModel = train.exportModel()
        trainIntegrator = TrainIntegrator(trainModel, opts.integrationMethod, opts.integrationOptions.toDict())

        # train parameters

        rho = train.rho
        totalMass = train.mass*rho

        powerLossesTr, powerLossesRgb = train.powerLossesFuns()

        withRgBrake = train.forceMin != 0
        withPnBrake = train.forceMinPn != 0

        accInf = 10  # value for acceleration when bound not defined

        forceMax = train.forceMax/totalMass if train.forceMax is not None else accInf
        forceMin = train.forceMin/totalMass if train.forceMin is not None else -accInf
        forceMinPn = train.forceMinPn/totalMass if train.forceMinPn is not None else -accInf

        powerMax = train.powerMax/totalMass if train.powerMax is not None else None
        powerMin = train.powerMin/totalMass if train.powerMin is not None else None

        accMax = min(accInf, train.accMax if train.accMax is not None else accInf)
        accMin = max(-accInf, -abs(train.accMin) if train.accMin is not None else -accInf)

        velocityMax = train.velocityMax

        if opts.integrateLosses:

            trainIntegrator.initLosses(powerLossesTr, powerLossesRgb, totalMass)

        # track parameters

        self.points = computeDiscretizationPoints(track, numIntervals)
        self.steps = np.diff(self.points.index)

        # real-time parameters

        self.initialTime = ca.MX.sym('t0')
        self.terminalTime = ca.MX.sym('tN')
        self.initialVelocitySquared = ca.MX.sym('v0sq')
        self.terminalVelocitySquared = ca.MX.sym('vNsq')

        # optimization variables

        time = var('time', numIntervals+1)  # time [s]
        velSq = var('velSquared', numIntervals+1)  # speed squared[m**2/s**2]

        Fel = var('Fel', numIntervals)  # electric traction and reg. brake force [N]
        Fpb = var('Fpb', numIntervals) if withPnBrake else [0]*numIntervals  # pneumatic brake [N]

        s = var('slack', numIntervals)  # slack variables for objective

        # objective function

        if not opts.energyOptimal:

            eps = 1e-4

            obj = time[-1] + eps*(ca.dot(Fel, Fel) + ca.dot(Fpb, Fpb))

        else:

            obj = 0  # will add energy of each interval in loop

        # stage-wise construction of optimization problem

        z = []  # vector of optimization variables
        g = []  # vector of inequality constraints

        lbz = []  # lower bound on variables
        ubz = []  # upper bound on variables
        lbg = []  # lower bound of inequality constraints
        ubg = []  # upper bound of inequality constraints

        for i in range(numIntervals+1):

            if i < numIntervals:

                # control variables
                u = [Fel[i]] + ([Fpb[i]] if withPnBrake else [])
                z += u

                # force constraints
                lbz += ([forceMin] if withRgBrake else [0]) + ([forceMinPn] if withPnBrake else [])
                ubz += [forceMax] + ([0] if withPnBrake else [])

                # slack variables
                z += [s[i]]
                lbz += [0]
                ubz += [ca.inf]

                # power constraints
                if powerMax is not None or powerMin is not None:

                    upperBound = powerMax if powerMax is not None else forceMax*velocityMax
                    lowerBound = 0 if not withRgBrake else powerMin if powerMin is not None else forceMin*velocityMax

                    g += [Fel[i]*ca.sqrt(velSq[i]), Fel[i]*ca.sqrt(velSq[i+1])]

                    ubg += [abs(upperBound)]*2
                    lbg += [-abs(lowerBound)]*2

                # gradient and curvature of current index
                grad = self.points.iloc[i]['Gradient [permil]']/1e3
                curv = self.points.iloc[i]['Curvature [1/m]']

                # acceleration constraints
                g += [trainModel.accelerationFun(ca.vertcat(time[i], velSq[i]), ca.vcat(u), grad, curv)]
                lbg += [accMin]
                ubg += [accMax]

                # coupling constraints
                out = trainIntegrator.solve(time=time[i], velocitySquared=velSq[i], ds=self.steps[i],
                    traction=Fel[i], pnBrake=Fpb[i], gradient=grad, curvature=curv)

                xNxt1 = ca.vertcat(time[i+1], velSq[i+1])
                xNxt2 = ca.vertcat(out['time'], out['velSquared'])

                g += [xNxt1 - xNxt2]

                lbg += [0, 0]
                ubg += [0, 0]

                # objective
                if opts.energyOptimal:

                    if not opts.integrateLosses:

                        # approximating interval with mid-point rule
                        vMid = (ca.sqrt(velSq[i]) + ca.sqrt(velSq[i+1]))/2

                        obj += self.steps[i]*(Fel[i] + s[i])

                        g += [s[i] - powerLossesTr(Fel[i], vMid)/vMid]
                        g += [s[i] - powerLossesRgb(Fel[i], vMid)/vMid]

                        lbg += [0]*2
                        ubg += [ca.inf]*2

                    else:

                        energyLossesTr, energyLossesRgb = trainIntegrator.calcLosses(ca.sqrt(velSq[i]), time[i+1]-time[i], Fel[i], Fpb[i], grad, curv)

                        obj += self.steps[i]*Fel[i] + s[i]

                        g += [s[i] - energyLossesTr]
                        g += [s[i] - energyLossesRgb]

                        lbg += [0]*2
                        ubg += [ca.inf]*2

                    if i > 0:  # small penalty on deviations to remove unnecessary spikes

                        obj += 1e-3*(Fel[i] - Fel[i-1])**2

            # state variables
            z += [time[i]]
            z += [velSq[i]]

            if i == 0:

                # initial state constraints
                lbz += [self.initialTime, self.initialVelocitySquared]
                ubz += [self.initialTime, self.initialVelocitySquared]

            elif i == numIntervals:

                # terminal state constraints
                lbz += [self.initialTime, self.terminalVelocitySquared]
                ubz += [self.terminalTime, self.terminalVelocitySquared]

            else:

                # state constraints
                speedLimit = self.points.iloc[i]['Speed limit [m/s]']
                speedLimit = min(speedLimit, velocityMax)

                speedLimit = min(speedLimit, self.points.iloc[i-1]['Speed limit [m/s]'])  # do not accelerate before speed limit increase

                lbz += [self.initialTime, velocityMin**2]
                ubz += [self.terminalTime, speedLimit**2]

        # scaling of objective function (fixes convergence issues when using powerLosses)

        if opts.energyOptimal:

            scalingFactorObjective = 3.6/(1e-6*totalMass)  # convert to kWh

        else:

            scalingFactorObjective = track.length/train.velocityMax  # divide by fastest possible

        obj /= scalingFactorObjective

        # NLP solver

        nlp = {'x': ca.vcat(z), 'f': obj, 'g': ca.vcat(g)}

        self.solver = ca.nlpsol('solver', 'ipopt', nlp, {'ipopt':{'max_iter':opts.maxIterations}})

        # parameters needed in solve step

        self.totalMass = totalMass
        self.velocityMin = velocityMin
        self.numIntervals = numIntervals
        self.withRgBrake = withRgBrake
        self.withPnBrake = withPnBrake
        self.train = train
        self.energyOptimal = opts.energyOptimal
        self.scalingFactorObjective = scalingFactorObjective
        self.opts = opts

        self.lbz = ca.vcat(lbz)
        self.ubz = ca.vcat(ubz)
        self.lbg = ca.vcat(lbg)
        self.ubg = ca.vcat(ubg)


    def solve(self, terminalTime, initialTime=0, terminalVelocity=1, initialVelocity=1):

        # check boundary conditions on time

        if not isinstance(initialTime, (int, float)) or initialTime < 0:

            raise ValueError("Initial time must be a positive number, not {}!".format(initialTime))

        if not isinstance(terminalTime, (int, float)) or terminalTime <= 0:

            raise ValueError("Terminal time must be a strictly positive number, not {}!".format(terminalTime))

        # initial guess
        # NOTE: good idea vel0 to be compatible with f0 (power-wise) to avoid nans at first iteration

        vel0 = (60/3.6)**2
        dt = (terminalTime - initialTime)/self.numIntervals
        t0 = initialTime

        z0 = []

        for _ in range(self.numIntervals):

            z0.append([0.5] + ([-0.1]*self.withPnBrake))
            z0.append([1])  # slacks
            z0.append([t0, vel0])
            t0 += dt

        z0.append([t0, vel0])
        z0 = ca.vcat(z0)

        # substitution of real-time parameters

        v0 = min(max(initialVelocity, self.velocityMin), self.points.iloc[0]['Speed limit [m/s]'])
        vN = min(max(terminalVelocity, self.velocityMin), self.points.iloc[-1]['Speed limit [m/s]'])

        lbz = self.lbz
        lbz = ca.substitute(lbz, self.initialTime, initialTime)
        lbz = ca.substitute(lbz, self.initialVelocitySquared, v0**2)
        lbz = ca.substitute(lbz, self.terminalVelocitySquared, vN**2)

        ubz = self.ubz
        ubz = ca.substitute(ubz, self.initialTime, initialTime)
        ubz = ca.substitute(ubz, self.terminalTime, terminalTime)
        ubz = ca.substitute(ubz, self.initialVelocitySquared, v0**2)
        ubz = ca.substitute(ubz, self.terminalVelocitySquared, vN**2)

        # NLP solution and post-processing of results

        solution = self.solver(lbx=ca.evalf(lbz), ubx=ca.evalf(ubz), lbg=self.lbg, ubg=self.ubg, x0=z0)
        zOpt = solution['x'].full().flatten().tolist()
        cost = (((1e-6/3.6)*self.totalMass if self.energyOptimal else 1)*solution['f']*self.scalingFactorObjective).full().flatten().tolist()[0]  # cost [kWh] or [s]
        stats = {'Solver status': self.solver.stats()['return_status'], 'IP iterations':self.solver.stats()['iter_count'], 'CPU time [s]':self.solver.stats()['t_wall_total'], 'Cost':cost}

        if not self.solver.stats()['success']:

            if stats['Solver status'] != 'USER_INTERRUPT':

                print("Solver failed with status '{}'".format(stats['Solver status']))

            df = None

        else:

            print("Solver converged in {:4d} iterations.".format(stats['IP iterations']))

            # terminal stage (only states)
            tOpt = [zOpt[-2]]
            velOpt = [ca.sqrt(zOpt[-1])]

            # remaining stages (states, controls and slacks)
            zOpt = zOpt[:-2]

            offset = 0
            stp = 4 + self.withPnBrake

            FelOpt = zOpt[offset::stp] + [np.nan]
            offset += 1

            if self.withPnBrake:

                FpbOpt = zOpt[offset::stp] + [np.nan]
                offset += 1

            sOpt = zOpt[offset::stp] + [np.nan]
            offset += 1

            tOpt = zOpt[offset::stp] + tOpt
            offset += 1
            velOpt = [ca.sqrt(b) for b in zOpt[offset::stp]] + velOpt

            df = pd.DataFrame({'Time [s]':tOpt, 'Position [m]':self.points.index.values}).set_index('Time [s]')
            df['Velocity [m/s]'] = velOpt
            df['Force (el) [N]'] = np.array(FelOpt)*self.totalMass
            df['Force (pnb) [N]'] = np.array(FpbOpt)*self.totalMass if self.withPnBrake else np.array([0]*(self.numIntervals+1))
            df['Slacks'] = np.array(sOpt)*self.totalMass

            df = postProcessDataFrame(df, self.points, self.train)

        return df, stats

    def computeDPSolution(self, NU=10, NX=10):
        """ Computes reference solution using Dynamic Programming.

        Parameters
        ----------

        NU  :   int
                number of points in the input space (per dimension)
        NX  :   int
                number of points in the state space (per dimension)
        """

        # hard-coded state and input dimension (code should work for arbitrary nx and nu
        # though)

        def project_homogeneous_grid(value:np.ndarray, grid:list):
            '''project a value to the index of its "closest" 
            value in a homogeneous grid

            Parameters
            ----------
            value : np.ndarray
                    column vector to be projected

            grid  : list
                    list of np.ndarrays with grid values

            Returns
            -------

            list
                indexes of "closest" element in the grid,
                None if the vector is outside of the
                extrema of the grid
            '''

            M = value.shape[0]
            indexes = [None] * M
            for i in range(M):
                N = grid[i].size
                min_val = np.amin(grid[i])
                max_val = np.amax(grid[i])
                # we assume the grid is a monotonous
                # increasing 1D array
                indexes[i] = np.around(((value-min_val)/\
                    (max_val-min_val))*(N-1)).astype(int)
                if indexes[i] < 0 or indexes[i] > N:
                    indexes[i] = None
                # else
            return indexes

        def DPOperator(J, l, x_values, u_values,\
                integrator, constraints):
            '''
            Compute updated value function using DP operator, i.e, for all x, J_k(x) = min_u l(x,u) + J_{k+1}(f(x,u))

            Parameters
            ----------

            J : np.ndarray (NDX X nx)
                value function in tabular form

            l : function
                stage cost function x,u -> l(x,u)
                

            x_values : list of np.ndarrays
                discretized state space
            
            u_values : list of np.ndarrays
                discretized input space

            integrator : function
                integrator function x,u -> x_+

            constraints : function
                constraints function x,u -> g(x,u)
            '''

            X = np.array(np.meshgrid(*x_values)).T.reshape(-1,nx)
            U = np.array(np.meshgrid(*u_values)).T.reshape(-1,nu)

            # number of discretized states
            NDX = X.shape[0]

            # number of discretized inputs
            NDU = U.shape[0]

            # loop over states
            for j in range(NDX):
                x_ = X[j,:].T

                # gradient and curvature of current index
                grad = self.points.iloc[i]['Gradient [permil]']/1e3
                curv = self.points.iloc[i]['Curvature [1/m]']

                # loop over inputs
                for k in range(NDU):
                    u_ = U[k,:].T

                    # integrate dynamics
                    out = integrator.solve(time=time[i], velocitySquared=velSq[i], ds=self.steps[i],
                        traction=Fel[i], pnBrake=Fpb[i], gradient=grad, curvature=curv)

                    xNxt1 = ca.vertcat(time[i+1], velSq[i+1])
                    xNxt2 = ca.vertcat(out['time'], out['velSquared'])
            return J_new
        
        nx = 2
        nu = 2
        
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
        U = np.array(np.meshgrid(*u_values)).T.reshape(-1,nu)
        # create integrator (use options fed to MS solver)
        opts = self.opts

        numIntervals = opts.numIntervals
        velocityMin = opts.minimumVelocity

        trainModel = train.exportModel()
        trainIntegrator = TrainIntegrator(trainModel, opts.integrationMethod, opts.integrationOptions.toDict())

        # DP recursion
        numIntervals = opts.numIntervals
        
        # optimal value function
        J_opt = np.full_like(X)
        J_opt_new = np.full_like(X)

        # loop over time
        for i in range(numIntervals):
            


if __name__ == '__main__':

    from train import Train
    from track import Track

    # Example on how to solve an OCP

    train = Train(config={'id':'NL_intercity_VIRM6', 'max deceleration':None, 'max acceleration':{'unit':'m/s^2', 'value':0.45}})

    track = Track(config={'id':'00_var_speed_limit_100'})

    opts = {'numIntervals':200, 'integrationMethod':'RK', 'integrationOptions':{'numApproxSteps':1}, 'energyOptimal':True}

    solver = casadiSolver(train, track, opts)

    df, stats = solver.solve(1541)

    # print some info
    if df is not None:

        print("")
        print("Objective value = {:.2f} {}".format(stats['Cost'], 'kWh' if solver.opts.energyOptimal else 's'))
        print("")
        print("Maximum acceleration: {:5.2f}, with bound {}".format(df.max()['Acceleration [m/s^2]'], train.accMax if train.accMax is not None else 'None'))
        print("Maximum deceleration: {:5.2f}, with bound {}".format(df.min()['Acceleration [m/s^2]'], train.accMin if train.accMin is not None else 'None'))

    else:

        print("Solver failed!")
