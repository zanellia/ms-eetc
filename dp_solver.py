import numpy as np
import progressbar
import multiprocessing
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
import threading
import time

def DP_loop_fun(args): 
    j, X, U, ubx, lbx, ubu, lbu, lbg, ubg, dynamics, constraints, stage_cost, J, NDX, NDU, NX, NU, nx, nu, x_values, u_values, params = args
    x_ = np.atleast_2d(X[j,:]).T

    # my_pid = os.getpid()
    # print("Executing our Task on Process {}".format(my_pid))
    # print("Executing on Thread {}".format(threading.get_ident()))
    J_new = np.inf
    U_opt = np.nan * np.zeros((1, nu))
    # print(j)

    for i in range(100):
        # simple state bound satisfaction
        if ubx is not None:
            if np.any(x_ > ubx):
                return j, J_new, U_opt

        if lbx is not None:
            if np.any(x_ < lbx):
                return j, J_new, U_opt

        # loop over inputs
        for k in range(NDU):
            u_ = np.atleast_2d(U[k,:]).T

            # simple input bound satisfaction
            if ubu is not None:
                if np.any(u_ > ubu):
                    continue

            if lbu is not None:
                if np.any(u_ < lbu):
                    continue

            # integrate dynamics
            x_next = dynamics(x_,u_, params)

            # project onto state grid
            idx_next, x_next_p = project_onto_homogeneous_grid(x_next, x_values)

            # constraint satisfaction
            con_ = constraints(x_,u_,x_next, params)
            if constraints is not None:
                if np.any(con_ > ubg):
                    continue

                if np.any(con_ < lbg):
                    continue

            if idx_next is None:
                continue

            # obtain index in reshaped form
            idx_next_rs = np.unravel_index(np.ravel_multi_index(idx_next, [NX]*nx), (NDX))

            # evaluate argument of minimization
            J_ = stage_cost(x_, u_, x_next, params) + J[idx_next_rs]

            # print("u = [%f, %f], x = [%f, %f], x_+ = [%f, %f], x_+_p = [%f, %f], J = %f, J_opt = % f"\
            #     % (u_[0], u_[1], x_[0], x_[1], np.squeeze(x_next[0]), np.squeeze(x_next[1]),\
            #     np.squeeze(x_next_p[0]), np.squeeze(x_next_p[1]), J_, J_new[j]))

            if J_ < J_new:
                J_new = J_
                U_opt = u_.T

    return j, J_new, U_opt

def project_onto_homogeneous_grid(value, grid):
    '''
    Project a value to the index of its "closest" 
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

    np.ndarray 
        projected vector
    '''

    M = value.shape[0]
    indexes = [np.nan] * M
    for i in range(M):
        N = grid[i].size
        min_val = np.amin(grid[i])
        max_val = np.amax(grid[i])
        # we assume the grid is a monotonous
        # increasing 1D array
        indexes[i] = np.around(((value[i]-min_val)/\
            (max_val-min_val))*(N-1)).astype(int)
        # if not in the interval, return None
        if indexes[i] < 0 or indexes[i] > N-1:
            return None, None

    value_p = np.full_like(value, 0.0)

    for i in range(M):
        value_p[i] = grid[i][indexes[i]]

    return indexes, value_p

class DPSolver():
    def __init__(self, nx, nu, NX, NU, stage_cost, dynamics, terminal_cost=None,\
            constraints=None, lbg=None, ubg=None, lbx=None, ubx=None, lbu=None, ubu=None,\
            x_bounds=None, u_bounds=None, x_values=None, u_values=None, params=None):
        
        '''
        Parameters
        ----------
        nx  :   int
            number of states

        nu  :   int
            number of inputs 

        NX  :   int
            number of discretization points for the state space (per dimension)

        NU  :   int
            number of discretization points for the input space (per dimension)

        stage_cost : function or list of functions
            stage cost function R^{nx} x R^{nu} -> R - list of functions for time-varying formulations
            
        dynamics : function
            dynamics function R^{nx} x R^{nu} -> R^{nx} - list of functions for time-varying formulations

        terminal_cost : function or list of functions
            terminal_cost function R^{nx} -> R

        constraints : function or list of functions
            constraints function R^{nx} x R^{nu} -> R^{ng} - list of functions for time-varying formulations

        lbg : np.ndarray or list of np.ndarrays
            lower bounds on constraints

        ubg : np.ndarray or list of np.ndarrays
            upper bounds on constraints

        lbx : np.ndarray or list of np.ndarrays
            lower bounds on states

        ubx : np.ndarray or list of np.ndarrays
            upper bounds on states

        lbu : np.ndarray or list of np.ndarrays
            lower bounds on inputs

        ubu : np.ndarray or list of np.ndarrays
            upper bounds on inputs 

        x_bounds : tuple
            state bounds to be used for homogeneous discretization
        
        u_bounds : tuple
            state bounds to be used for homogeneous discretization

        x_values : list of nx np.ndarrays (1 X NX)
            discretized state space
        
        u_values : list of nu np.ndarrays (1 X NU)
            discretized input space


        Returns
        -------

        np.ndarray
            updated value function according to DP operator
        '''
        
        # TODO(andrea): add some more checks here

        self.nx = nx
        self.nu = nu
        self.NX = NX
        self.NU = NU
        self.x_bounds = x_bounds
        self.u_bounds = u_bounds
        self.stage_cost = stage_cost
        self.terminal_cost = terminal_cost
        self.dynamics = dynamics
        self.constraints = constraints
        self.params = params
        self.lbg = lbg
        self.ubg = ubg
        self.lbx = lbx
        self.ubx = ubx
        self.lbu = lbu
        self.ubu = ubu

        # grid state and input space
        X = np.zeros(tuple([NX] * nx))
        U = np.zeros(tuple([NU] * nu))

        if x_values is None:
            if x_bounds is None:
                raise Exception('Either x_bounds or x_values need to be provided.') 
            x_values = []
            for i in range(nx):
                x_values.append(np.linspace(x_bounds[i][0], x_bounds[i][1], NX))
            self.x_values = x_values
        else:
            self.x_values = x_values

        if u_values is None:
            if u_bounds is None:
                raise Exception('Either u_bounds or u_values need to be provided.') 
            u_values = []
            for i in range(nu):
                u_values.append(np.linspace(u_bounds[i][0], u_bounds[i][1], NU))
            self.u_values = u_values
        else:
            self.u_values = u_values

        X = np.array(np.meshgrid(*x_values)).T.reshape(-1,nx)
        NDX = X.shape[0]
        U = np.array(np.meshgrid(*u_values)).T.reshape(-1,nu)

        # number of discretized states
        self.NDX = X.shape[0]

        # number of discretized inputs
        self.NDU = U.shape[0]

        self.X = X
        self.U = U
            
        return

    def DPOperator(self, J, stage_idx = None):
        '''
        Compute updated value function using DP operator, i.e, for all x, J_k(x) = min_u l(x,u) + J_{k+1}(f(x,u))

        Parameters
        ----------

        J : np.ndarray (NDX)
            value function in tabular form

        stage_idx : int
            stage index (for time-varying formulations)

        Returns
        -------

        np.ndarray
            updated value function according to DP operator
        '''

        nx = self.nx
        nu = self.nu

        NX = self.NX
        NU = self.NU

        X = self.X
        U = self.U

        NDX = self.NDX
        NDU = self.NDU

        x_values = self.x_values
        u_values = self.u_values

        J_new = np.inf * np.ones((NDX, 1))
        U_opt = np.nan * np.zeros((NDX, nu))

        # if stage_idx is not None:
        #     # check that any of dynamics, stage_cost, constraints is stage-varying
        #     if (not isinstance(self.dynamics, list)) and (not isinstance(self.constraints, list)) and (not isinstance(self.stage_cost, list)):
        #         raise Exception('stage_idx provided, but problem formulation is not stage-varying')

        # if isinstance(self.dynamics, list):
        #     if stage_idx is None:
        #         raise Exception('Dynamics are stage-varying, but no stage index was provided.')
        #     dynamics = self.dynamics[stage_idx]
        # else:
        #     dynamics = self.dynamics

        dynamics = self.dynamics

        # if isinstance(self.stage_cost, list):
        #     if stage_idx is None:
        #         raise Exception('Stage cost is stage-varying, but no stage index was provided.')
        #     stage_cost = self.stage_cost[stage_idx]
        # else:
        #     stage_cost = self.stage_cost
        stage_cost = self.stage_cost

        # if isinstance(self.constraints, list):
        #     if stage_idx is None:
        #         raise Exception('Constraints are stage-varying, but no stage index was provided.')
        #     constraints = self.constraints[stage_idx]
        # else:
        #     constraints = self.constraints

        constraints = self.constraints

        if isinstance(self.lbg, list):
            if stage_idx is None:
                raise Exception('Bounds are stage-varying, but no stage index was provided.')
            lbg = self.lbg[stage_idx]
        else:
            lbg = self.lbg

        if isinstance(self.ubg, list):
            if stage_idx is None:
                raise Exception('Bounds are stage-varying, but no stage index was provided.')
            ubg = self.ubg[stage_idx]
        else:
            ubg = self.ubg

        if isinstance(self.lbx, list):
            if stage_idx is None:
                raise Exception('Bounds are stage-varying, but no stage index was provided.')
            lbx = self.lbx[stage_idx]
        else:
            lbx = self.lbx

        if isinstance(self.ubx, list):
            if stage_idx is None:
                raise Exception('Bounds are stage-varying, but no stage index was provided.')
            ubx = self.ubx[stage_idx]
        else:
            ubx = self.ubx

        if isinstance(self.lbu, list):
            if stage_idx is None:
                raise Exception('Bounds are stage-varying, but no stage index was provided.')
            lbu = self.lbu[stage_idx]
        else:
            lbu = self.lbu

        if isinstance(self.ubu, list):
            if stage_idx is None:
                raise Exception('Bounds are stage-varying, but no stage index was provided.')
            ubu = self.ubu[stage_idx]
        else:
            ubu = self.ubu

        if isinstance(self.params, list):
            if stage_idx is None:
                raise Exception('params are stage-varying, but no stage index was provided.')
            params = self.params[stage_idx]
        else:
            params = self.params

        # loop over states

        # prepare args
        args = []
        for j in range(NDX):
            args.append((j, X, U, ubx, lbx, ubu, lbu, lbg, ubg, dynamics, constraints, stage_cost, J, NDX, NDU, NX, NU, nx, nu, x_values, u_values, params))

        # with progressbar.ProgressBar(max_value=NDX) as bar:
        #     for j in range(NDX):
        #         _ , J_new[j], U_opt[j,:] = DP_loop_fun(args[j]) 
        #         bar.update(j)

        # executor = ProcessPoolExecutor(max_workers=10)
        # for result in executor.map(DP_loop_fun, args):
        #     # print(result[0])
        #     J_new[result[0]] = result[1]
        #     U_opt[result[0],:] = result[2]


        start_time = time.time()
        with multiprocessing.Pool(processes=20) as pool:
            for result in pool.map(DP_loop_fun, args):
                J_new[result[0]] = result[1]
                U_opt[result[0],:] = result[2]
        exec_time = time.time() - start_time
        print('DP iteration executed in %f s' % (exec_time))

        return J_new, U_opt
    
