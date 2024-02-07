import numpy as np

def project_onto_homogeneous_grid(value:np.ndarray, grid:list):
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
    def __init__(self, nx, nu, NX, NU, x_bounds, u_bounds, stage_cost,\
            dynamics, terminal_cost=None, constraints=None):
        
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
        
        return

    def DPOperator(self, J, l, x_values, u_values,\
            integrator, constraints=None):
        '''
        Compute updated value function using DP operator, i.e, for all x, J_k(x) = min_u l(x,u) + J_{k+1}(f(x,u))

        Parameters
        ----------

        J : np.ndarray (NDX)
            value function in tabular form

        l : function
            stage cost function x,u -> l(x,u)
            

        x_values : list of nx np.ndarrays (1 X NX)
            discretized state space
        
        u_values : list of nu np.ndarrays (1 X NU)
            discretized input space

        integrator : function
            integrator function x,u -> x_+

        constraints : function
            constraints function x,u -> g(x,u)

        Returns
        -------

        np.ndarray
            updated value function according to DP operator
        '''

        nx = len(x_values)
        nu = len(u_values)

        NX = self.NX
        NU = self.NU

        X = np.array(np.meshgrid(*x_values)).T.reshape(-1,nx)
        U = np.array(np.meshgrid(*u_values)).T.reshape(-1,nu)

        # number of discretized states
        NDX = X.shape[0]

        # number of discretized inputs
        NDU = U.shape[0]

        J_new = np.inf*np.ones((NDX, 1))

        # loop over states
        for j in range(NDX):
            x_ = X[j,:].T

            # loop over inputs
            for k in range(NDU):
                u_ = U[k,:].T

                # integrate dynamics
                x_next = integrator(x_,u_)

                # project onto state grid
                idx_next, x_next_p = project_onto_homogeneous_grid(x_next, x_values)

                if idx_next is None:
                    continue

                # obtain index in reshaped form
                idx_next_rs = np.unravel_index(np.ravel_multi_index(idx_next, [NX]*nx), (NDX))

                # evaluate argument of minimization
                J_ = l(x_, u_) + J[idx_next_rs]

                if J_ < J_new[j]:
                    J_new[j] = J_

        return J_new
    
