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
    def __init__(self, nx, nu, NX, NU, stage_cost,\
            dynamics, terminal_cost=None, constraints=None, x_bounds=None, u_bounds=None, x_values=None, u_values=None):
        
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

        stage_cost : function
            stage cost function R^{nx} x R^{nu} -> R
            
        dynamics : function
            dynamics function R^{nx} x R^{nu} -> R^{nx}

        terminal_cost : function
            terminal_cost function R^{nx} -> R

        constraints : function
            constraints function R^{nx} x R^{nu} -> R^{ng}

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

    def DPOperator(self, J):
        '''
        Compute updated value function using DP operator, i.e, for all x, J_k(x) = min_u l(x,u) + J_{k+1}(f(x,u))

        Parameters
        ----------

        J : np.ndarray (NDX)
            value function in tabular form

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

        # loop over states
        for j in range(NDX):
            x_ = np.atleast_2d(X[j,:]).T

            # loop over inputs
            for k in range(NDU):
                u_ = np.atleast_2d(U[k,:]).T

                # integrate dynamics
                x_next = self.dynamics(x_,u_)

                # project onto state grid
                idx_next, x_next_p = project_onto_homogeneous_grid(x_next, x_values)

                if idx_next is None:
                    continue

                # obtain index in reshaped form
                idx_next_rs = np.unravel_index(np.ravel_multi_index(idx_next, [NX]*nx), (NDX))

                # evaluate argument of minimization
                J_ = self.stage_cost(x_, u_) + J[idx_next_rs]
                # print("u = [%f, %f], x = [%f, %f], x_+ = [%f, %f], J = %f, J_opt = % f" % (u_[0], u_[1], x_[0], x_[1], np.squeeze(x_next_p[0]), np.squeeze(x_next_p[1]), J_, J_new[j]))
                if J_ < J_new[j]:
                    # import pdb; pdb.set_trace()
                    J_new[j] = J_
                    U_opt[j,:] = u_.T

        return J_new, U_opt
    
