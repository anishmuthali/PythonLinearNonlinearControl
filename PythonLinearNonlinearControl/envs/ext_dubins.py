import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from .env import Env
from ..plotters.plot_objs import circle_with_angle, square, circle


def f(x_vec, u_vec):
    return np.array([x_vec[2]*np.cos(x_vec[3]), x_vec[2]*np.sin(x_vec[3]), u_vec[0], u_vec[1]])

def rk_step(x_n_vec, u_n_vec, dt=0.1):
        k_1 = f(x_n_vec, u_n_vec)
        k_2 = f(x_n_vec + (dt/2)*k_1, u_n_vec)
        k_3 = f(x_n_vec + (dt/2)*k_2, u_n_vec)
        k_4 = f(x_n_vec + dt*k_3, u_n_vec)
        x_np1_vec = x_n_vec + (1/6)*(k_1 + 2*k_2 + 2*k_3 + k_4)*dt
        return x_np1_vec

def step_dubins(curr_x, u, dt, method="Oylar"):
    """ step two wheeled enviroment

    Args:
        curr_x (numpy.ndarray): current state, shape(state_size, )
        u (numpy.ndarray): input, shape(input_size, )
        dt (float): sampling time
    Returns:
        next_x (numpy.ndarray): next state, shape(state_size. )

    Notes:
        TODO: deal with another method, like Runge Kutta
    """
    next_x = rk_step(curr_x, u, dt=dt)

    return next_x


class DubinsTrackEnv(Env):
    """ Two wheeled robot with constant goal Env
    """

    def __init__(self, reference_traj, x0, u1_bds=[-30,10], u2_bds=[-np.pi/4, np.pi/4]):
        """
        """
        self.config = {"state_size": 4,
                       "input_size": 2,
                       "dt": 0.1,
                       "max_step": reference_traj.shape[0]-1,
                       "input_lower_bound": (u1_bds[0], u2_bds[0]),
                       "input_upper_bound": (u1_bds[1], u2_bds[1]),
                       "car_size": 0.2,
                       "wheel_size": (0.075, 0.015)
                       }
        self.reference_traj = reference_traj
        self.x0 = x0
        super(DubinsTrackEnv, self).__init__(self.config)

    def reset(self, init_x=None):
        """ reset state

        Returns:
            init_x (numpy.ndarray): initial state, shape(state_size, )  
            info (dict): information
        """
        self.step_count = 0

        self.curr_x = self.x0

        if init_x is not None:
            self.curr_x = init_x

        # goal
        self.g_traj = self.reference_traj
    

        # clear memory
        self.history_x = [self.curr_x]
        self.history_g_x = []

        return self.curr_x, {"goal_state": self.g_traj}

    def step(self, u):
        """ step environments

        Args:
            u (numpy.ndarray) : input, shape(input_size, )
        Returns:
            next_x (numpy.ndarray): next state, shape(state_size, ) 
            cost (float): costs
            done (bool): end the simulation or not
            info (dict): information 
        """
        # clip action
        u = np.clip(u,
                    self.config["input_lower_bound"],
                    self.config["input_upper_bound"])

        # step
        next_x = step_dubins(self.curr_x, u, self.config["dt"])

        costs = 0.
        costs += 0.1 * np.sum(u**2)
        costs += np.min(np.linalg.norm(self.curr_x[np.arange(4) != 2] - self.g_traj[:,np.arange(4)!= 2], axis=1))

        # save history
        self.history_x.append(next_x.flatten())

        # update
        self.curr_x = next_x.flatten()
        # update costs
        self.step_count += 1

        return next_x.flatten(), costs, \
            self.step_count > self.config["max_step"], \
            {"goal_state": self.g_traj}

    def plot_traj(self, bad_plan, history_u):
        bad_history = [self.x0]
        good_history = [self.x0]
        for i in range(bad_plan.shape[0]):
            curr_u = bad_plan[i,:]
            next_x = step_dubins(bad_history[-1], curr_u, 0.1)
            bad_history.append(next_x)
        for i in range(history_u.shape[0]):
            curr_u = np.clip(history_u[i,:],
                            self.config["input_lower_bound"],
                            self.config["input_upper_bound"])
            next_x = step_dubins(good_history[-1], curr_u, 0.1)
            good_history.append(next_x)
        plt.plot(self.g_traj[:,0], self.g_traj[:,1], color='blue', label='HJ xtraj')
        hist_np = np.array(self.history_x).reshape(len(self.history_x), 4)
        bad_hist_np = np.array(bad_history).reshape(len(bad_history), 4)
        good_hist_np = np.array(good_history).reshape(len(good_history), 4)
        plt.plot(bad_hist_np[:,0], bad_hist_np[:,1], color='red', label='real HJ trajectory')
        plt.plot(good_hist_np[:,0], good_hist_np[:,1], color='green', label='iLQR trajectory')
        plt.legend(loc="best")
        plt.show()
