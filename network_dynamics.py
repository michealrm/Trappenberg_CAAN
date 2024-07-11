import numpy as np
import scipy as sp
import math
from matplotlib import pyplot as plt


def weights_with_gaussian_kernel(N: int, a: int = 0.2):
    """
    Generates weight matrix where W[c][c\\'] denotes the weight from neuron c to neuron c\' using the Gaussian function.

    The row vector W[c] will form a Gaussian distribution around c. Similarly, the column vector W[:c\\'] will form a
    Gaussian distribution around c\\\'.

    :param N: Number of neurons in the network
    :param a: Width of the Gaussian kernel
    """
    w = np.zeros((N, N))

    for r in range(1, N + 1):
        for c in range(1, N + 1):
            w[r - 1][c - 1] = math.exp(-math.pow(r - c, 2) / 2 * math.pow(a, 2))  # TODO no two for loops :) -- linspace

    return w

class Network:
    def __init__(self, N: int, T: int, W_func = weights_with_gaussian_kernel, U = None, ode_func = None, W_args: tuple = None):
        self.W = W_func(N, *W_args) if W_args is not None else W_func(N)
        self.U = np.zeros((N, T))
        self.params = {
            'N': N,
            'T': T,
            'ode_func': ode_func
        }

    def simulate(self, external_input: np.ndarray, ivp_func, ivp_func_args: tuple = None):
        # store_vars()
        result = ivp_func(self, external_input) if ivp_func_args is None else ivp_func(self, external_input, *ivp_func_args)
        return result

    def store_network_state(self):
        pass

    def recall_network_state(self):
        pass


def nn_ode(t, Uc, W, I, tau=1, mu=2):
    # Transpose W because our W[r,c] signifies the weight from neuron r to c
    # We want to add up the weighted connections to each neuron in Uc
    t = int(t)
    Uc[Uc < 0] = 0 # rectify Uc the negative part -- set negative Uc to be 0

    Oc = Uc**2 / (1 + mu * Uc.T.dot(Uc)) # global inhibition
    return (-Uc + W.T.dot(Oc) + I[:, t]) / tau

def sp_solve_ivp(nn: Network, external_input, ode_func = nn_ode, nn_ode_args: tuple = None):
    solve_ivp_args = (nn.W, external_input) if nn_ode_args is None else (nn.W, external_input, *nn_ode_args)
    sp.integrate.solve_ivp(ode_func, [0, nn.params['T'] - 1], nn.U[:, 0], t_eval=np.arange(0, nn.params['T']), args=solve_ivp_args)

def plot_weight_matrix(nn: Network, title: str):
    w = nn.W
    plt.title(title)
    plt.imshow(w);

def plot_external_input(external_input):
    plt.title('External Input per node, per timestep')
    plt.xlabel("Time step")
    plt.ylabel("Nodes")
    plt.imshow(external_input);

def plot_firing_rate(nn: Network, title = 'U (firing rate)'):
    U = np.zeros((num_nodes, num_time_steps))
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("Nodes")
    plt.imshow(nn.U);