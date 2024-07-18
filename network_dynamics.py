import numpy as np
import scipy as sp
from matplotlib import pyplot as plt


def get_diff_matrix(nn, around_c=None):
    row_pref_stim = nn.prefStim.reshape(-1, 1)
    col_pref_stim = nn.prefStim.reshape(1, -1)

    if around_c is None:
        diff_matrix = np.fabs(col_pref_stim - row_pref_stim)
    else:
        diff_matrix = np.fabs(row_pref_stim - around_c)

    return diff_matrix


def gaussian_func(diff_matrix: np.ndarray, a: float = 0.03):
    """
    Generates weight matrix where W[c][c\\'] denotes the weight from neuron c to neuron c\' using the Gaussian function.

    The row vector W[c] will form a Gaussian distribution around c. Similarly, the column vector W[:c\\'] will form a
    Gaussian distribution around c\\\'.

    :param N: Number of neurons in the network
    :param a: Width of the Gaussian kernel
    """
    return np.exp(-np.power(diff_matrix, 2) / 2 * np.power(a, 2))


class Network:
    def __init__(self, N: int, T: int, L: float = 180., diff_matrix_func=get_diff_matrix, W_func = gaussian_func, U=None,
                 W_kwargs: dict = None):
        self.prefStim = np.linspace(-L, L, N)
        self.L = L
        self.N = N
        self.T = T
        self.W = W_func(diff_matrix_func(self), **W_kwargs) if W_kwargs is not None else W_func(
            diff_matrix_func(self))
        self.U = np.zeros((N, T)) if U is None else U

    def simulate(self, external_input: np.ndarray, ivp_func, get_U_func, ivp_func_args: tuple = None):
        # store_vars()
        result = ivp_func(self, external_input) if ivp_func_args is None else ivp_func(self, external_input,
                                                                                       *ivp_func_args)
        self.U = get_U_func(result)
        return result

    def store_network_state(self):
        pass

    def recall_network_state(self):
        pass


def nn_ode(t, Uc, W, I, tau=1, mu=2):
    # Transpose W because our W[r,c] signifies the weight from neuron r to c
    # We want to add up the weighted connections to each neuron in Uc
    t = int(t)
    Uc[Uc < 0] = 0  # rectify Uc the negative part -- set negative Uc to be 0

    Oc = Uc ** 2 / (1 + mu * Uc.T.dot(Uc))  # global inhibition
    return (-Uc + W.T.dot(Oc) + I[:, t]) / tau


def sp_solve_ivp(nn: Network, external_input, ode_func=nn_ode, nn_ode_args: tuple = None):
    solve_ivp_args = (nn.W, external_input) if nn_ode_args is None else (nn.W, external_input, *nn_ode_args)
    return sp.integrate.solve_ivp(ode_func, [0, nn.T - 1], nn.U[:, 0], t_eval=np.arange(0, nn.T), args=solve_ivp_args)


def plot_weight_matrix(nn: Network, title: str):
    w = nn.W
    plt.title(title)
    plt.imshow(w);


def plot_weight(nn: Network, weight_index_to_plot: int, title: str):
    fig, axs = plt.subplots(2, figsize=(12, 10))
    fig.suptitle(title)
    axs[0].imshow(nn.W)
    axs[1].title.set_text('Weight for c=%d' % weight_index_to_plot)
    axs[1].plot(np.arange(nn.N), nn.W[weight_index_to_plot])
    axs[1].axvline(weight_index_to_plot, linestyle='--')
    plt.show();


def plot_external_input(external_input, title='External Input per node, per timestep'):
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("Nodes")
    plt.imshow(external_input);
    plt.show();


def plot_external_input_at_t(external_input, t: int, title='External Input'):
    fig, axs = plt.subplots(2, figsize=(12, 10))
    fig.suptitle(title)
    axs[0].imshow(external_input)
    axs[1].title.set_text('t=%d' % t)
    axs[1].plot(external_input[:, t])
    plt.show();


def plot_firing_rate(nn: Network, title='Firing rate over time (U)'):
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("Nodes")
    plt.imshow(nn.U);
    plt.show();
