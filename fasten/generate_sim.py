""" 
Definition of classes to generate A, b, and x for the simulations in different regression framework

class GenerateSimFS, GenerateSimFF, GenerateSimFC, GenerateSimSF
"""

import numpy as np
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import multivariate_normal
from tqdm import tqdm


class GenerateSimFS:

    def __init__(self, seed):
        self.seed = seed

    def generate_A(self, n, m):

        """
        Generate design matrix A: np.array((n, m))
        """

        np.random.seed(self.seed)
        print('  * creating A')

        A = np.random.normal(0, 1, (m, n))
        return (A - A.mean(axis=0)) / (A.std(axis=0) + 1e-32)

    def generate_x(self, not0, grid, sd_x, mu_x, l_x, nu_x):

        """
        Generate coefficient matrix x: np.array((n, neval, neval))
        """

        np.random.seed(self.seed)
        print('  * creating features')

        neval = grid.shape[0]
        cov_x = sd_x ** 2 * Matern(length_scale=l_x, nu=nu_x)(grid.reshape(-1, 1))
        x_true = np.random.multivariate_normal(mu_x * np.ones(neval), cov_x, not0)
        return x_true

    def compute_b_plus_eps(self, A, x_true, not0, grid, snr, mu_eps, l_eps, nu_eps):

        """
        Compute the response the errors terms epsilon and the response b
        """

        np.random.seed(self.seed)
        print('  * computing b')

        neval = grid.shape[0]
        m = A.shape[0]
        b = A[:, 0:not0] @ x_true
        b -= b.mean(axis=0)

        # create the errors -- and their covariance using a matern process
        print('  * creating errors')

        sd_eps = np.std(b) / np.sqrt(snr)
        cov_eps = sd_eps ** 2 * Matern(length_scale=l_eps, nu=nu_eps)(grid.reshape(-1, 1))
        eps = np.random.multivariate_normal(mu_eps * np.ones(neval), cov_eps, m)
        eps -= eps.mean(axis=0)

        b += eps

        return b, eps


class GenerateSimFF:

    def __init__(self, seed):

        self.seed = seed

    def generate_A(self, n, m, grid, mu_A, sd_A, l_A, nu_A, test=False):

        """
        Generate design matrix A: np.array((n, m, neval))
        """

        np.random.seed(self.seed)
        print('  * creating A')

        # neval = grid.shape[0]
        # A = np.zeros((n, int(m+np.floor(m/3)), neval))
        # for i in tqdm(range(n)):
        #     cov_Ai = sd_A ** 2 * Matern(length_scale=l_A, nu=nu_A)(grid.reshape(-1, 1))
        #     Ai = np.random.multivariate_normal(mu_A * np.ones(neval), cov_Ai, int(m+np.floor(m/3)))
        #     A[i, :, :] = (Ai - Ai.mean(axis=0)) / Ai.std(axis=0)  # each A should already have std = 1 and mean = 0
        #
        # if test:
        #     return A[:, 0:m, :], A[:, m:, :]
        # else:
        #     return A[:, 0:m, :]

        neval = grid.shape[0]
        if test:

            A = np.zeros((n, int(m+np.floor(m/3)), neval))
            for i in tqdm(range(n)):
                cov_Ai = sd_A ** 2 * Matern(length_scale=l_A, nu=nu_A)(grid.reshape(-1, 1))
                Ai = np.random.multivariate_normal(mu_A * np.ones(neval), cov_Ai, int(m+np.floor(m/3)))
                A[i, 0:m, :] = (Ai[0:m, ] - Ai[0:m, ].mean(axis=0)) / (Ai[0:m, ].std(axis=0) + 1e-32)
                A[i, m:, :] = (Ai[m:, ] - Ai[m:, ].mean(axis=0)) / (Ai[m:, ].std(axis=0) + 1e-32)

            return A[:, 0:m, :], A[:, m:, :]

        else:
            A = np.zeros((n, m, neval))
            for i in tqdm(range(n)):
                cov_Ai = sd_A ** 2 * Matern(length_scale=l_A, nu=nu_A)(grid.reshape(-1, 1))
                Ai = np.random.multivariate_normal(mu_A * np.ones(neval), cov_Ai, m)
                A[i, :, :] = (Ai - Ai[0:m, ].mean(axis=0)) / (Ai.std(axis=0) + 1e-32)

            return A

    def generate_x(self, not0, grid, x_npeaks_set, x_sd_min, x_sd_max, x_range):

        """
        Generate coefficient matrix x: np.array((n, neval, neval))
        """

        np.random.seed(self.seed)
        print('  * creating features')

        neval = grid.shape[0]
        grid_expanded = np.outer(grid, np.ones(neval))
        pos = np.empty((neval, neval, 2))
        pos[:, :, 0], pos[:, :, 1] = grid_expanded, grid_expanded.T
        sign_vec = np.array([1, -1])

        x_true = np.zeros((not0, neval, neval))

        for i in range(not0):
            n_peaks = np.random.choice(x_npeaks_set, 1)[0]
            peaks_x, peaks_y = np.random.choice(grid, n_peaks), np.random.choice(grid, n_peaks)
            sd_peaks = np.random.uniform(low=x_sd_min, high=x_sd_max, size=(n_peaks,))
            x_temp = np.zeros((neval, neval))
            for j in range(n_peaks):
                x_temp += np.random.choice(sign_vec, 1) * multivariate_normal(mean=[peaks_x[j], peaks_y[j]],
                                                                              cov=[[sd_peaks[j], 0],
                                                                                   [0, sd_peaks[j]]]).pdf(pos)

            x_true[i, :, :] = (x_range[1] - x_range[0]) * \
                              (x_temp - np.min(x_temp)) / (np.max(x_temp) - np.min(x_temp)) + \
                              x_range[0]

        return x_true

    def compute_b_plus_eps(self, A, x_true, not0, grid, snr, mu_eps, l_eps, nu_eps):

        """
        Compute the response the errors terms epsilon and the response b
        """

        np.random.seed(self.seed)
        print('  * computing b')
        neval = grid.shape[0]
        m = A.shape[1]

        b = A[0:not0, :, :].transpose(1, 0, 2).reshape(m, not0 * neval) @ x_true.reshape(not0 * neval, neval)
        b -= b.mean(axis=0)

        # create the errors -- and their covariance using a matern process
        print('  * creating errors')

        sd_eps = np.std(b) / np.sqrt(snr)
        cov_eps = sd_eps ** 2 * Matern(length_scale=l_eps, nu=nu_eps)(grid.reshape(-1, 1))
        eps = np.random.multivariate_normal(mu_eps * np.ones(neval), cov_eps, m+1)[1:, :]
        eps -= eps.mean(axis=0)

        b += eps

        return b, eps


class GenerateSimFC(GenerateSimFF):

    def __init__(self, seed):
        
        self.seed = seed

    def generate_x(self, not0, grid, sd_x, mu_x, l_x, nu_x):

        """
        Generate coefficient matrix x: np.array((n, neval, neval))
        """

        np.random.seed(self.seed)
        # np.random.seed(self.seed + 1)
        # np.random.seed(np.random.randint(0, 1e5))
        print('  * creating features')

        neval = grid.shape[0]

        cov_x = sd_x ** 2 * Matern(length_scale=l_x, nu=nu_x)(grid.reshape(-1, 1))
        x_true = np.random.multivariate_normal(mu_x * np.ones(neval), cov_x, not0)
        return x_true

    def compute_b_plus_eps(self, A, x_true, not0, grid, snr, mu_eps, l_eps, nu_eps):

        """
        Compute the response the errors terms epsilon and the response b
        """

        np.random.seed(self.seed)
        # np.random.seed(self.seed + 2)
        # np.random.seed(np.random.randint(0, 1e5))
        print('  * computing b')

        neval = grid.shape[0]
        m = A.shape[1]
        x_true_expanded = (np.eye(neval) * x_true.reshape(not0, 1, neval)).reshape(not0 * neval, neval)
        # b = A[10:(not0+10), :, :].transpose(1, 0, 2).reshape(m, not0 * neval) @ x_true_expanded
        b = A[0:not0, :, :].transpose(1, 0, 2).reshape(m, not0 * neval) @ x_true_expanded

        # b2 = 0
        # for i in range(not0):
        #     b2 += A[i, :, :] * np.repeat(x_true[i, :].reshape(1, neval), m, axis=0)
        # print(np.sum(b - b2))

        b -= b.mean(axis=0)

        # create the errors -- and their covariance using a matern process
        print('  * creating errors')

        sd_eps = np.std(b) / np.sqrt(snr)
        cov_eps = sd_eps ** 2 * Matern(length_scale=l_eps, nu=nu_eps)(grid.reshape(-1, 1))
        eps = np.random.multivariate_normal(mu_eps * np.ones(neval), cov_eps, m)
        eps -= eps.mean(axis=0)

        b += eps

        return b, eps


class GenerateSimSF(GenerateSimFC):

    def __init__(self, seed):
       
        self.seed = seed

    def compute_b_plus_eps(self, A, x_true, not0, grid, snr, mu_eps, l_eps, nu_eps):

        """
        Compute the response the errors terms epsilon and the response b
        """

        np.random.seed(self.seed)
        print('  * computing b')

        neval = grid.shape[0]
        m = A.shape[1]
        # x_true_expanded = (np.eye(neval) * x_true.reshape(not0, 1, neval)).reshape(not0 * neval, neval)
        # b = np.sum(A[0:not0, :, :].transpose(1, 0, 2).reshape(m, not0 * neval) @ x_true_expanded, axis=1)
        b = A[0:not0, :, :].transpose(1, 0, 2).reshape(m, not0 * neval) @ x_true.ravel()
        b -= b.mean(axis=0)

        # create the errors -- and their covariance using a matern process
        print('  * creating errors')

        sd_eps = np.std(b) / np.sqrt(snr)
        eps = np.random.normal(0, sd_eps, (m, ))
        eps -= eps.mean(axis=0)

        b += eps

        return b, eps