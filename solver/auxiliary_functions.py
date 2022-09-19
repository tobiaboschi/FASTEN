""" Definition of classes and functions needed by the different solvers

    class RegressionType: definition of the regression type
        FS = Function-on-Scalar
        FF = Function-on-Function
        FC = Function Concurrent
        SF = Scalar-on-Function

    class SelectionCriteria: definition of the selection criterion to evaluate the best model
        CV = Cross Validation
        GCV = Generalized Cross Validation
        EBIC = Extended-BIC

    class AdaptiveScheme: definition of the adaptive scheme
        NONE = No adaptive scheme is performed
        SOFT = The adaptive step is performed just on the optimal value of lambda
        FULL = a new path is investigated starting from the weights obtained at the previous path

    class AuxiliaryFunctionsFS, AuxiliaryFunctionsFF, AuxiliaryFunctionsFC:
        contain the auxiliary functions for the FS, FF and FC (and SF) model.
            prox: proximal operator of the penalty
            prox_star: conjugate function of the proximal operator
            p_star: conjugate function of the penalty
            prim_obj: primal object of the minimization problem
            dual_obj: dual object of the minimization problem
            phi_y: function phi(y) defined in the dal algorithm
            grad_phi: gradient of the function phi(y)

    class OutputSolverCore: definition of the output for the core part of each solver (FS, FF, FC, and SF)

    class OutputSolver: definition of the output for each solver (FS, FF, FC, and SF)

    class OutputPathCore: definition of the output for the core part of path solver

    class OutputPath: definition of the output for the path solver

    function standardize_A: take as input A and returns A standardized

    function plot_selection_criterion: function to plot the selection criterion for different c_lam at the end of
        path_solver

"""


import numpy as np
from numpy import linalg as LA
from sklearn.model_selection import KFold
import enum
import matplotlib.pyplot as plt


class RegressionType(enum.Enum):

    """
    Definition of the class to select regression type

    """

    FS = 1
    FC = 2
    FF = 3
    SF = 4


class SelectionCriteria(enum.Enum):

    """
    Definition of the class to select model selection criterion

    """

    GCV = 1
    EBIC = 2
    CV = 3


class AdaptiveScheme(enum.Enum):

    """
    Definition of the class to select adaptive scheme

    """

    NONE = 1
    FULL = 2
    SOFT = 3


class AuxiliaryFunctionsFS:

    def prox(self, v, par1, par2):

        """
        proximal operator of the penalty

        """

        return v / (1 + par2) * np.maximum(0, 1 - par1 / LA.norm(v, axis=1).reshape(v.shape[0], 1))

    def prox_star(self, v, par1, par2, t):

        """
        conjugate function of the proximal operator
        :param v: the argument is already divided by sigma: prox_(p*/sgm)(x/sgm = v)

        """

        return v - self.prox(v * t, par1, par2) / t

    def p_star(self, v, par1, par2):

        """
        conjugate function of the penalty

        """

        return np.sum(np.maximum(0, LA.norm(v, axis=1).reshape(v.shape[0], 1) - par1) ** 2 / (2 * par2))


    def prim_obj(self, A, x, b, lam1, lam2, wgts):

        """
        primal object of the minimization problem

        """

        norms_x = LA.norm(x, axis=1).reshape(x.shape[0], 1)

        return 0.5 * LA.norm(A @ x - b) ** 2 + lam2 / 2 * np.sum(wgts * norms_x ** 2) + lam1 * np.sum(wgts * norms_x)

    def dual_obj(self, y, z, b, lam1, lam2, wgts):

        """
         dual_obj: dual object of the minimization problem

        """

        return - (0.5 * LA.norm(y) ** 2 + np.sum(b * y) + self.p_star(z, wgts * lam1, wgts * lam2))

    def phi_y(self, y, x, b, Aty, sgm, lam1, lam2, wgts):

        """
        function phi(y) defined in the dal algorithm

        """

        return (LA.norm(y) ** 2 / 2 + np.sum(b * y) +
                LA.norm(np.sqrt((1 + wgts * sgm * lam2) / (2 * sgm)) *
                        self.prox(x - sgm * Aty, wgts * sgm * lam1, wgts * sgm * lam2)) ** 2
                - LA.norm(x) ** 2 / (2 * sgm))

    def grad_phi(self, A, y, x, b, Aty, sgm, lam1, lam2, wgts):

        """
        gradient of the function phi(y)

        """

        return y + b - A @ self.prox(x - sgm * Aty, wgts * sgm * lam1, wgts * sgm * lam2)

    def compute_coefficients_form(self, A, b, k):

        """
        compute coefficient form given the orginal matrices

        """

        # find k
        eigvals, b_basis_full = LA.eigh(b.T @ b)
        var_exp = np.cumsum(np.flip(eigvals)) / np.sum(eigvals)
        k_suggested = max(np.argwhere(var_exp > 0.95)[0][0] + 1, 2)
        if not k:
            k = k_suggested

        # find basis
        b_basis = b_basis_full[:, -k:]
        x_basis = b_basis

        # find scores
        b = b @ b_basis

        return A, b, k, k_suggested, b_basis_full, x_basis, var_exp

    def compute_curves(self, fit, b_std):

        """
        Computes the final functional estimates from the x coefficients

        """

        x_basis, indx, x, r = fit.x_basis, fit.indx, fit.x_coeffs, fit.r
        k = x.shape[1]

        if x_basis.ndim > 2:
            x_curves = b_std * (x[indx, :].reshape(r, 1, k) @
                                x_basis[indx, :, :].transpose(0, 2, 1)).reshape(r, x_basis.shape[1])

        else:
            x_curves = x[indx, :] @ (b_std.reshape(b_std.shape[0], 1) * x_basis).T

        return x_curves


class AuxiliaryFunctionsFF(AuxiliaryFunctionsFS):

    def prox(self, v, par1, par2):

        """
        proximal operator of the penalty

        """

        k = v.shape[1]
        n = np.int(v.shape[0] / k)
        v_line = v.reshape(n, k * k)

        return (v_line / (1 + par2) * np.maximum(0, 1 - par1 / LA.norm(v_line, axis=1).reshape(n, 1))).reshape(n * k, k)

    def p_star(self, v, par1, par2):

        """
        conjugate function of the penalty

        """

        k = v.shape[1]
        n = np.int(v.shape[0] / k)

        return np.sum(np.maximum(0, LA.norm(v.reshape(n, k * k), axis=1).reshape(n, 1) - par1) ** 2 / (2 * par2))

    def prim_obj(self, A, x, b, lam1, lam2, wgts):

        """
        primal object of the minimization problem

        """

        k = x.shape[1]
        n = np.int(x.shape[0] / k)
        norms_x = LA.norm(x.reshape(n, k * k), axis=1).reshape(n, 1)

        return 0.5 * LA.norm(A @ x - b) ** 2 + lam2 / 2 * np.sum(wgts * norms_x ** 2) + lam1 * np.sum(wgts * norms_x)

    def phi_y(self, y, x, b, Aty, sgm, lam1, lam2, wgts):

        """
        function phi(y) defined in the dal algorithm

        """

        k = x.shape[1]
        n = np.int(x.shape[0] / k)

        return (LA.norm(y) ** 2 / 2 + np.sum(b * y) +
                LA.norm(np.sqrt((1 + wgts * sgm * lam2) / (2 * sgm)) *
                        self.prox(x - sgm * Aty, wgts * sgm * lam1, wgts * sgm * lam2).reshape(n, k * k)) ** 2
                - LA.norm(x) ** 2 / (2 * sgm))

    def compute_coefficients_form(self, A, b, k):

        """
        compute coefficient form given the orginal matrices

        """

        n, m, _ = A.shape

        # find b basis and k
        eigvals, b_basis_full = LA.eigh(b.T @ b)
        var_exp = np.cumsum(np.flip(eigvals)) / np.sum(eigvals)
        k_suggested = max(np.argwhere(var_exp > 0.9)[0][0] + 1, 3)
        if not k:
            k = k_suggested

        # find basis
        b_basis = b_basis_full[:, -k:]
        x_basis = b_basis

        # find scores
        b = b @ b_basis
        A = (A @ b_basis).transpose(1, 0, 2).reshape(m, n * k)

        # A_coeffs = np.zeros((n, m, k))
        # A_basis = np.zeros((n, A.shape[2], k))
        # # k_suggested = -1
        # # var_exp = 0
        # for i in tqdm(range(n)):
        #     eigvals, eigenfuns = LA.eigh(A[i, :, :].T @ A[i, :, :])
        #     # var_exp += np.cumsum(np.flip(eigvals)) / np.sum(eigvals)
        #     A_basis[i, :, :] = eigenfuns[:, -k:]
        #     A_coeffs[i, :, :] = A[i, :, :] @ A_basis[i, :, :]
        # # var_exp = var_exp / n
        # x_basis = np.zeros((2, n, A.shape[2], k))
        # A = A_coeffs.transpose(1, 0, 2).reshape(m, n * k)
        # x_basis[0, :, :, :] = A_basis
        # for i in range(n):
        #     x_basis[1, i, :, :] = b_basis
        # b = b @ b_basis

        return A, b, k, k_suggested, b_basis_full, x_basis, var_exp

    def select_k_estimation(self, A_full, b_full, b_basis_full, fit):

        """
           computes k for estimation based on cv criterion

        """

        n, m, _ = A_full.shape
        k = fit.x_coeffs.shape[1]

        if k > 6:
            return fit, k
        else:

            rss_cv = []
            for ki in range(max(k, 3), 7):
                # find b basis and k
                bi_basis = b_basis_full[:, -ki:]
                bi = b_full @ bi_basis
                Ai = (A_full @ bi_basis).transpose(1, 0, 2).reshape(m, n * ki)
                AJi = Ai[:, np.repeat(fit.indx, ki)]

                kf = KFold(n_splits=5)
                kf.get_n_splits(AJi)
                rss_folds = []
                for train_index, test_index in kf.split(AJi):
                    A_cv_train, A_cv_test = AJi[train_index], AJi[test_index]
                    b_cv_train, b_cv_test = bi[train_index], bi[test_index]
                    xji = LA.solve(A_cv_train.T @ A_cv_train, A_cv_train.T @ b_cv_train)
                    rss_folds.append(LA.norm(b_cv_test - A_cv_test @ xji) ** 2)

                rss_cv.append(np.mean(rss_folds))

            k_estimation = max(k, 3) + np.argmin(rss_cv)
            if k_estimation > k:
                fit.x_basis = b_basis_full[:, -k_estimation:]
                fit.b_coeffs = b_full @ fit.x_basis
                fit.A_coeffs = (A_full @ fit.x_basis).transpose(1, 0, 2).reshape(m, n * k_estimation)
                AJ = fit.A_coeffs[:, np.repeat(fit.indx, k_estimation)]
                fit.x_coeffs = np.zeros((n * k_estimation, k_estimation))
                fit.x_coeffs[np.repeat(fit.indx, k_estimation), ] = LA.solve(AJ.T @ AJ, AJ.T @ fit.b_coeffs)

        return fit, k_estimation

    def compute_curves(self, fit, b_std):

        """
           Computes the final functional estimates from the x coefficients

        """

        x_basis, indx, x, r = fit.x_basis, fit.indx, fit.x_coeffs, fit.r
        b_std = b_std.reshape(b_std.shape[0], 1)

        if x_basis.ndim > 2:
            x_basis2 = x_basis[1, indx, :, :]
            x_curves = b_std * x_basis[0, indx, :, :] @ x[indx, :, :] @ x_basis2.transpose(0, 2, 1)
            # x_curves = b_std * x_basis[0, indx, :, :] @ x[indx, :, :] @ x_basis[1, indx, :, :].transpose(0, 1, 3, 2)
        else:
            x_curves = b_std * x_basis @ x[indx, :, :] @ x_basis.T

        return x_curves


class AuxiliaryFunctionsFC(AuxiliaryFunctionsFS):

    def prim_obj(self, A, x, b, lam1, lam2, wgts):

        """
        primal object of the minimization problem

        """

        norms_x = LA.norm(x, axis=1).reshape(x.shape[0], 1)

        return (0.5 * LA.norm(A @ x.ravel() - b) ** 2
                + lam2 / 2 * np.sum(wgts * norms_x ** 2) + lam1 * np.sum(wgts * norms_x))

    def grad_phi(self, A, y, x, b, Aty, sgm, lam1, lam2, wgts):

        """
        gradient of the function phi(y)

        """

        return y + b - A @ self.prox(x - sgm * Aty, wgts * sgm * lam1, wgts * sgm * lam2).ravel()

    def compute_coefficients_form(self, A, b, k):
        """
        compute coefficient form given the orginal matrices

        """

        n, m, _ = A.shape

        # find k
        eigvals, b_basis_full = LA.eigh(b.T @ b)
        var_exp = np.cumsum(np.flip(eigvals)) / np.sum(eigvals)
        k_suggested = max(np.argwhere(var_exp > 0.95)[0][0] + 1, 2)
        if not k:
            k = k_suggested

        # find basis
        b_basis = b_basis_full[:, -k:]
        x_basis = b_basis

        # find scores
        b = np.sum(b, axis=1)
        A = (A @ b_basis).transpose(1, 0, 2).reshape(m, n * k)

        return A, b, k, k_suggested, b_basis_full, x_basis, var_exp


class AuxiliaryFunctionsSF(AuxiliaryFunctionsFC):

    def compute_coefficients_form(self, A, b, k):
        """
        compute coefficient form given the orginal matrices

        """

        n, m, _ = A.shape
        k_suggested = -1
        if not k:
            k = 3

        # find a different basis for each feature
        print('  * performing pca for all features')
        A_coeffs = np.zeros((n, m, k))
        A_basis = np.zeros((n, A.shape[2], k))
        var_exp = 0
        for i in range(n):
            eigvals, eigenfuns = LA.eigh(A[i, :, :].T @ A[i, :, :])
            var_exp += np.cumsum(np.flip(eigvals)) / np.sum(eigvals)
            A_basis[i, :, :] = eigenfuns[:, -k:]
            A_coeffs[i, :, :] = A[i, :, :] @ A_basis[i, :, :]
        var_exp = var_exp / n

        # find other basis
        b_basis_full = None
        x_basis = A_basis

        # find scores
        A = A_coeffs.transpose(1, 0, 2).reshape(m, n * k)

        return A, b, k, k_suggested, b_basis_full, x_basis, var_exp


class OutputSolverCore:

    """
    Definition of the output class for solver_core

    """

    def __init__(self, x, xj, AJ, y, z, r, sgm, indx, time, iters, Aty, prim, dual, kkt3, convergence):
        self.x = x
        self.xj = xj
        self.AJ = AJ
        self.y = y
        self.z = z
        self.r = r
        self.sgm = sgm
        self.indx = indx
        self.time = time
        self.iters = iters
        self.Aty = Aty
        self.prim = prim
        self.dual = dual
        self.kkt3 = kkt3
        self.convergence = convergence


class OutputSolver:

    """
    Definition of the output class for solver

    """

    def __init__(self, x_curves, x_coeffs, x_basis, b_coeffs, A_coeffs, y, z, r, r_no_adaptive, indx,
                 selection_criterion_value, sgm, c_lam, alpha, lam1_max, lam1, lam2, time_tot, iters, Aty, convergence):
        self.x_curves = x_curves
        self.x_coeffs = x_coeffs
        self.x_basis = x_basis
        self.b_coeffs = b_coeffs
        self.A_coeffs = A_coeffs
        self.y = y
        self.z = z
        self.r = r
        self.r_no_adaptive = r_no_adaptive
        self.indx = indx
        self.selection_criterion_value = selection_criterion_value
        self.sgm = sgm
        self.c_lam = c_lam
        self.alpha = alpha
        self.lam1_max = lam1_max
        self.lam1 = lam1
        self.lam2 = lam2
        self.time = time_tot
        self.iters = iters
        self.Aty = Aty
        self.convergence = convergence


class OutputPathCore:

    """
    Definition of the output class for solver_path_core

    """

    def __init__(self, best_model, time_path, time_cv, r_vec, c_lam_entry_value, times_vec, iters_vec,
                 selection_criterion_vec, convergence):
        self.best_model = best_model
        self.time_path = time_path
        self.time_cv = time_cv
        self.r_vec = r_vec
        self.c_lam_entry_value = c_lam_entry_value
        self.times_vec = times_vec
        self.iters_vec = iters_vec
        self.selection_criterion_vec = selection_criterion_vec
        self.convergence = convergence


class OutputPath:

    """
    Definition of the output class for solver_path

    """

    def __init__(self, best_model, k_selection, k_estimation,
                 r_vec, selection_criterion_vec, c_lam_entry_value, c_lam_vec, alpha, lam1_vec, lam2_vec, lam1_max,
                 time_total, time_path, time_cv, time_adaptive, time_curves, iters_vec, times_vec):
        self.best_model = best_model
        self.k_selection = k_selection
        self.k_estimation = k_estimation
        self.r_vec = r_vec
        self.selection_criterion_vec = selection_criterion_vec
        self.c_lam_entry_value = c_lam_entry_value
        self.c_lam_vec = c_lam_vec
        self.alpha = alpha
        self.lam1_vec = lam1_vec
        self.lam2_vec = lam2_vec
        self.lam1_max = lam1_max
        self.time_total = time_total
        self.time_path = time_path
        self.time_cv = time_cv
        self.time_adaptive = time_adaptive
        self.time_curves = time_curves
        self.iters_vec = iters_vec
        self.times_vec = times_vec


def standardize_A(A):

    """
    function to standardize the design matrix

    """

    if len(A.shape) == 2:
        return (A - A.mean(axis=0)) / (A.std(axis=0) + 1e-32)

    else:
        for i in range(A.shape[0]):
            A[i, :, :] = (A[i, :, :] - A[i, :, :].mean(axis=0)) / (A[i, :, :].std(axis=0) + 1e-32)
        return A


def plot_selection_criterion(r, selection_criterion, alpha, grid, main=None):

    """
    plots of: r, ebic, gcv, cv for different values of alpha

    :param r: list of r_vec. Each element of the list is the r_vec values for the respective alpha in alpha_list
    :param selection_criterion: list of selection_criterion_vec. One vector for each value of alpha.
    :param alpha: vec of different value of alpha considered
    :param grid: array of all the c_lam considered (same for all alphas)
    :param main: main for selection criterion plot

    """

    # if the inputs are not list, we create them:
    if type(r) != list:
        r_list, selection_criterion_list = list(), list()
        r_list.append(r)
        selection_criterion_list.append(selection_criterion)
        alpha_vec = np.array([alpha])
        n_alpha = 1
    else:
        r_list, selection_criterion_list, alpha_vec = r, selection_criterion, alpha
        n_alpha = alpha.shape[0]

    fig, ax = plt.subplots(2, 1)

    # r
    for i in range(n_alpha):
        indx = r_list[i] != -1
        t = grid[:r_list[i].shape[0]][indx]
        ax[0].plot(t, r_list[i][indx], label=('alpha = %.2f' % alpha_vec[i]))
    ax[0].legend(loc='best')
    ax[0].set_title('selected features')
    ax[0].set_xlim([grid[0], grid[-1]])

    # selection_criterion_list
    for i in range(n_alpha):
        indx = selection_criterion_list[i] != -1
        t = grid[:selection_criterion_list[i].shape[0]][indx]
        ax[1].plot(t, selection_criterion_list[i][indx], label=('alpha = %.2f' % alpha_vec[i]))
    ax[1].legend(loc='best')
    ax[1].set_title(main)
    ax[1].set_xlim([grid[0], grid[-1]])

    plt.show()
