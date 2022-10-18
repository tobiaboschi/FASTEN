"""code to run the FC fasten on synthetic data"""


import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from fasten.solver_path import FASTEN
from fasten.auxiliary_functions import RegressionType, SelectionCriteria, AdaptiveScheme
from fasten.auxiliary_functions import standardize_A
from fasten.generate_sim import GenerateSimFC

if __name__ == '__main__':

    seed = np.random.randint(0, 1e5)
    # seed = 50
    np.random.seed(seed)

    # ------------------------ #
    #  choose simulation type  #
    # ------------------------ #

    regression_type = RegressionType.FC  # FF, FC, SF, FC
    GenSim = GenerateSimFC(seed)

    selection_criterion = SelectionCriteria.GCV  # CV, GCV, or EBIC
    n_folds = 5  # number of folds if cv is performed
    adaptive_scheme = AdaptiveScheme.SOFT  # type of adaptive scheme: FULL, SOFT, NONE

    easy_x = True  # if the features are easy or complex to estimate
    relaxed_criteria = True  # if True a linear regression is fitted on the features to select the best lambda
    relaxed_estimates = True  # if True a linear regression is fitted on the features before returning them

    # --------------------------- #
    #  set simulation parameters  #
    # --------------------------- #

    m = 300  # number of samples
    n = 500  # number of features
    not0 = 5  # number of non 0 features

    domain = np.array([0, 1])  # domains of the curves
    neval = 100  # number of points to construct the true predictors and the response

    mu_A = 0  # mean of features
    sd_A = 1  # standard deviation of the A Matern covariance
    l_A = 0.25  # range parameter of A Matern covariance
    nu_A = 3.5  # smoothness of A Matern covariance

    mu_eps = 0  # mean of errors
    snr = 10  # signal to noise ratio to determine sd_eps
    l_eps = 0.25  # range parameter of eps Matern covariance
    nu_eps = 1.5  # smoothness of eps Matern covariance

    # ----------------------- #
    #  set fasten parameters  #
    # ----------------------- #

    k = None  # number of FPC scores, if None automatically selected

    # c_lam_vec = 0.8  # if we chose to run for just one value of lam1 = lam1 = c_lam * lam1_max
    c_lam_vec = np.geomspace(1, 0.01, num=100)  # grid of lam1 to explore, lam1 = c_lam * lam1_max
    c_lam_vec_adaptive = np.geomspace(1, 0.0001, num=50)

    max_selected = max(50, 2 * not0)  # max number of selected features
    # max_selected = 100
    check_selection_criterion = False  # if True and the selection criterion has a discontinuity, we stop the search

    # wgts = np.ones((n, 1))  # individual penalty weights
    wgts = 1
    alpha = 0.5  # lam2 = (1-alpha) * c_lam * lam1_max

    sgm = not0 / n  # starting value of sigma
    sgm_increase = 5  # sigma increasing factor
    sgm_change = 1  # number of iteration that we wait before increasing sigma

    use_cg = True  # decide if you want to use conjugate gradient
    r_exact = 2000  # number of features such that we start using the exact method

    maxiter_nwt = 40  # nwt max iterations
    maxiter_dal = 100  # dal max iterations
    tol_nwt = 1e-6  # nwt tolerance
    tol_dal = 1e-6  # dal tolerance

    plot = False  # plot selection criteria
    print_lev = 2  # decide level of printing

    # ------------------ #
    #  create variables  #
    # ------------------ #

    if easy_x:
        # Easy x
        mu_x = 0  # mean of the true predictors
        sd_x = 1  # standard deviation of the x Matern covariance
        l_x = 0.25  # range parameter of x Matern covariance
        nu_x = 3.5  # smoothness of x Matern covariance
    else:
        # Difficult x
        mu_x = 0  # mean of the true predictors
        sd_x = 2  # standard deviation of the x Matern covariance
        l_x = 0.25  # range parameter of x Matern covariance
        nu_x = 2.5  # smoothness of x Matern covariance

    # create equispaced grid where the curves are evaluated at
    grid = np.linspace(domain[0], domain[1], neval)

    # generate design matrix A
    A = GenSim.generate_A(n, m, grid, mu_A, sd_A, l_A, nu_A)

    # generate coefficient matrix x
    x_true = GenSim.generate_x(not0, grid, sd_x, mu_x, l_x, nu_x)

    # compute errors and response
    b, eps = GenSim.compute_b_plus_eps(A, x_true, not0, grid, snr, mu_eps, l_eps, nu_eps)

    # n, m, _ = A.shape
    # b = b - eps
    #
    # x_true.shape
    #
    # # find k
    # eigvals, b_basis_full = LA.eigh(b.T @ b)
    # var_exp = np.cumsum(np.flip(eigvals)) / np.sum(eigvals)
    # k_suggested = max(np.argwhere(var_exp > 0.95)[0][0] + 1, 2)
    # if not k:
    #     k = k_suggested
    #
    # plt.figure()
    # plt.plot(grid, b.T, lw=1)
    # plt.show()
    #
    # # find basis
    # b_basis = b_basis_full[:, -50:]
    # e = np.copy(b_basis)
    # k = b_basis.shape[1]
    # x_true_expanded = (np.eye(neval) * x_true.reshape(not0, 1, neval)).reshape(not0 * neval, neval)
    # b1 = A[0:not0, :, :].transpose(1, 0, 2).reshape(m, not0 * neval) @ x_true_expanded
    # print(b - b1)
    # x_true_3d = (np.eye(neval) * x_true.reshape(not0, 1, neval))
    #
    # plt.figure()
    # plt.plot(grid, b_basis, lw=1)
    # plt.show()
    #
    # obs = 2
    # b_1 = np.zeros((neval, ))
    # for i in range(n):
    #     b_1 += A[i, obs, :] * x_true[i, :]
    # print(b_1 - b[obs, :])
    #
    # b_mat = b @ b_basis
    # A_mat = (A @ b_basis).transpose(1, 0, 2).reshape(m, n * k)
    # x_mat = x_true @ b_basis
    # x_mat_long = x_mat.reshape(n * k, 1)
    # x_mat_big_3d = np.eye(k) * x_mat.reshape(not0, 1, k)
    # x_mat_big = (np.eye(k) * x_mat.reshape(not0, 1, k)).reshape(not0 * k, k)
    #
    # k = 100
    # e = b_basis_full[:, -k:]
    #
    # ### with p = 1
    # x_true_expanded - e @ e.T @ x_true_expanded
    # (e @ e.T @ x_true.ravel()) * np.eye(neval)
    # x_true_expanded - (e @ e.T @ x_true.ravel()) * np.eye(neval)
    # A - A @ e @ e.T
    # b - A @ e @ e.T @ ((e @ e.T @ x_true.ravel()) * np.eye(neval))
    # b - A @ e @ e.T @ e @ e.T @ ((x_true.ravel()) * np.eye(neval))
    # np.sum(b, axis=1) - A @ e @ e.T @ x_true
    #
    # ### with p > 1
    # (e @ e.T @ x_true.ravel()) * np.eye(neval)
    # x_true_expanded - (e @ e.T @ x_true.ravel()) * np.eye(neval)
    # A - (A @ e).transpose(1, 0, 2).reshape(m, n * k) @ e.T
    # np.sum(b, axis=1) - (A @ e).transpose(1, 0, 2).reshape(m, n * k) @ (e.T @ x_true.reshape(n, neval, 1)).ravel()
    #
    # A.shape
    # (e.T @ x_true).shape
    # x_true.shape
    #
    # x_true.ravel().shape
    #
    #
    # A2D = A.reshape(m, neval)
    # A2D = A[0:not0, :, :].transpose(1, 0, 2).reshape(m, not0 * neval)
    #
    # _, A_basis_full = LA.eigh(A2D.T @ A2D)
    # A_basis = A_basis_full[:, -k:]
    #
    # A.shape
    #
    #
    # (A_mat @ e.T).shape
    #
    # (x_mat @ e.T).shape
    #
    # (b_mat @ e.T).shape
    #
    # (b_mat @ e.T)[0, :] - b[0, :]
    # (A_mat @ e.T)[0, :] - A[0, 0, :]
    # (A2D @ A_basis @ A_basis.T)[0, :] - A[0, 0, :]
    #
    # (x_mat @ e.T)[0, :] - x_true
    #
    #
    #
    # (A @ A_basis).shape
    #
    # A.shape
    #
    # x_mat.shape
    #
    # (b_mat @ e.T)[0:1, :] - (A_mat @ e.T)[0:1, :] * (x_mat @ e.T)
    # (b_mat @ e.T)[0:1, :] - (A_mat @ e.T)[0:1, :] @ ((x_mat @ e.T) * np.eye(neval))
    # (b_mat @ e.T)[0:2, :] - (A_mat @ e.T)[0:2, :] @ ((x_mat @ e.T) * np.eye(neval))
    #
    #
    #
    # A2D @ b_basis @ x_mat.T
    # np.sum(b, axis=1)
    #
    # np.sum(b, axis=1).ravel() - ((A2D @ A_basis @ A_basis.T) @ b_basis @ x_mat.T).ravel()
    #
    # np.sum(b, axis=1).ravel() - (A2D @ b_basis @ b_basis.T @ b_basis @ x_mat.T).ravel()
    #
    # np.sum(b, axis=1).ravel() - (A_mat @ b_basis.T @ b_basis @ x_mat.T).ravel()
    #
    #
    #
    # (np.sum(b, axis=1).ravel() - (A_mat  @ x_mat.ravel()).ravel()) / np.sum(b, axis=1).ravel()
    # b @ b_basis @ np.sum(b_basis, axis=0) - (A_mat @ x_mat.ravel()).ravel()
    #
    # b @ b_basis @ np.sum(b_basis, axis=0) - np.sum(b, axis=1).ravel()
    #
    #
    #
    # b @ b_basis - b_mat
    #
    #
    # e.shape
    #
    #
    # (A2D @ b_basis @ b_basis.T @ b_basis @ x_mat.T).ravel() - (A2D @ b_basis @ x_mat.T).ravel()
    #
    #
    #
    #
    #
    #
    # ((A2D @ A_basis @ A_basis.T) @ b_basis)
    #
    #
    #
    #
    # (b_mat @ e.T)[0:2, :] - (A_mat @ e.T)[0:2, :] @ (x_mat_big @ e.T)
    #
    # (A_mat @ e.T)[0:2, :].shape
    #
    # (x_mat_big @ e.T).T - ((x_mat @ e.T) * np.eye(neval))
    #
    # ((x_mat @ e.T) * np.eye(neval)) - ((x_mat * np.eye(neval)) @ e.T) # NO
    #
    # E = np.zeros((k, k*neval))
    # for i in range(k):
    #     E[i, (i*neval):((i + 1) * neval)] = b_basis[:, i]
    #
    #
    # E.shape
    #
    # np.sum(E[88, :])
    # np.sum(b_basis[:, 88])
    #
    #
    # E[0, 0:100] = b_basis[:, 0]
    # E[1, 100:200] = b_basis[:, 1]
    # E[2, 200:300] = b_basis[:, 2]
    # E[3, 300:400] = b_basis[:, 3]
    #
    #
    # (x_mat @ e.T).shape
    #
    #
    #
    # (x_mat * np.eye(neval)).shape
    #
    # ((x_mat @ e.T) * np.eye(neval)).shape
    #
    # (A_mat @ e.T)[0:2, :] - A2D[0:2, :]
    # (b_mat @ e.T)[0:2, :].shape
    # (A_mat @ e.T)[0:2, :].shape
    #
    #
    # (x_mat @ e.T)[0:2, :].T.shape
    # (A_mat @ e.T)[0:2, :].shape
    #
    #
    # E = np.zeros((4, 400))
    # E[0, 0:100] = b_basis[:, 0]
    # E[1, 100:200] = b_basis[:, 1]
    # E[2, 200:300] = b_basis[:, 2]
    # E[3, 300:400] = b_basis[:, 3]
    #
    # A_mat @ b_basis.T
    # A_mat.shape
    # b_basis.shape
    #
    # b @ b_basis @ np.sum(b_basis, axis=0)
    #
    #
    # print(A_mat @ x_mat_big - b_mat @ b_basis.T)
    #
    # (A_mat @ x_mat_big).shape
    # b_mat.shape
    #
    # b_mat.shape
    #
    #
    # print((A_mat @ x_mat_long).reshape(m, ) - np.sum(b, axis=1))
    # print((A_mat @ x_mat_long).reshape(m, ) - b @ b_basis @ np.sum(b_basis, axis=0))
    #
    # A[0:not0, :, :].transpose(1, 0, 2).reshape(m, not0 * neval)
    #
    # # np.sum((b_basis.T @ b_basis), axis=0).shape
    #
    # # find scores
    # b_old = np.copy(b)
    # b2 = b_old @ b_basis @ np.sum(b_basis, axis=0)
    # b1 = np.sum(b, axis=1)
    #
    # b = np.sum(b, axis=1)
    # # b = b2
    # # print(b2 - b1)
    # A = (A @ b_basis).transpose(1, 0, 2).reshape(m, n * k)

    # --------------- #
    #  standardize A  #
    # --------------- #
    print('  * standardizing A')
    A = standardize_A(A)

    # A[0, :, :] = np.zeros((m, neval))

    # A0 = A[0, :, :]
    # A[0, :, :] = A[8, :, :]
    # A[8, :, :] = A0

    # A = np.insert(A, 0, np.ones((m, neval)), axis=0)

    # --------------- #
    #  launch fasten  #
    # --------------- #
    print('')
    print('  * start fgen')
    print('  * sgm = %.4f' % sgm)

    # -------- #
    #  FASTEN  #
    # -------- #

    solver = FASTEN()
    out_path_FC = solver.solver(
        regression_type=regression_type,
        A=A, b=b, k=k, wgts=wgts,
        selection_criterion=selection_criterion, n_folds=n_folds,
        adaptive_scheme=adaptive_scheme,
        coefficients_form=False, x_basis=None,
        c_lam_vec=c_lam_vec, c_lam_vec_adaptive=c_lam_vec_adaptive,
        max_selected=max_selected, check_selection_criterion=check_selection_criterion,
        alpha=alpha, lam1_max=None,
        x0=None, y0=None, z0=None, Aty0=None,
        relaxed_criteria=relaxed_criteria, relaxed_estimates=relaxed_estimates,
        sgm=sgm, sgm_increase=sgm_increase, sgm_change=sgm_change,
        tol_nwt=tol_nwt, tol_dal=tol_dal,
        maxiter_nwt=maxiter_nwt, maxiter_dal=maxiter_dal,
        use_cg=use_cg, r_exact=r_exact,
        plot=plot, print_lev=print_lev)

    # ------------------ #
    #  model evaluation  #
    # ------------------ #

    out_FC = out_path_FC.best_model

    # MSE false positive and false negatives
    # indx = out_FC.indx
    # print(indx)
    # r = out_FC.r
    # pos_curves = np.where(indx * 1 > 0)[0]
    # false_negatives = np.sum(1 - indx[10:(not0+10)])
    # false_positives = np.sum(indx[(not0+10):]) + np.sum(indx[0:10])
    # true_positive = np.sum(indx[10:(not0+10)])
    # x_hat_true_positive = out_FC.x_curves[0:true_positive, :]
    # x_true_sub = x_true[indx[10:(not0+10)], :]

    indx = out_FC.indx
    print(indx)
    r = out_FC.r
    pos_curves = np.where(indx * 1 > 0)[0]
    false_negatives = np.sum(1 - indx[:not0])
    false_positives = np.sum(indx[10:])
    true_positive = np.sum(indx[:not0])
    x_hat_true_positive = out_FC.x_curves[0:true_positive, :]
    x_true_sub = x_true[indx[:not0], :]

    # MSE for y
    xj_curves = out_FC.x_curves
    AJ = A[indx, :, :].transpose(1, 0, 2).reshape(m, r * neval)
    xj_curves_expanded = (np.eye(neval) * xj_curves.reshape(r, 1, neval)).reshape(r * neval, neval)
    b_hat = AJ @ xj_curves_expanded
    MSEy = np.mean(LA.norm(b - b_hat, axis=1) ** 2 / LA.norm(b, axis=1) ** 2)

    # MSE for x
    resx = x_hat_true_positive - x_true_sub
    MSEx = np.mean(LA.norm(resx, axis=1) ** 2 / LA.norm(x_true_sub, axis=1) ** 2)

    print('MSEy = %.4f' % MSEy)
    print('MSEx = %f' % MSEx)
    print('false negatives = ', false_negatives)
    print('false positives = ', false_positives)

    # ----------------------- #
    #  plot estimated curves  #
    # ----------------------- #

    plot = True
    if plot:

        # # plot b observed and b without error
        # plt.plot(grid, (b-eps)[0:5, :].T, lw=1)
        # plt.gca().set_prop_cycle(None)
        # plt.plot(grid, b[0:5, :].T, '--')
        # plt.title('true (-) vs observed (--) b')
        # plt.show()

        # for i in range(5):
        #     plt.plot(grid, (b - eps)[i, :].T, lw=1)
        #     plt.gca().set_prop_cycle(None)
        #     plt.plot(grid, b[i, :].T, '--')
        #     plt.title('true (-) vs observed (--) b')
        #     plt.show()

        # plot b and b_hat (first 5 samples)
        plt.plot(grid, b[0:5, :].T, lw=1)
        plt.gca().set_prop_cycle(None)
        plt.plot(grid, b_hat[0:5, :].T, '--')
        plt.title('observed (-) vs fitted (--) responses')
        plt.show()

        # # plot x and x_hat all together
        # plt.plot(grid, out_FC.x_curves[0:, :].T, lw=1)
        # plt.gca().set_prop_cycle(None)
        # plt.plot(grid, x_true[0:, :].T, '--')
        # plt.title('true (-) vs estimated (--) x')
        # plt.show()

        # plot x and x_hat one at a time
        ind_curve = 0
        for i in range(not0):
            if indx[i]:
                plt.plot(grid, out_FC.x_curves[ind_curve, :].T, '--')
                ind_curve += 1
            plt.gca().set_prop_cycle(None)
            plt.plot(grid, x_true[i, :].T)
            plt.title('true (-) vs estimated (--) x')
            plt.show()

        for i in range(r):
            if pos_curves[i] > not0:
                plt.plot(grid, out_FC.x_curves[i, :].T, '--')
                plt.title('true (-) vs estimated (--) x')
                plt.show()

        # plt.plot(grid, out_FC.x_curves[0, :].T, lw=1)
        # plt.show()
        #
        # print(out_FC.x_curves.shape)
        # print(x_true.shape)
        #
        # ind_curve = 0
        # for i in range(not0):
        #     plt.plot(grid, out_FC.x_curves[i+1, :].T, '--')
        #     ind_curve += 1
        #     plt.gca().set_prop_cycle(None)
        #     plt.plot(grid, x_true[i, :].T)
        #     plt.title('true (-) vs estimated (--) x')
        #     plt.show()

        # plt.plot(grid, out_FC.x_curves[0, :].T, lw=1)
        # plt.gca().set_prop_cycle(None)
        # plt.plot(grid, x_true[0, :].T, '--')
        # plt.plot(grid, (out_FC.x_curves[0, :] - x_true[0, :]).T, '--')
        # plt.show()

        # print(out_FC.x_curves.shape)
        # print(np.mean(out_FC.x_curves[0, :]))

        # print(A[0, :, :] @ out_FC.x_curves[0, :])
        # print(np.mean((A[0, :, :] @ out_FC.x_curves[0, :])))
        # print(np.mean((A[2, :, :] @ out_FC.x_curves[0, :])))
