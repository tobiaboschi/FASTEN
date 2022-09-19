"""code to run the FF solver on synthetic data"""


import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from solver.solver_path import FASTEN
from solver.auxiliary_functions import RegressionType, SelectionCriteria, AdaptiveScheme
from solver.auxiliary_functions import standardize_A
from solver.generate_sim import GenerateSimFF
import seaborn as sns


if __name__ == '__main__':

    #seed = np.random.randint(1, 2**30, 1)
    seed = 1
    # np.random.seed(seed)

    # ------------------------ #
    #  choose simulation type  #
    # ------------------------ #

    regression_type = RegressionType.FF
    GenSim = GenerateSimFF(seed)

    selection_criterion = SelectionCriteria.CV
    n_folds = 5  # number of folds if cv is performed
    adaptive_scheme = AdaptiveScheme.SOFT  # type of adaptive scheme: FULL, SOFT, NONE

    easy_x = True  # if the features are easy or complex to estimate
    relaxed_criteria = True  # if True a linear regression is fitted on the features to select the best lambda
    relaxed_estimates = True  # if True a linear regression is fitted on the features before returning them
    select_k_estimation = True  # if True we allow k to change k for estimation (chosen based on CV)

    # ----------------------------- #
    #  other simulation parameters  #
    # ----------------------------- #

    m = 300  # number of samples
    n = 600  # number of features
    not0 = 30  # number of non 0 features

    domain = np.array([0, 1])  # domains of the curves
    neval = 100  # number of points to construct the true predictors and the response

    k = None  # number of FPC scores, if None automatically selected

    if easy_x:
        # Easy x
        x_npeaks_set = np.array([1])  # number of possible peaks of the features
        x_sd_min = 0.2  # minimum sd of the features peaks
        x_sd_max = 0.3  # max sd of the features peaks
        x_range = np.array([-1, 1])  # max and min values of the x
    else:
        # Difficult x
        x_npeaks_set = np.array([2, 3])  # number of possible peaks of the features
        x_sd_min = 0.01  # minimum sd of the features peaks
        x_sd_max = 0.15  # max sd of the features peaks
        x_range = np.array([-1, 1])  # max and min values of the x
        # x_npeaks_set = np.array([2, 7])  # number of possible peaks of the features
        # x_sd_min = 0.01  # minimum sd of the features peaks
        # x_sd_max = 0.01  # max sd of the features peaks
        # x_range = np.array([-1, 1])  # max and min values of the x

    mu_A = 0  # mean of features
    sd_A = 1  # standard deviation of the A Matern covariance
    l_A = 0.25  # range parameter of A Matern covariance
    nu_A = 3.5  # smoothness of A Matern covariance

    mu_eps = 0  # mean of errors
    snr = 10  # signal to noise ratio to determine sd_eps
    l_eps = 0.25  # range parameter of eps Matern covariance
    nu_eps = 2.5  # smoothness of eps Matern covariance

    # ----------------------- #
    #  set solver parameters  #
    # ----------------------- #

    # c_lam_vec = 0.3  # if we chose to run for just one value of lam1 = lam1 = c_lam * lam1_max
    c_lam_vec = np.geomspace(1, 0.01, num=100)  # grid of lam1 to explore, lam1 = c_lam * lam1_max
    c_lam_vec_adaptive = np.geomspace(1, 0.0001, num=50)

    # max_selected = max(50, 2 * not0)  # max number of selected features
    max_selected = 100

    wgts = np.ones((n, 1))  # individual penalty weights
    alpha = 0.2  # lam2 = (1-alpha) * c_lam * lam1_max

    sgm = not0 / n  # starting value of sigma
    sgm_increase = 5  # sigma increasing factor
    sgm_change = 1  # number of iteration that we wait before increasing sigma

    use_cg = True  # decide if you want to use conjugate gradient
    r_exact = 2000  # number of features such that we start using the exact method

    maxiter_nwt = 40  # nwt max iterations
    maxiter_dal = 100  # dal max iterations
    tol_nwt = 1e-6  # nwt tolerance
    tol_dal = 1e-6  # dal tolerance

    plot = True  # plot selection criteria
    print_lev = 2  # decide level of printing

    # ------------------ #
    #  create variables  #
    # ------------------ #

    # create equispaced grid where the curves are evaluated at
    grid = np.linspace(domain[0], domain[1], neval)
    grid_expanded = np.outer(grid, np.ones(neval))

    # generate design matrix A
    A, A_test = GenSim.generate_A(n, m, grid, mu_A, sd_A, l_A, nu_A, test=True)

    # generate coefficient matrix x
    x_true = GenSim.generate_x(not0, grid, x_npeaks_set, x_sd_min, x_sd_max, x_range)

    # compute errors and response
    b, eps = GenSim.compute_b_plus_eps(A, x_true, not0, grid, snr, mu_eps, l_eps, nu_eps)
    b_test, eps_test = GenSim.compute_b_plus_eps(A_test, x_true, not0, grid, snr, mu_eps, l_eps, nu_eps)

    # ------------ #
    #  some plots  #
    # ------------ #

    # # plot A and b
    # plt.plot(grid, A[0:5, 0, :].T, lw=1)
    # plt.gca().set_prop_cycle(None)
    # plt.plot(grid, b[0:5, :].T, '--')
    # plt.show()

    # # plot errors and b
    # plt.plot(grid, eps[0, :].T, lw=1)
    # plt.gca().set_prop_cycle(None)
    # plt.plot(grid, b[0, :].T, '--')
    # plt.show()

    # # plot b with and without errors
    # plt.plot(grid, (b-eps)[0, :].T, lw=1)
    # plt.gca().set_prop_cycle(None)
    # plt.plot(grid, b[0, :].T, '--')
    # plt.show()

    # # plot b basis
    # eigvals, eigenfuns = LA.eigh(b.T @ b)
    # for i in range(1,10):
    #     plt.plot(eigenfuns[:, -i])
    #     plt.show()

    # --------------------- #
    #  compute marginal R2  #
    # --------------------- #

    # # coefficient forms
    # b_std = b.std(axis=0) + 1e-32
    # b = (b - b.mean(axis=0)) / b_std
    # eigvals, eigenfuns = LA.eigh(b.T @ b)
    # var_exp = np.cumsum(np.flip(eigvals)) / np.sum(eigvals)
    # k_suggested = max(np.argwhere(var_exp > 0.9)[0][0] + 1, 3)
    # if not force_k:
    #     k = k_suggested
    # b_basis = eigenfuns[:, -k:]
    # x_basis = b_basis
    # b = b @ b_basis
    # A = (A @ b_basis).transpose(1, 0, 2).reshape(m, n * k)
    #
    # R2 = []
    # sstot = np.sum(LA.norm(b, axis=1) ** 2)
    # from tqdm import tqdm
    # for i in tqdm(range(n)):
    #     indx = [False] * n
    #     indx[i] = True
    #     indx_krep = np.repeat(indx, k)
    #     A_r2 = A[:, indx_krep]
    #     x_r2 = LA.solve(A_r2.T @ A_r2, A_r2.T @ b)
    #     b_hat = A_r2 @ x_r2
    #     resy = b - b_hat
    #     ssres = np.sum(LA.norm(resy, axis=1) ** 2)
    #     R2.append(1 - ssres/sstot)

    # --------------- #
    #  standardize A  #
    # --------------- #
    print('  * standardizing A')
    A = standardize_A(A)

    # --------------- #
    #  launch solver  #
    # --------------- #
    print('')
    print('  * start fgen')
    print('  * sgm = %.4f' % sgm)

    # -------- #
    #  FASTEN  #
    # -------- #
    
    solver = FASTEN()
    out_path_FF = solver.solver(
        regression_type=regression_type,
        A=A, b=b, k=k, wgts=wgts,
        selection_criterion=selection_criterion, n_folds=n_folds,
        adaptive_scheme=adaptive_scheme,
        coefficients_form=False, x_basis=None,
        c_lam_vec=c_lam_vec, c_lam_vec_adaptive=c_lam_vec_adaptive,
        max_selected=max_selected,
        alpha=alpha, lam1_max=None,
        x0=None, y0=None, z0=None, Aty0=None,
        relaxed_criteria=relaxed_criteria, relaxed_estimates=relaxed_estimates,
        select_k_estimation=select_k_estimation,
        sgm=sgm, sgm_increase=sgm_increase, sgm_change=sgm_change,
        tol_nwt=tol_nwt, tol_dal=tol_dal,
        maxiter_nwt=maxiter_nwt, maxiter_dal=maxiter_dal,
        use_cg=use_cg, r_exact=r_exact,
        plot=plot, print_lev=print_lev)

    out_FF = out_path_FF.best_model

    # ------------------ #
    #  model evaluation  #
    # ------------------ #

    # false positive and false negatives
    indx = out_FF.indx
    r = out_FF.r
    pos_curves = np.where(indx * 1 > 0)[0]
    false_negatives = np.sum(1-indx[0:not0])
    false_positives = np.sum(indx[not0:])
    true_positive = np.sum(indx[0:not0])
    x_hat_true_positive = out_FF.x_curves[0:true_positive, :, :]
    x_true_sub = x_true[indx[0:not0], :, :]

    # b_std = b.std(axis=0) + 1e-32
    # bs = (b - b.mean(axis=0)) / b_std
    # eigvals, eigenfuns = LA.eigh(bs.T @ bs)
    # # lam2 = out_FF.lam2
    # # for i in range(1, 10):
    # #     print(eigenfuns[:, -i].std())
    # from sklearn.model_selection import KFold
    #
    # var_surf = []
    # var_surf_d = []
    # var_surf_d2 = []
    # norm_surf = []
    # norm_surf_d = []
    # norm_surf_d2 = []
    # rss_cv = []
    # gcv_list = []
    # ebic_list = []
    # MSEx_list = []
    # my_range = range(3, 10)
    # for ki in my_range:
    #     # find b basis and k
    #     b_basis = eigenfuns[:, -ki:]
    #     x_basis = b_basis
    #     bi = bs @ b_basis
    #     Ai = (A @ b_basis).transpose(1, 0, 2).reshape(m, n * ki)
    #     AJi = Ai[:, np.repeat(indx, ki)]
    #
    #     xj = LA.solve(AJi.T @ AJi, AJi.T @ bi)
    #     df_core = LA.inv(AJi.T @ AJi + out_FF.lam2 * np.eye(out_FF.r * ki))
    #     df = np.trace(AJi @ df_core @ AJi.T)
    #     rss = LA.norm(bi - AJi @ xj) ** 2
    #     gcv_list.append(rss / (m - ki * df) ** 2)
    #     ebic_list.append(np.log(rss / m) + df * np.log(m) / m + df * np.log(out_FF.r + 1e-32) / m)
    #
    #     kf = KFold(n_splits=5)
    #     kf.get_n_splits(AJi)
    #     rss_folds = []
    #     for train_index, test_index in kf.split(AJi):
    #         A_cv_train, A_cv_test = AJi[train_index], AJi[test_index]
    #         b_cv_train, b_cv_test = bi[train_index], bi[test_index]
    #         xj = LA.solve(A_cv_train.T @ A_cv_train, A_cv_train.T @ b_cv_train)
    #         res = b_cv_test - A_cv_test @ xj
    #         rss = LA.norm(res) ** 2
    #         rss_folds.append(rss)
    #
    #     rss_cv.append(np.mean(rss_folds))

        # xj = LA.solve(AJi.T @ AJi, AJi.T @ bi).reshape(r, ki , ki)
        # x_curves = b_std.reshape(b_std.shape[0], 1) * x_basis @ xj @ x_basis.T
        # x_curves_sub = x_curves[0:true_positive, :, :]
        # resx = (x_curves_sub - x_true_sub).reshape(true_positive, neval ** 2)
        # MSEx_list.append(np.mean(LA.norm(resx, axis=1) ** 2
        #                          / LA.norm(x_true_sub.reshape(true_positive, neval ** 2), axis=1) ** 2))

    #     # x_lines = x_curves.reshape(r, neval * neval)
    #     # dx = grid[1] - grid[0]
    #     # x_deriv = np.copy(x_lines)
    #     # x_deriv2 = np.copy(x_lines)
    #     # for c in range(r):
    #     #     x_deriv[c, :] = np.gradient(x_lines[c, :], dx)
    #     #     x_deriv2[c, :] = np.gradient(x_deriv[c, :], dx)
    #     # sdi = np.mean(x_lines.std(axis=1))
    #     # normi = np.mean(LA.norm(x_lines, 1) ** 2)
    #     # sdi_d = np.mean(x_deriv.std(axis=1))
    #     # normi_d = np.mean(LA.norm(x_deriv, 1) ** 2)
    #     # sdi_d2 = np.mean(x_deriv2.std(axis=1))
    #     # normi_d2 = np.mean(LA.norm(x_deriv2, 1) ** 2)
    #     # norm_surf.append(normi)
    #     # var_surf.append(sdi)
    #     # norm_surf_d.append(normi_d)
    #     # var_surf_d.append(sdi_d)
    #     # norm_surf_d2.append(normi_d2)
    #     # var_surf_d2.append(sdi_d2)
    #     # print('')
    #     # print('ki = ', ki)
    #     # print('sdi = ', sdi)
    #     # print('normi = ', normi)
    #     # print('sdi_d = ', sdi_d)
    #     # print('normi_d = ', normi_d)
    #     # print('sdi_d2 = ', sdi_d2)
    #     # print('normi_d2 = ', normi_d2)
    #     # print('')
    #     # df_core = LA.inv(AJ.T @ AJ + lam2 * np.eye(r * ki))
    #     # df = np.trace(AJ @ df_core @ AJ.T)
    #     # res = bi - AJ @ xj
    #     # rss = LA.norm(res) ** 2
    #     # ebic = np.log(rss / m) + df * np.log(m) / m + df * np.log(r + 1e-32) / m
    #     # print('')
    #     # print('k =', ki)
    #     # print('ebic =', ebic)
    #     # print('rss =', rss / m)
    #     # print('df =', df)
    #     # print('')

    # print('')
    # print('rss_cv vector')
    # print(rss_cv)
    # print('')
    # print('')
    # print('MSEx vector')
    # print(MSEx_list)
    # print('')

    # # plt.plot(my_range, rss_cv)

    # # plt.plot(my_range, var_surf)
    # # plt.plot(my_range, var_surf_d)
    # # plt.plot(my_range, var_surf_d2)

    # # plt.plot(my_range, norm_surf)
    # # plt.plot(my_range, norm_surf_d)
    # # plt.plot(my_range, norm_surf_d2)

    # MSE for y
    xj_curves = out_FF.x_curves
    AJ = A[indx, :, :].transpose(1, 0, 2).reshape(m, r * neval)
    b_hat = AJ @ xj_curves.reshape(r * neval, neval)

    # x_est = np.zeros((n * neval, neval))
    # indx_neval_rep = np.repeat(out_FF.indx, neval)
    # x_est[indx_neval_rep, :] = out_FF.x_curves.reshape(r * neval, neval)
    # b_hat_2 = A.transpose(1, 0, 2).reshape(m, n * neval) @ x_est

    resy = b - b_hat
    MSEy = LA.norm(resy) ** 2 / (m * neval ** 2)
    MSEy_std2 = np.mean(LA.norm(resy, axis=1) ** 2 / LA.norm(b, axis=1) ** 2)
    MSEy_std = np.mean(LA.norm(resy, axis=1)/ LA.norm(b, axis=1))
    # MSEy = np.mean(LA.norm(resy, axis=1) ** 2) / neval ** 2
    # MSEy_std2 = np.mean(LA.norm(resy / neval, axis=1) ** 2 / LA.norm(b / neval, axis=1) ** 2)

    # MSE for x
    resx = (x_hat_true_positive - x_true_sub).reshape(true_positive, neval ** 2)
    # MSEx = LA.norm(resx) ** 2 / (true_positive * neval ** 4)
    MSEx_std2 = np.mean(LA.norm(resx, axis=1) ** 2 / LA.norm(x_true_sub.reshape(true_positive, neval ** 2), axis=1) ** 2)
    MSEx_std = np.mean(LA.norm(resx, axis=1) / LA.norm(x_true_sub.reshape(true_positive, neval ** 2), axis=1))
    # MSEx = np.mean(LA.norm(resx.reshape(true_positive, neval * neval), axis=1) ** 2) / neval ** 4
    # MSEx_std2 = np.mean(LA.norm(resx / neval ** 2, axis=1) ** 2 / LA.norm(x_true_sub / neval ** 2, axis=1) ** 2)
    resx_old = x_hat_true_positive - x_true_sub
    MSEx_old = np.mean(LA.norm(resx_old, axis=1) ** 2 / LA.norm(x_true_sub, axis=1) ** 2)

    # Out-of-sample error
    m_test = b_test.shape[0]
    xj_curves = out_FF.x_curves
    AJ = A_test[indx, :, :].transpose(1, 0, 2).reshape(m_test, r * neval)
    b_hat_out = AJ @ xj_curves.reshape(r * neval, neval)
    resy = b_test - b_hat_out
    MSEy = LA.norm(resy) ** 2 / (int(np.floor(m/3)-1) * neval ** 2)
    # MSEy_std2 = np.mean(LA.norm(resy, axis=1) ** 2 / LA.norm(b, axis=1) ** 2)
    MSEy_std_out = np.mean(LA.norm(resy, axis=1) / LA.norm(b_test, axis=1))

    print('MSEy = %f' % MSEy_std_out)
    print('MSEx2 = %f' % MSEx_std2)
    print('MSEx = %f' % MSEx_std)
    print('false negatives = ', false_negatives)
    print('false positives = ', false_positives)
    # print('cv list =', rss_cv)
    # print('gcv list =', gcv_list)
    # print('ebic list =', ebic_list)
    # print('features order of entry:')
    # print(np.argsort(-out_path_FF.c_lam_entry_value[out_path_FF.c_lam_entry_value > 0]))

    # ------------------------- #
    #  plot estimated surfaces  #
    # ------------------------- #

    # # plot b and b - eps
    # plt.plot(grid, (b - eps)[0:5, :].T, lw=1)
    # plt.gca().set_prop_cycle(None)
    # plt.plot(grid, b[0:5, :].T, '--')
    # plt.show()

    # plot b - eps and b_hat
    # plt.plot(grid, b[0:5, :].T, lw=1)
    # plt.gca().set_prop_cycle(None)
    # plt.plot(grid, b_hat[0:5, :].T, '--')
    # plt.show()

    # ind_curve = 0
    # for i in range(not0):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(1, 2, 1, projection='3d')
    #     ax.set_zlim(x_range)
    #     if indx[i]:
    #         ax.plot_wireframe(grid_expanded, grid_expanded.T, out_FF.x_curves[ind_curve, :, :],
    #                           color='salmon', alpha=0.3)
    #         ind_curve += 1

    #     ax = fig.add_subplot(1, 2, 2, projection='3d')
    #     ax.set_zlim(x_range)
    #     ax.plot_wireframe(grid_expanded, grid_expanded.T, x_true[i, :, :],
    #                       alpha=0.3)
    #     plt.show()

    # for i in range(r):
    #     if pos_curves[i] > not0:
    #         fig = plt.figure()
    #         ax = fig.add_subplot(1, 2, 1, projection='3d')
    #         ax.set_zlim(x_range)
    #         ax.plot_wireframe(grid_expanded, grid_expanded.T, out_FF.x_curves[i, :, :],
    #                           color='salmon', alpha=0.3)
    #         ax = fig.add_subplot(1, 2, 2, projection='3d')
    #         ax.set_zlim(x_range)
    #         plt.show()

    # how_many = 5
    # plt.figure()
    # sns.set_theme()
    # sns.set_style("ticks")
    # size = 14
    # plt.rcParams.update({'font.size': size,
    #                         'axes.labelsize': size+2,
    #                         'axes.titlesize': size+2,
    #                         'xtick.labelsize':size,
    #                         'ytick.labelsize':size})
    # # plt.figure(figsize=(6,4))
    # plt.plot(grid, b_hat_out[0:how_many, :].T, lw=2)
    # plt.gca().set_prop_cycle(None)
    # plt.plot(grid, b_test[0:how_many, :].T, '--')
    # # plt.xticks(rotation=45, fontsize='xx-small', rotation_mode='anchor')
    # # plt.ylabel('Response', fontsize=15)
    # # plt.title('Response and errors', fontsize=15)
    # line = plt.Line2D([0,1],[0,1],linestyle='-', color='grey')
    # line2 = plt.Line2D([0,1],[0,1],linestyle='--', color='grey')
    # plt.legend([line, line2],['Estimated curve', 'Observed curve'], loc='upper left')
    # # plt.show()
    # plt.savefig('out_of_sample_prediction.pdf', bbox_inches="tight", transparent=True)


    # plt.figure()
    # sns.set_theme()
    # sns.set_style("ticks")
    # size = 14
    # plt.rcParams.update({'font.size': size,
    #                         'axes.labelsize': size+2,
    #                         'axes.titlesize': size+2,
    #                         'xtick.labelsize':size,
    #                         'ytick.labelsize':size})
    # # plt.figure(figsize=(6,4))
    # plt.plot(grid, (b_test-eps_test)[0:how_many, :].T, lw=2)
    # plt.gca().set_prop_cycle(None)
    # #plt.plot(grid, b[0:5, :].T, '--')
    # #plt.xticks(rotation=45, fontsize='xx-small', rotation_mode='anchor')
    # #plt.ylabel('Response', fontsize=15)
    # #plt.title('Response and errors', fontsize=15)
    # #line = plt.Line2D([0,1],[0,1],linestyle='-', color='grey')
    # #line2 = plt.Line2D([0,1],[0,1],linestyle='--', color='grey')
    # #plt.legend([line, line2],['True signal', 'Observed curve'], loc='upper left')
    # plt.savefig('true_signal.pdf', bbox_inches="tight", transparent=True)


    # plt.figure()
    # sns.set_theme()
    # sns.set_style("ticks")
    # size = 14
    # plt.rcParams.update({'font.size': size,
    #                         'axes.labelsize': size+2,
    #                         'axes.titlesize': size+2,
    #                         'xtick.labelsize':size,
    #                         'ytick.labelsize':size})
    # # plt.figure(figsize=(6,4))
    # plt.plot(grid, (eps_test)[0:how_many, :].T, lw=2)
    # plt.gca().set_prop_cycle(None)
    # #plt.plot(grid, b[0:5, :].T, '--')
    # #plt.xticks(rotation=45, fontsize='xx-small', rotation_mode='anchor')
    # #plt.ylabel('Response', fontsize=15)
    # #plt.title('Response and errors', fontsize=15)
    # #line = plt.Line2D([0,1],[0,1],linestyle='-', color='grey')
    # #line2 = plt.Line2D([0,1],[0,1],linestyle='--', color='grey')
    # #plt.legend([line, line2],['True signal', 'Observed curve'], loc='upper left')
    # plt.savefig('error.pdf', bbox_inches="tight", transparent=True)


    # sns.set_theme()
    # sns.set_style("ticks")
    # size = 14
    # plt.rcParams.update({'font.size': size,
    #                         'axes.labelsize': size+2,
    #                         'axes.titlesize': size+2,
    #                         'xtick.labelsize':size,
    #                         'ytick.labelsize':size})
    # ind_curve = 0
    # for i in range(1):
    #     fig = plt.figure()
    #     ax = plt.axes(projection='3d')
    #     ax.set_zlim(x_range)
    #     if indx[i]:
    #         ax.plot_wireframe(grid_expanded, grid_expanded.T, out_FF.x_curves[ind_curve, :, :],
    #                           color='salmon', alpha=0.3)
    #         ind_curve += 1
    #     plt.savefig('sim_coef_est'+str(i)+'.pdf', bbox_inches="tight", transparent=True)
    #
    #     fig = plt.figure()
    #     ax = plt.axes(projection='3d')
    #     ax.set_zlim(x_range)
    #     ax.plot_wireframe(grid_expanded, grid_expanded.T, x_true[i, :, :],
    #                        alpha=0.3)
    #     plt.savefig('sim_coef_true'+str(i)+'.pdf', bbox_inches="tight", transparent=True)


    # plt.plot(my_range, rss_cv)
    # plt.show()

    # from scipy.interpolate import BSpline
    # import scipy.interpolate as intrp
    #
    # n_basis = 8
    # knots_pos = np.concatenate(([0, 0, 0], np.linspace(0, 1, n_basis - 2), [1, 1, 1]))  # because??
    # bsplines = np.zeros((grid.shape[0], n_basis))
    # for i in range(n_basis):
    #     bsplines[:, i] = intrp.BSpline(knots_pos, (np.arange(n_basis) == i).astype(float), 3,
    #                                    extrapolate=False)(grid)
    #
    # plt.plot(grid, bsplines, lw=1)
    #
    # b_basis = bsplines
    # A_s = (A @ b_basis).transpose(1, 0, 2).reshape(m, n * n_basis)
    # x_basis = b_basis
    # b_s = b @ b_basis
    # x = LA.solve(A_s.T @ A_s, A_s.T @ b_s)
    # x_curves = (x_basis @ x.reshape(n, n_basis, n_basis) @ x_basis.T) / n_basis ** 2

    # b_std = b.std(axis=0) + 1e-32
    # b = (b - b.mean(axis=0)) / b_std
    # eigvals, eigenfuns = LA.eigh(b.T @ b)
    # b_basis = eigenfuns[:, -k:]
    # x_basis = b_basis
    # A_s = (A @ b_basis).transpose(1, 0, 2).reshape(m, n * k)
    # b_s = b @ b_basis
    # x = LA.solve(A_s.T @ A_s, A_s.T @ b_s)
    # x_curves = b_std.reshape(b_std.shape[0], 1) * (x_basis @ x.reshape(n, k, k) @ x_basis.T)
    #
    # resx3 = (x_curves - x_true_sub).reshape(true_positive, neval ** 2)
    # MSEx_stdm3 = np.mean(LA.norm(resx3, axis=1) ** 2 / LA.norm(x_true_sub.reshape(true_positive, neval ** 2), axis=1) ** 2)
    #
    # print('MSEx_1 = %f' % MSEx_std2)
    # print('MSEx_new = %f' % MSEx_stdm3)
    # print('MSEx_m = %f' % MSEx_stdm2)


    # ind_curve = 0
    # for i in range(not0):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(1, 2, 1, projection='3d')
    #     ax.set_zlim(x_range)
    #     if indx[i]:
    #         ax.plot_wireframe(grid_expanded, grid_expanded.T, x_curves[ind_curve, :, :],
    #                           color='salmon', alpha=0.3)
    #         ind_curve += 1
    #
    #     ax = fig.add_subplot(1, 2, 2, projection='3d')
    #     ax.set_zlim(x_range)
    #     ax.plot_wireframe(grid_expanded, grid_expanded.T, x_true[i, :, :],
    #                       alpha=0.3)
    #     plt.show()
    #
    # for i in range(r):
    #     if pos_curves[i] > not0:
    #         fig = plt.figure()
    #         ax = fig.add_subplot(1, 2, 1, projection='3d')
    #         ax.set_zlim(x_range)
    #         ax.plot_wireframe(grid_expanded, grid_expanded.T, x_curves[i, :, :],
    #                           color='salmon', alpha=0.3)
    #         ax = fig.add_subplot(1, 2, 2, projection='3d')
    #         ax.set_zlim(x_range)
    #         plt.show()
