# FASTEN
Functional Adaptive Feature Selection with Elastic-Net penalty

The repo contains the code to perform functional feature selection usinf a Dual Augmented Lagrangian (DAL) algorithm. The methods are presented in the following papers: REMEMBER TO ADD PAPERS 


FILES DESCRIPTION:
    
    fasten ---------------------------------------------------------------------------------------------------------------------------

        fasten/solver_path.py:
          class to run FASTEN

        fasten/solver_FF.py:
          class to solve the DAL problem for the Function-on-Function (FF) problem. 

        fasten/solver_FS.py:
          class to solve the DAL problem for the Function-on-Scalar (FS) problem. 

        fasten/solver_FC.py:
          class to solve the DAL problem for the Functional Concurrent (FC) and Scalar-on-Function (SF) problem. 

        fasten/solver_FC.py:
          class to solve the DAL problem for the Functional Concurrent problem. 

        fasten/auxiliary_functions.py
          auxiliary functions' classes called by the different solver, including proximal operator functions and conjugate functions.

        fasten/generate_sim.py
          classes to generate syntehtic data for FF, FS, FC and SF. For FF it is also possible to generate a test and a train data. 
      
      
    expes ---------------------------------------------------------------------------------------------------------------------------- 
    
      expes/sim_FF.py:
        main file to run FASTEN on synthetic data for the FF model 

      expes/sim_FS.py:
        main file to run FASTEN on synthetic data for the FS model
        
      expes/sim_FC.py:
        main file to run FASTEN on synthetic data for the FC model
        
      expes/sim_SF.py:
        main file to run FASTEN on synthetic data for the SF model
        



THE FOLLOWING PYTHON PACKAGES ARE REQUIRED:
  
    - numpy
    - Scikit-learn
    - scipy
    - tqdm
    - matplotlib
    - pandas



TO RUN THE CODE: 

    1) open a python3.10 environment
    2) Install the package by running `pip install -e .` at the root of the repository, i.e. where the setup.py file is.
    3) Lunch the desired experiments, e.g. `python expes/sim_FF.py`    
    
    !) for Apple Mx processors' users, it is suggested to manually install 'numpy' for achieving better performance. See:  
       https://gist.github.com/MarkDana/a9481b8134cf38a556cf23e1e815dafb 
    



THE CODE FOLLOWS A NOTATION DIFFERENT FROM THE ONE OF THE PAPER. It follows the notation of the majority of optimization sofwtares:

    m: number of observations
    n: number of features 
    k: number of elements in each group
    A: desing matrix
    b: response matrix
    x: coefficient matrix
    y: dual variable 1
    z: dual variable 2
        
        
 FASTEN.solver PARAMETERS DESCRIPTION: 
    
    INPUT PARAMETERS:
    ----------------------------------------------------------------------------------------------------------------------------------
    :param regression_type: RegressionType object
        FS: function on scalar
        SF: scalar on function
        FC: concurrent model
        FF: function on function
        
    :param A: design matrix
        FS: np.array((m, n))
        FF, SF, FC: np.array((n, m, neval))
        
    :param b: response matrix/vector
        FS, FF, FC: np.array((m, neval))
        SF: np.array(m, )
        
    :param k: number of basis function. If k is not passed to the function, it is selected automatically such that:
        FF: more than 90% of response variability
        FC: more than 95% of response variability
        FS: min(5, more than 99% of response variability)
        SF: k = 5
        
    :param wgts: individual weights for the penalty. 1 (default) or np.array with shape (n, 1)
    
    :param selection_criterion: an object of class SelectionCriteria, it can be CV, GCV, EBIC.
        The output of the fasten will contain the best model according to the chosen criterion.
        We recommend to use CV for FC model
        
    :param n_folds: if selection_criterion is CV, number of folds to compute it. Default = 10
    
    :param adaptive_scheme: an object of class AdaptiveScheme. It can be NONE, SOFT, FULL.
        NONE: no adaptive step is performed
        SOFT (default): just one adaptive iteration is performed
        FULL: a new path is investigated starting from the weights obtained at the previous path
        
    :param coefficients_form: If TRUE the inputs A and b are already in the coefficients form and x_basis MUST be given.
        Deafult is False. The coefficient form has be obtained as follows:
        (remember, if g_basis and f_basis orthogonal, then g_basis.T @ f_basis = I)
            For b - with b_scores = b @ b_basis, we have:
                FS: b_coeff = b_scores @ b_basis.T @ x_basis
                SF: b_coeff = b
                FC: b_coeff = integral(b)
                FF: b_coeff = b_scores @ b_basis.T @ x_basis2
            For A - with A_scores = A @ A_basis, we have:
                FS: A_coeff = A
                SF: A_coeff = A_scores @ A_basis.T @ x_basis
                FC: A_coeff = A_scores @ A_basis.T @ x_basis
                FF: A_coeff = A_scores @ A_basis.T @ x_basis1
        If FALSE the coefficients form is automatically computed using the following basis
            FS: b_basis = x_basis = FPC of b
            SF: A_basis (all feat) = x_basis (all feat) = FPC of the first features of A
            FC: A_basis = x_basis = FPC of b
            FF: A_basis = x_basis1 = x_basis2 = FPC of b
            
    :param x_basis: Default is False. if coefficient_form = TRUE, you have to pass the basis function of x.
        If you use the same basis for all the features then:
            x_basis: (neval x k).
        If you use different basis for each feature, then:
            function-on-function: x_basis is a (2, n, neval, k) tensor with:
                first dimension: x_basis1 and x_basis2, second dimension: basis of each features, and it has to be:
                x_basis1 = A_basis, x_basis2 = b_basis
            All other models: x_basis is an (n, neval, k) tensor
            
    :param c_lam_vec: np.array to determine the path of lambdas. Default: np.geomspace(1, 0.01, num=100)
        If just one number, a single run is performed and the output is in best_models.single_run
        Different regression model and different alpha, may requires longer/shorter grid. We reccomend the
        user to investigate a long grid and maybe use max_selected to stop the search.
        
    :param c_lam_vec_adaptive: np.array to determine the path of lambdas in the adaptive step.
        Used if adaptive_scheme = FULL. DEfault: np.geomspace(1, 0.0001, num=50)
        
    :param max_selected: if given, the algorithm stops when a number of features > max_selected is selected
        Default is None
        
    :param check_selection_criterion: if True and the selection criterion has  a strong discontinuity,
        we stop the search. If max selected is None or bigger than 80, we suggest to set
        check_selection_criterion = True. Default is False.
        
    :param alpha: we have lam1 = alpha * c_lam * lam1_max, lam2 = (1 - alpha) * c_lam * lam1_max
        We recoomend to use alpha = 0.5 for the FC model. Default is 0.2
        
    :param lam1_max: smallest values of lam1 that selects 0 features. If it is None, it is computed inside the function
    
    :param x0: initial value for the variable of the primal problem -- vector 0 if not given
        FF: np.array((n * k, k))
        all the others: np.array((n, k))
        
    :param y0: initial value fot the first variable of the dual problem -- vector of 0 if not given
        FF, FS: np.array((m, k))
        FC, SF: np.array((m))
        
    :param z0: initial value for the second variable of the dual problem -- vector of 0 if not given
        FF: np.array((n * k, k))
        all the others: np.array((n, k))
        
    :param Aty0: A.T @ y0
    
    :param select_k_estimation: used just if regression_type = FF. Default is True.
        If true, k can change after the selection and before the surfaces' estimation (chosen based on CV)
        
    :param relaxed_criteria: if True a linear regression is fitted on the selected features before computing
        the selection criterion. Default is True
        
    :param relaxed_estimates: if True a linear regression is fitted on the features to produce the final estimates.
        Default is True. We suggest to set relaxed_criteria = relaxed_estimates = True
        If adaptive_scheme = FULL, relaxed_estimates and relaxed_criteria are forced to be False
        (the weights already are a relaxation of the estimates)
        
    :param sgm: starting value of the augmented lagrangian parameter sigma. Default is 5e-3
    
    :param sgm_increase: increasing factor of sigma.  Default is 5.
    
    :param sgm_change: we increase sgm -- sgm *= sgm_increase -- every sgm_change iterations. Default is 1
    
    :param tol_nwt: tolerance for the nwt algorithm. Default is 1e-6
    
    :param tol_dal: global tolerance of the dal algorithm. Default is 1e-6
    
    :param maxiter_nwt: maximum number of iterations for nwt. Default is 50
    
    :param maxiter_dal: maximum number of global iterations. Default is 100
    
    :param use_cg: True/False. If true, the conjugate gradient method is used to find the direction of the nwt
        Dfault is False
        
    :param r_exact: number of features such that we start using the exact method. Default is 2e4
    
    :param plot: True/False. If true a plot of r, gcv, extended bic and cv (if cv == True) is displayed
    
    :param print_lev: different level of printing (0, 1, 2, 3, 4)
    ----------------------------------------------------------------------------------------------------------------------------------


    OUTPUT: OutputPath object with the following attributes
    ----------------------------------------------------------------------------------------------------------------------------------
    
    :attribute best_model: 
        an OutputSolver object containing the best model according to the chosen selection criterion. It has the following attributes: 
            --------------------------------------------------------------------------------------------------------------------------
            :attribute x_curves: curves computed just for the not 0 estimated coefficients
                FF: x_basis1 @ x_scores @ x_basis2.T, np.array((r, neval, neval))
                FS, FC, SF: x_curves = x_scores @ x_basis.T,  np.array((r, neval))
                They are returned as None by this function then and computed for the best models in path_solver
            :attribute x_coeffs: standardized estimated coefficients. They are estimated based on: (b - b.mean) / b.std()
                FS, FC, SF: np.array((n, k))
                FF: np.array((n, k, k))
            :attribute x_basis: they are returned as None by this function then inserted in path_solver
            :attribute b_coeffs: coefficient form of b
                FS, FF, FC: np.array((m, k))
                SF: np.array(m, ), same as b, but standardized
            :attribute A_coeffs: coefficient form of A
                FS: np.array((m, n))
                FF, SF, FC: np.array((n, m, k))
            :attribute y: optimal value of the first dual variable
            :attribute z: optimal value of the second dual variable
            :attribute r: number of selected features (after adaptive step if it is performed)
            :attribute r_no_adaptive: number of selected features before adaptive step. None if adaptive is not performed
            :attribute indx: position of the selected features
            :attribute selection_criterion_value: value of the chosen selected criterion
            :attribute sgm: final value of the augmented lagrangian parameter sigma
            :attribute c_lam: specifc c_lam value used for the returned model
            :attribute alpha: same as input
            :attribute lam1_max: same as input
            :attribute lam1: specifc lasso penalization value used for the returned model 
            :attribute lam2: specifc ridge penalization value used for the returned model 
            :attribute time: total time of dal
            :attribute iters: total dal's iteration
            :attribute Aty: np.dot(A.T(), y) computed at the optimal y. Useful to implement warmstart
            :attribute convergence: True/False. If false the algorithm has not converged
            --------------------------------------------------------------------------------------------------------------------------
            
    :attribute k_selection: k used fot the feature selection
    
    :attribute k_estimation: k used fot the feature estimation. It can be different from k_selection, only in the FF
        regression model if select_k_estimation = TRUE
        
    :attribute r_vec: np.array, number of selected features for each value of c_lam
    
    :attribute selection_criterion_vec: np.array, value of the selection criterion for each value of c_lam
    
    :attribute c_lam_entry_value: np.array, contains the c_lam value for which each selected feature entered the model
    
    :attribute c_lam_vec: np.array, vector containing all the values of c_lam
    
    :attribute alpha: same as input
    
    :attribute lam1_vec: np.array, lasso penalization for each value of c_lam
    
    :attribute lam2_vec: np.array, ridge penalization for each value of c_lam
    
    :attribute lam1_max: same as input
    
    :attribute time_total: total time of FAStEN
    
    :attribute time_path: time to compute the solution path
    
    :return time_cv: time to perform cross validation
    
    :return time_adaptive: time to perform the adaptive step
    
    :return time_curves: total times to compute the final estimated curves/surfaces from the basis coefficients
    
    :return iters_vec: array, iteration to converge for each value of c_lam
    
    :return times_vec: array, time to compute the solution for each value of c_lam

