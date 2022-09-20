# FASTEN
Functional Adaptive Feature Selection with Elastic-Net penalty


FILES DESCRIPTION:
    
fasten 

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
      
      
    expes 
    
      expes/main_core.py:
        main file to run fgen_core and competitor solvers on synthetic data 

      expes/main_path.py:
        main file to run fgen_path and competitor solvers on synthetic data 
