# FASTEN
Functional Adaptive Feature Selection with Elastic-Net penalty


FILES DESCRIPTION:
    
    fasten ------------------------------------------------------------------------------------------------------------------------------------ 

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
      
      
    expes ------------------------------------------------------------------------------------------------------------------------------------- 
    
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




THE CODE FOLLOWS A NOTATION DIFFERENT FROM THE ONE OF THE PAPER. It follows the notation of the majority of optimization sofwtares:

    m: number of observations
    n: number of features 
    k: number of elements in each group
    A: desing matrix
    b: response matrix
    x: coefficient matrix
    y: dual variable 1
    z: dual variable 2
        
        
