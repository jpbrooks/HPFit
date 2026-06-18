# Algorithm 3 is proposed in Bertsimas and Mazumder (2014).  The 
# result is used as a warm start for a mixed-integer optimization
# approach.  
# algorithm3() is a composite heuristic for fitting a hyperplane to 
# minimize the q^th residual.  This version is called by mio.m. 
# To run algorithm 3 by itself, use run_alg3.m
# Purpose: to find the best fitting hyperplance to a set of input 
# data, where best fitting means that the absolute value of the 
# q^th residual is minimized.  Bertsimas and Mazumder (2014) call
# this Least Quantile of Squres (LQS) even though there is no square.
# which can be affected by outliers.
# Algorithm 3 begins by running L1 regression (LAD).  If no response
# variable is specified, we use PCA. 
# The result is
# perturbed 100 times and sent to Algorithm 2 for 500 iterations. 
# Algorithm 2 is a subdifferential-based algorithm for LQS.  The best
# solution is passed to Algorithm 1.  Algorithm 1 is a sequential 
# linear optimization algorithm for LQS.
# Algorithms 1 and 2 have been adapted to the case when no response
# variable is specified. 
#
# Main options: 
# q: 
#   - the order statistic used to evaluate the fit of a hyperplane. 
#     The fit is determined by the q^th largest residual.
# dep_var:
#   - true: a dependent variable is specified, as in ordinary
#           regression.  A residual is measured as the absolute
#           difference between the response value and the 
#           vertical projection of the point on the fitted hyperplane
#   - false: no dependent variable is specified. The intercept term 
#            is fixed to n, the original number of variables in the
#            dataset.  A residual is measured using the elastic LP
#            measure which is the absolute difference between 
#            beta^T x (including beta_0=n) and zero.
# 
# NOTES:
# - when a dependent variable is specified, the corresponding 
#   coefficient is set to -1.  When one is not specified, the 
#   intercept is set to n.
#
# INPUTS: 
# - options mentioned above
# 
# OUTPUTS:
# - beta_star: the coefficients of the best-fit hyperplane.  When 
#              dep_var=true, the first coefficient is -1 for the 
#              dependent variable, the second coefficient is the 
#              intercept, and the remaining are the coefficients for 
#              other variables.  When dep_var=false, the first 
#              coefficient is n for the intercept and the remaining
#              are the coefficients for the other variables.
# - f_beta_star: gamma, the objective function value.  The absolute
#                value of the q^th largest residual.

import numpy as np
import gurobipy as GRB

def algorithm3(X, q, dep_var, init_method):
    rg = np.default_rng(12345)
    
    m, n = X.shape  # Get the size of the dataset
    
    # Get initial LAD solution 
    beta = lad(X, dep_var, init_method) 
    
    f_beta_star = float('inf')
    nu = 2.0
    
    for k in range(100):  # Perturb LAD solution 100 times
        if dep_var:
            # Random uniform numbers in range [-nu, nu] for (n-1) elements
            perturb = rg.uniform(-nu, nu, size=(n - 1, 1))
        else:
            # Random uniform numbers in range [-nu, nu] for n elements
            perturb = rg.uniform(-nu, nu, size=(n, 1))
            
        beta = beta + (perturb * np.abs(beta))
        
        # Call Algorithm 2.  Use perturbed LAD as starting point for Alg. 2
        f_beta, beta = algorithm2(X, beta, 500, q, dep_var)
        
        # If better, update 
        if f_beta < f_beta_star:
            beta_star = beta.copy()  # Use .copy() to prevent reference binding bugs
            f_beta_star = f_beta
            
    # Use best Algorithm 2 solution as input to Algorithm 1
    beta_star, f_beta_star = algorithm1(X, beta_star, q, dep_var)
    
    # Final vector adjustment/scaling
    if dep_var:
        # With a dependent variable, prepend the -1 coefficient back onto the vector
        beta_star = np.vstack(([-1.0], beta_star))
    else:
        # Scale the solution vector based on the first element
        print(beta_star)
        beta_star = beta_star * ((n - 1) / beta_star[0, 0])
        
    return beta_star, f_beta_star


def lad(X, dep_var, init_method):
    m, n = X.shape
    
    # Initialize the Gurobi model environment
    model = GRB.Model("LAD_Regression")
    model.setParam("Threads", 14)
    model.setParam("OutputFlag", 0)  # Quiets output by default
    
    if dep_var: # Dependent variable - L1 regression
        # 1. Define variables
        # beta has length n-1 
        beta = model.addVars(n - 1, lb=-GRB.INFINITY, name="beta")
        eplus = model.addVars(m, lb=0.0, name="eplus")
        eminus = model.addVars(m, lb=0.0, name="eminus")
        
        # 2. Set Objective function: Min sum(eplus + eminus)
        model.setObjective(eplus.sum() + eminus.sum(), GRB.MINIMIZE)
        
        # 3. Add Constraints: beta^T * X[:, 1:n] + eplus - eminus = X[:, 0]
        for i in range(m):
            expr = sum(beta[j] * X[i, j + 1] for j in range(n - 1)) + eplus[i] - eminus[i]
            model.addConstr(expr == X[i, 0], name=f"eq_{i}")
            
        model.optimize()
        
        # Pull solution vector back into an (n-1, 1) numpy column vector
        beta_star = np.array([beta[j].X for j in range(n - 1)]).reshape(-1, 1)
        
    else: # No dependent variable
        if init_method == "LP": # Elastic LP
            # 1. Define variables
            beta = model.addVars(n - 1, lb=-GRB.INFINITY, name="beta")
            eplus = model.addVars(m, lb=0.0, name="eplus")
            eminus = model.addVars(m, lb=0.0, name="eminus")
            
            # 2. Set Objective function: Min sum(eplus + eminus)
            model.setObjective(eplus.sum() + eminus.sum(), GRB.MINIMIZE)
            
            # 3. Add Constraints: beta^T * X[:, 1:n] + eplus - eminus = -n
            # Check if -n or -n +1
            for i in range(m):
                expr = sum(beta[j] * X[i, j + 1] for j in range(n - 1)) + eplus[i] - eminus[i]
                model.addConstr(expr == -n, name=f"eq_lp_{i}")
                
            model.optimize()
            
            # Extract beta array and prepend scalar value 'n' matching MATLAB [n; result.x]
            beta_extracted = [beta[j].X for j in range(n - 1)]
            beta_star = np.array([n] + beta_extracted).reshape(-1, 1)
            
        else: # PCA
            weights, intercept = getPCAHP(X[:, 1:n]) 
            
            # Formulate array layout with intercept placed first matching column of 1s
            beta_star = np.vstack(([-intercept], weights.reshape(-1, 1)))
            
    return beta_star


# next: check whether RHS is -n or -n+1.
# update getPCAHP


