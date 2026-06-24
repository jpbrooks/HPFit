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
from getPCAHP import getPCAHP

def algorithm3(X, q, dep_var, init_method):
    rg = np.default_rng(12345)
    
    m, n = X.shape  # Get the size of the dataset
    # for dep_var = True, n is number of variables plus 2 (intercept, response)
    # for dep_var = False, n is number of variables plus 1 (intercept)
    
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

def solve_lp(X, beta, q, dep_var):
    m, n = X.shape
    
    # Initialize Gurobi Model
    model = GRB.Model("Solve_LP")
    model.setParam("Threads", 14)
    model.setParam("OutputFlag", 0)
    
    if dep_var:  # First variable is response
        # 1. Calculate residuals and subgradient using NumPy
        residuals = X[:, 0:1] - X[:, 1:n] @ beta
        abs_residuals = np.abs(residuals).flatten()
        
        # Sort ascending and get indices of the q largest residuals
        idx = np.argsort(abs_residuals)
        w_star = np.zeros((m, 1))
        w_star[idx[m - q : m], 0] = 1.0
        
        # Calculate subgradient vector (shape: n-1 x 1)
        subg = X[:, 1:n].T @ (-w_star * np.sign(residuals))
        
        # 2. Define LP Variables
        theta = model.addVar(lb=-GRB.INFINITY, name="theta")
        nu = model.addVars(m, lb=0.0, name="nu")
        beta_var = model.addVars(n - 1, lb=-GRB.INFINITY, name="beta")
        
        # 3. Define Objective Function
        # obj = (m - q + 1)*theta + sum(nu) - sum(subg_j * beta_j)
        obj_expr = (m - q + 1) * theta + nu.sum() - sum(subg[j, 0] * beta_var[j] for j in range(n - 1))
        model.setObjective(obj_expr, GRB.MINIMIZE)
        
        # 4. Add Constraints
        for i in range(m):
            # Projections on hyperplane: x_i^T * beta
            proj = sum(X[i, j + 1] * beta_var[j] for j in range(n - 1))
            
            # theta + nu_i >= y_i - x_i^T * beta
            model.addConstr(theta + nu[i] >= X[i, 0] - proj, name=f"upper_{i}")
            # theta + nu_i >= -y_i + x_i^T * beta
            model.addConstr(theta + nu[i] >= -X[i, 0] + proj, name=f"lower_{i}")
            
        # Objective lower bound constraint: (m - q + 1)*theta + sum(nu) - subg^T * beta >= 0
        subg_beta_term = sum(subg[j, 0] * beta_var[j] for j in range(n - 1))
        model.addConstr((m - q + 1) * theta + nu.sum() - subg_beta_term >= 0.0, name="obj_bound")
        
        model.optimize()
        
        # Extract beta coefficients safely back into a column vector
        beta_star = np.array([beta_var[j].X for j in range(n - 1)]).reshape(-1, 1)
        
    else:  # No dependent variable
        # 1. Calculate residuals and subgradient using NumPy
        residuals = X @ beta
        abs_residuals = np.abs(residuals).flatten()
        
        idx = np.argsort(abs_residuals)
        w_star = np.zeros((m, 1))
        w_star[idx[m - q : m], 0] = 1.0
        
        # Calculate subgradient vector (shape: n x 1)
        subg = X.T @ (-w_star * np.sign(residuals))
        
        # 2. Define LP Variables
        theta = model.addVar(lb=-GRB.INFINITY, name="theta")
        nu = model.addVars(m, lb=0.0, name="nu")
        beta_var = model.addVars(n, lb=-GRB.INFINITY, name="beta")
        
        # 3. Define Objective Function
        obj_expr = (m - q + 1) * theta + nu.sum() - sum(subg[j, 0] * beta_var[j] for j in range(n))
        model.setObjective(obj_expr, GRB.MINIMIZE)
        
        # 4. Add Constraints
        for i in range(m):
            proj = sum(X[i, j] * beta_var[j] for j in range(n))
            # theta + nu_i >= -x_i^T * beta
            model.addConstr(theta + nu[i] >= -proj, name=f"upper_nd_{i}")
            # theta + nu_i >= x_i^T * beta
            model.addConstr(theta + nu[i] >= proj, name=f"lower_nd_{i}")
            
        # Intercept constraint: beta_0 = n - 1
        model.addConstr(beta_var[0] == n - 1, name="intercept_fix")
        
        # Objective lower bound constraint
        subg_beta_term = sum(subg[j, 0] * beta_var[j] for j in range(n))
        model.addConstr((m - q + 1) * theta + nu.sum() - subg_beta_term >= 0.0, name="obj_bound_nd")
        
        model.optimize()
        
        # Extract beta coefficients safely back into a column vector
        beta_star = np.array([beta_var[j].X for j in range(n)]).reshape(-1, 1)
        
    return beta_star

def algorithm1(X, beta, q, dep_var):
    m, n = X.shape
    tol = 0.0001
    k = 0
    

    if dep_var:
        # Calculate initial absolute residuals (m x 1 column vector)
        residuals_k = np.abs(X[:, 0:1] - X[:, 1:n] @ beta)
        
        # Sort and extract the q-th value
        out_k = np.sort(residuals_k.flatten())
        q_residual_k = out_k[q-1]
        
        while True:
            # Optimize via Gurobi LP solver
            beta_kplus1 = solve_lp_func(X, beta, q, dep_var)
            
            # Compute new absolute residuals
            residuals_kplus1 = np.abs(X[:, 0:1] - X[:, 1:n] @ beta_kplus1)
            out_kplus1 = np.sort(residuals_kplus1.flatten())
            q_residual_kplus1 = out_kplus1[q-1]
            
            # Convergence check
            if np.abs(q_residual_k - q_residual_kplus1) <= tol * q_residual_k:
                return beta_kplus1, q_residual_k
                
            # Update values for the next iteration
            q_residual_k = q_residual_kplus1
            beta = beta_kplus1  # Update the tracking beta for successive calls if needed
            k += 1
            
    else:  # No dependent variable
        # Calculate initial absolute distance residuals
        residuals_k = np.abs(X @ beta)
        out_k = np.sort(residuals_k.flatten())
        q_residual_k = out_k[q-1]
        
        while True:
            beta_kplus1 = solve_lp_func(X, beta, q, dep_var)
            
            residuals_kplus1 = np.abs(X @ beta_kplus1)
            out_kplus1 = np.sort(residuals_kplus1.flatten())
            q_residual_kplus1 = out_kplus1[q-1]
            
            if np.abs(q_residual_k - q_residual_kplus1) <= tol * q_residual_k:
                return beta_kplus1, q_residual_k
                
            q_residual_k = q_residual_kplus1
            beta = beta_kplus1
            k += 1

def algorithm2(X, beta, MaxIter, q, dep_var):
    beta_star = beta.copy()
    m, n = X.shape
    
    if dep_var:  # If a response variable is specified as in regression
        l2_norms = np.linalg.norm(X[:, 1:n], ord=2, axis=1)
        alpha_k = 1.0 / np.max(l2_norms)
        
        # Calculate initial residuals
        residuals = X[:, 0:1] - X[:, 1:n] @ beta_star
        abs_residuals = np.abs(residuals).flatten()
        
        # Sort and get indices
        idx = np.argsort(abs_residuals)
        f_beta_star = abs_residuals[idx[q-1]]
        
        for k in range(MaxIter):
            # Target the specific sample index representing the q-th largest residual
            target_idx = idx[q-1]
            
            # Extract scalar sign and row vector layout safely
            sign_val = np.sign(residuals[target_idx, 0])
            x_row = X[target_idx, 1:n].reshape(-1, 1)  # Reshape to a column vector (shape: n-1 x 1)
            
            # Update beta: beta_kplus1 = beta - alpha_k * (-sign) * x_row
            beta_kplus1 = beta - alpha_k * (-sign_val) * x_row
            
            # Recalculate residuals and evaluate performance
            residuals = X[:, 0:1] - X[:, 1:n] @ beta_kplus1
            abs_residuals = np.abs(residuals).flatten()
            idx = np.argsort(abs_residuals)
            f_beta = abs_residuals[idx[q-1]]
            
            if f_beta < f_beta_star:
                f_beta_star = f_beta
                beta_star = beta_kplus1.copy()
                
            beta = beta_kplus1.copy()
            
    else:  # No dependent variable
        l2_norms = np.linalg.norm(X, ord=2, axis=1)
        alpha_k = 1.0 / np.max(l2_norms)
        
        residuals = X @ beta_star
        abs_residuals = np.abs(residuals).flatten()
        
        idx = np.argsort(abs_residuals)
        f_beta_star = abs_residuals[idx[q-1]]
        
        for k in range(MaxIter):
            target_idx = idx[q-1]
            
            sign_val = np.sign(residuals[target_idx, 0])
            x_row = X[target_idx, :].reshape(-1, 1)  # Reshape to a column vector (shape: n x 1)
            
            beta_kplus1 = beta - alpha_k * (-sign_val) * x_row
            
            residuals = X @ beta_kplus1
            abs_residuals = np.abs(residuals).flatten()
            idx = np.argsort(abs_residuals)
            f_beta = abs_residuals[idx[q-1]]
            
            if f_beta < f_beta_star:
                f_beta_star = f_beta
                beta_star = beta_kplus1.copy()
                
            beta = beta_kplus1.copy()
            
    return f_beta_star, beta_star

# update getPCAHP
# check RHS = -n


