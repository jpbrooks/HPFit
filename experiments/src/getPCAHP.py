#==========================================================================
# February 27, 2024
# John W. Chinneck, Systems and Computer Engineering, 
#   Carleton University, Ottawa, Canada
# J. Paul Brooks, Dept. of Information Systems, 
#   Virginia Commonwealth University, Richmond, Virginia, USA
#
# Get the best-fitting hyperplane from an input dataset, using PCA.
# If there are insufficient points, default to an elastic linear
# programming solution, which uses the Gurobi LP solver.
#
# INPUTS: dataSet is the data matrix
# OUTPUTS: Hyperplane equation is: w*x = RHS.
#   w: weights in the hyperplane equation
#   RHS: right hand side constant in the hyperplane equation

import numpy as np
import gurobipy as GRB
from sklearn.decomposition import PCA

def get_pcahp(data_set):
    m, n = data_set.shape
    
    # PCA doesn't work if the number of rows is less than the number of columns,
    # so run an elastic LP solution instead.
    if m < n:
        print("  Too few observations vs. features for PCA: running elastic LP.")
        
        # Initialize Gurobi model
        model = GRB.Model("Elastic_LP")
        model.setParam("OutputFlag", 0)
        model.setParam("Threads", 14)
        
        # Define continuous variables
        # w can be negative (-Inf to Inf), while slack variables s_plus and s_minus must be >= 0
        w_vars = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="w")
        s_plus = model.addVars(m, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="s_plus")
        s_minus = model.addVars(m, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="s_minus")
        
        # Set up objective function: Minimize sum(s_plus) + sum(s_minus)
        obj_expr = s_plus.sum() + s_minus.sum()
        model.setObjective(obj_expr, GRB.MINIMIZE)
        
        # Build constraints: dataSet * w + s_plus - s_minus = n
        for i in range(m):
            row_expr = sum(data_set[i, j] * w_vars[j] for j in range(n))
            model.addConstr(row_expr + s_plus[i] - s_minus[i] == n, name=f"row_{i}")
            
        # Solve LP to find hyperplane
        model.optimize()
        
        # Extract w back into a column vector
        w = np.array([w_vars[j].X for j in range(n)]).reshape(-1, 1)
        RHS = float(n)
        return w, RHS

    # --- PCA Branch ---
    # 1. Compute column means
    my_means = np.mean(data_set, axis=0)
    
    # 2. Run PCA on the dataset
    # svd_solver='full' ensures standard, non-economy SVD is calculated
    pca_solver = PCA(svd_solver='full')
    pca_solver.fit(data_set)
    
    # 3. Extract the components
    # CRITICAL NOTE: scikit-learn stores loadings as row vectors (shape n x n),
    my_loadings_last_col = pca_solver.components_[-1, :]
    
    # Intercept of hyperplane (dot product of means and the last principal component)
    RHS = np.sum(my_means * my_loadings_last_col)
    
    # Norm vector of hyperplane reshaped into a column vector (shape: n x 1)
    w = my_loadings_last_col.reshape(-1, 1)
    
    return w, RHS
