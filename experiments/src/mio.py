# mio() uses MIP to fit a hyperplane that minimizes the q^th residual.
# Purpose: to find the best fitting hyperplance to a set of input 
# data, where best fitting means that the q^th residual is minimized.
# which can be affected by outliers.
# Two general formulations are presented.  One is based on the 
# original presented in Bertsimas and Mazumder 2014 (2.11). The other
# is an equivalent but more compact formulation developed by JC. For 
# each formulation, there are two versions: 1) a dependent variable is
# specified (like the original published version) and 2) no dependent
# variable is specified.  
# mio() is called by run_mio.R and run_cbqmio.m.
# The MIP solver is run for the timelimit in seconds minus the warm 
# start time and
# the best solution is reported.  
#
# Main versions and options:
# formulation:
#   - mio-bm: the original MIP formulation presented in (2.11) of 
#             Bertsimas and Mazumder (2014).  
#   - mio1: an equivalent but more compact formualtion developed by
#           JC.  
#   - lqs-mio-bm: mio-bm warmstarted with LQS.  LQS is run in R.
#   - lqs-mio1: mio1 warmstarted with LQS.  LQS is run in R.
#   - alg3-mio-bm: mio-bm warmstarted with Algorithm 3 from Bertsimas
#                  and Mazumder (2014).  mio() calls algorithm3.m.
#   - alg3-mio1: mio1 warmstarted with Algorithm 3 from Bertsimas
#                and Mazumder (2014).  mio() calls algorithm3.m.
#   - cbq-mio-bm: mio-bm warmstarted with CBq created by JC.  mio()
#                 is called by run_cbqmio.m.
#   - cbq-mio1: mio1 warmstarted with CBq created by JC. mio() is 
#               is called by run_cbqmio.m.
#   - mio-bm-first: take the first feasible solution for mio-bm.
#   - mio1-first: take the first feasible solution for mio1.
# dep_var:
#   - true: a dependent variable is specified, as in ordinary 
#           regression.  A residual is measured as the absolute 
#           difference between the response value and the 
#           vertical projection of the point on the fitted hyperplane
#   - false: no dependent is specified. The intercept term is 
#            fixed to n, the original number of variables in the
#            dataset.  A residual is measured as the  
#            distance of a point to its orthogonal projection.
# q: 
#   - the order statistic used to evaluate the fit of a hyperplane. 
#     The fit is determined by the q^th largest residual.
# NOTES:
# - when a dependent variable is specified, the corresponding 
#   coefficient is set to -1.  When one is not specified, the 
#   intercept is set to n.
# - a result datafile is created containing the data file with 
#     - full path, 
#     - iteration number, 
#     - number of rows, 
#     - number of variables, 
#     - number of non-outliers, 
#     - q, 
#     - formulation, 
#     - total squared error, 
#     - MIP runtime (timelimit), 
#     - MIP status, 
#     - gamma, 
#     - best MIP bound, 
#     - number of outliers identified as one of the q smallest by 
#       best MIP feasible solution,
#     - trimmed squared error (TSE) for the q smallest residuals, TSE
#       for m_normal residuals after 1 hour, 
#     - time used by the warm start heuristic,
#     - gamma obtained by using the beta from the warm start, 
#     - TSE for m_normal residuals for warm start, 
#     - TSE for m_normal residuals after, gamma obtained 
#       after 3600s, 
#     - TSE for m_normal after 3600s.
# 
# INPUTS:
# - options mentioned above
# - iteration: iteration number.  Used in the output filename.
# - datafname: full path to data file.
# - lqs_beta: an initial solution generated using LQS or CBq as implemented
#             in R or CBq.  Only used for lqs-mio-bm, lqs-mio1, cbq-mio1,
#             cbq-mio-bm.
# - m_normal: number of non-outlier rows of data.  After that they are
#             outliers.
# - resloc: path to folder where output file will reside.
# - timelimit: time recorded by R for LQS or CBq as a triplet if they are 
#              used for a warmstart. We are using the third number in the 
#              triplet.  The time is subtracted from the time allowed for 
#              the MIO.
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

import gurobipy as gp
from gurobipy import GRB
import scipy.sparse as sp
import numpy as np
import pandas as pd
from algorithm3 import algorithm3

def mio(iteration, datafname,q, lqs_beta, m_normal, dep_var, formulation, resloc, timelimit):

    X_df = pd.read_csv(datafname)  # Use pd.read_excel if reading .xlsx
    X = X_df.to_numpy()  # Extract raw data from the dataframe

    m, n = X.shape

    if dep_var:
        # In Python, indexing is 0-based:
        # X[:, 0] is the first column (response)
        # X[:, 1:n] are the remaining columns
        # We insert a column of ones (intercept) at index 1
        ones_col = np.ones((m, 1))
        X = np.hstack([X[:, [0]], ones_col, X[:, 1:n]])
    else:
        # Prepend a column of ones to the beginning of the matrix
        ones_col = np.ones((m, 1))
        X = np.hstack([ones_col, X])
    
    # Update dimensions to reflect the new intercept column
    m, n = X.shape

    # Set up the MIP
    #
    # Set up the callback for calculating the primal integral

    time_history = [0.0]
    obj_history = [float('inf')]

    def pi_callback(model, where):
        # 4 corresponds to GRB.Callback.MIPSOL
        if where == GRB.Callback.MIPSOL:
            # Retrieve values using the Gurobi callback constants
            runtime = model.cbGet(GRB.Callback.RUNTIME)
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            time_history.append(runtime)
            obj_history.append(obj)
    bm_formulations = {
       "alg3-mio-bm",
       "lqs-mio-bm",
       "mio-bm-first",
       "cbq-mio-bm",
       "mio-bm"
    }
    model = gp.Model("mio")
    model.ModelSense = GRB.MINIMIZE

    # Decision variables
    gamma = model.addVar(
        lb=0.0,
        obj=1.0,
        vtype=GRB.CONTINUOUS,
        name="gamma",
    )

    z = model.addVars(
        m,
        obj=0.0,
        vtype=GRB.BINARY,
        name="z",
    )

    beta = model.addVars(
        n,
        lb=-GRB.INFINITY,
        obj=0.0,
        vtype=GRB.CONTINUOUS,
        name="beta",
    )
    if formulation in bm_formulations:

        rplus = model.addVars(
            m,
            lb=0.0,
            obj=0.0,
            vtype=GRB.CONTINUOUS,
            name="rplus",
        )

        rminus = model.addVars(
            m,
            lb=0.0,
            obj=0.0,
            vtype=GRB.CONTINUOUS,
            name="rminus",
        )

        mu = model.addVars(
            m,
            lb=0.0,
            obj=0.0,
            vtype=GRB.CONTINUOUS,
            name="mu",
        )

        mubar = model.addVars(
            m,
            lb=0.0,
            obj=0.0,
            vtype=GRB.CONTINUOUS,
            name="mubar",
        )

        # rplus_i + rminus_i - gamma = mubar_i - mu_i
        # Equivalent:
        # -gamma + rplus_i + rminus_i + mu_i - mubar_i = 0
        model.addConstrs(
            (
                -gamma + rplus[i] + rminus[i] + mu[i] - mubar[i] == 0
                for i in range(m)
            ),
            name="abs_residual_ordering",
        )

        # rplus_i - rminus_i = y_i - x_i^T beta
        #
        # In the MATLAB matrix, this is written as:
        # rplus - rminus + X beta = 0
        #
        # because beta_1 is later fixed to -1 and X includes y in column 1.
        model.addConstrs(
            (
                rplus[i]
                - rminus[i]
                + gp.quicksum(X[i, j] * beta[j] for j in range(n))
                == 0
                for i in range(m)
            ),
            name="residual_definition",
        )

        # sum_i z_i = q
        model.addConstr(
            gp.quicksum(z[i] for i in range(m)) == q,
            name="sum_z",
        )

        # mu_i <= gamma
        # MATLAB row: gamma - mu_i <= 0
        model.addConstrs(
            (
                gamma - mu[i] >= 0
                for i in range(m)
            ),
            name="mu_le_gamma",
        )
        if dep_var:
            # beta_0 = -1 
            model.addConstr(
                beta[0] == -1.0,
                name="fix_response_coefficient",
            )
        else:
            # beta_0 = n - 1 
            model.addConstr(
                beta[0] == n - 1,
                name="fix_intercept_coefficient",
            )
    else: # MIO1 formulation
        r = model.addVars(
            m,
            lb=0.0,
            obj=0.0,
            vtype=GRB.CONTINUOUS,
            name="r",
        )

        eplus = model.addVars(
            m,
            lb=0.0,
            obj=0.0,
            vtype=GRB.CONTINUOUS,
            name="eplus",
        )

        eminus = model.addVars(
            m,
            lb=0.0,
            obj=0.0,
            vtype=GRB.CONTINUOUS,
            name="eminus",
        )
         # beta^T x + eplus - eminus = 0
        #
        # In dep_var == true, X already includes the response column,
        # and beta[0] will be fixed to -1, so this represents:
        #     beta^T x - y + eplus - eminus = 0
        #
        # In dep_var == false, this represents:
        #     beta^T x + eplus - eminus = 0
        model.addConstrs(
            (
                gp.quicksum(X[i, j] * beta[j] for j in range(n))
                + eplus[i]
                - eminus[i]
                == 0
                for i in range(m)
            ),
            name="residual_definition",
        )

        # eplus_i + eminus_i - r_i - gamma <= 0
        model.addConstrs(
            (
                eplus[i] + eminus[i] - r[i] - gamma <= 0
                for i in range(m)
            ),
            name="residual_bound",
        )

        # sum_i z_i = q
        model.addConstr(
            gp.quicksum(z[i] for i in range(m)) == q,
            name="sum_z",
        )

        # beta normalization
        if dep_var:
            model.addConstr(
                beta[0] == -1.0,
                name="fix_response_coefficient",
            )
        else:
            model.addConstr(
                beta[0] == n - 1,
                name="fix_intercept_coefficient",
            )
    if formulation in bm_formulations:
        for i in range(m):
            # SOS1: mubar_i and mu_i
            model.addSOS(
                GRB.SOS_TYPE1,
                [mubar[i], mu[i]],
            )

            # SOS1: rplus_i and rminus_i
            model.addSOS(
                GRB.SOS_TYPE1,
                [rplus[i], rminus[i]],
            )

            # SOS1: z_i and mubar_i
            # This is the correction from the paper.
            model.addSOS(
                GRB.SOS_TYPE1,
                [z[i], mubar[i]],
            )
    else:
        for i in range(m):
            # SOS1: r_i and z_i
            model.addSOS(
                GRB.SOS_TYPE1,
                [r[i], z[i]],
            )

  # Next: call to alg3
if formulation in ["alg3-mio-bm", "alg3-mio1"]:
    print("alg 3 start")
    t_start = time.time()                                                    
    beta_start, f_beta1 = algorithm3(X, q, dep_var, "PCA") 
    alg3_time = time.time() - t_start
    print(f"alg3_time = {alg3_time}")
    timelimit = timelimit - alg3_time 
    print(f"f_beta1 = {f_beta1}") 
    print("alg 3 end")


if formulation in ["lqs-mio-bm", "lqs-mio1", "cbq-mio1", "cbq-mio-bm"]:
    beta_start = lqs_beta  # Assign the warmstart vector

import numpy as np

newGammaHeur = -1.0 
tsestarHeur = -1.0 
newGamma = -1.0 
tsestar = -1.0 

# make sure beta_start is 2D
if beta_start.ndim == 1:
    beta_start = beta_start.reshape(-1,1)

# Calculate values for all variables based on the warm start gamma   
if formulation in ["alg3-mio-bm", "lqs-mio-bm", "cbq-mio-bm"]: # MIO-BM          
    
    dist = X @ beta_start # Matrix multiplication in Python uses @
    absdist = np.abs(dist)                                                                                                    
    # sortedabsdist tracking column indices: np.arange(m) generates 0 to m-1
    # We reshape to column vector and hstack with absdist
    indices = np.arange(m).reshape(-1, 1)
    sortedabsdist = np.hstack([indices, absdist])                                                                             
    # Sort by the second column (index 1)
    sortedabsdist = sortedabsdist[sortedabsdist[:, 1].argsort()]
    
    # MATLAB: sortedabsdist(q, 2) -> Python: index is q-1 due to 0-based indexing
    newGammaHeur = sortedabsdist[q - 1, 1]  # gamma from warmstart
    gamma.Start = newGammaHeur
    print(f"Gamma recalculated from beta_start is {newGammaHeur}")
    
    # Calculate rminus and rplus variable values
    for i in range(m):  
        if dist[i, 0] > 0:
            rminus[i].Start = dist[i, 0]
            rplus[i].Start = 0.0
        else:
            rplus[i].Start = -dist[i, 0]
            rminus[i].Start = 0.0
            
        # Calculate mubar and mu values                                        
        if absdist[i,0] - newGammaHeur > 0:
            mubar[i].Start = absdist[i,0] - newGammaHeur
            mu[i].Start = 0.0
        else:
            mu[i].Start = newGammaHeur - absdist[i,0]
            mubar[i].Start = 0.0
        z[i].Start = 0.0
        
    # Identify which variables are selected as inliers
    for i in range(q):
        # Cast the stored matrix index back to an integer to use as an array index
        row_idx = int(sortedabsdist[i, 0])
        z[row_idx].Start = 1.0 

    for j in range(n):
        beta[j].Start = beta_start[j,0]
    # Warm start values assigned.
    if dep_var: # Get error along response direction (beta_1 = -1)
        tot_err = np.sum(dist[0:m_normal, 0] * dist[0:m_normal, 0])
        sorteddist = np.sort(dist[:, 0] * dist[:, 0])
        
        tsestarHeur = np.sum(sorteddist[0:m_normal]) # TSEstar based on m_normal
        tse = np.sum(sorteddist[0:q]) # TSE based on q
    else: # Get orthogonal error
        # Exclude the first element (intercept) from norm calculation: index 1 to end
        gradLen = np.linalg.norm(beta_start[1:n, 0]) 
        edist = np.abs(dist / gradLen)
        sortededist = np.sort(edist[:, 0] * edist[:, 0])
        
        tsestarHeur = np.sum(sortededist[0:m_normal]) 
        tse = np.sum(sortededist[0:q]) 
        tot_err = np.sum(edist[0:m_normal, 0] * edist[0:m_normal, 0])
elif formulation in ["alg3-mio1", "lqs-mio1", "cbq-mio1"]: # MIO1
    dist = X @ beta_start
    absdist = np.abs(dist)
    indices = np.arange(m).reshape(-1, 1)
    sortedabsdist = np.hstack([indices, absdist])                                                                             
    sortedabsdist = sortedabsdist[sortedabsdist[:, 1].argsort()]
    newGammaHeur = sortedabsdist[q - 1, 1]  # gamma from warmstart
    gamma.Start = newGammaHeur
    print(f"Gamma recalculated from beta_start is {newGammaHeur}")
    for i in range(m):
        if dist[i,0] > 0:
            eminus[i].Start = dist[i,0]
            eplus[i].Start = 0.0
        else:
            eplus[i].Start = -dist[i,0]
            eminus[i].Start = 0.0
        z[i].Start = 0.0
        rel[i].Start = absdist[i,0]
     
    for i in range(q):
        row_idx = int(sortedabsdist[i,0])
        z[row_idx].Start = 1.0
        rel[row_idx].Start = 0.0
    for j in range(n):
        beta[j].Start = beta_start[j,0]
    # Warm start values assigned.
    if dep_var: # Get error along response direction (beta_1 = -1)
        tot_err = np.sum(dist[0:m_normal, 0] * dist[0:m_normal, 0])
        sorteddist = np.sort(dist[:, 0] * dist[:, 0])
        
        tsestarHeur = np.sum(sorteddist[0:m_normal]) # TSEstar based on m_normal
        tse = np.sum(sorteddist[0:q]) # TSE based on q
    else: # Get orthogonal error
        # Exclude the first element (intercept) from norm calculation: index 1 to end
        gradLen = np.linalg.norm(beta_start[1:n, 0]) 
        edist = np.abs(dist / gradLen)
        sortededist = np.sort(edist[:, 0] * edist[:, 0])
        
        tsestarHeur = np.sum(sortededist[0:m_normal]) 
        tse = np.sum(sortededist[0:q]) 
        tot_err = np.sum(edist[0:m_normal, 0] * edist[0:m_normal, 0])

if 'first' in formulation:
    model.setParam('SolutionLimit', 1)

if timelimit >= 0.0:
    model.setParam('TimeLimit', timelimit)
    model.setParam('Symmetry', 2)
    model.setParam('Threads', 14)

    model.optimize(pi_callback)

    status = model.Status
    print(f"Optimization Status: {status}")

    time_history.append(model.Runtime)
    if model.SolCount > 0:
        obj_history.append(model.ObjVal)
    else:
        obj_history.append(float('inf'))

    beta_values = model.getAttr('X', beta).values()
    z_values = model.getAttr('X', z).values()

    beta_star = np.array(list(beta_values)).reshape(-1,1)
    z_star = np.array(list(z_values)).reshape(-1,1)
    
    f_beta_star = model.ObjVal

    if 'first' in formulation:
        output_runtime = model.Runtime

    num_outliers_in_q = np.sum(z_values[m_normal:m])
    print(f"num_outliers_in_q = {num_outliers_in_q}")

    # get sum of squared error on non outliers
    dist = np.abs(X @ beta_star)
    if dep_var: # get error along response direction, recall that beta_1 = -1
        tot_err = np.sum(dist[0:m_normal,0] * dist(0:m_normal,0])
        sorteddist = np.sort(dist[:,0]*dist[:,0])
        tsestar = np.sum(sorteddist[0:m_normal]) # TSE* is based on m_normal, which we usually don't know
        tse = np.sum(sorteddist[0:q]) # TSE is based on q, which we usually set to 0.5
    else: # get orthogonal error
        gradLen = np.linalg.norm(beta_star[1:n,0])
        edist = np.abs(dist/gradLen)
        sorteddist = np.sort(edist[:,0]*edist[:,0])
        tsestar = np.sum(sorteddist[0:m_normal]) #TSE* is based on m_normal, which we usually don't know
        tse = np.sum(sorteddist[0:q]) # TSE is based on q, which we usually set to 0.5
        tot_err = np.sum(edist[0:m_normal,0]*edist[0:m_normal,0])
    if formulation in bm_formulations: # MIO-BM
        dist = X @ beta_star
        absdist = np.abs(dist)
        indices = np.arange(m).reshape(-1,1)
        sortedabsdist = np.hstack([indices, absdist])
        sortedabsdist = sortedabsdist[sortedabsdist[:,1].argsort()]
        newGamma = sorted absdist[q-1,1]
        gamma.Start = newGamma
        print(f"Gamma recalculated from beta_start is {newGamma}")
        for i in range(m):
            if dist[i,0] > 0:
                rminus[i].Start = dist[i,0]
                rplus[i].Start = 0.0
            else:
                rplus[i].Start = -dist[i,0]
            if absdist[i,0] - newGamma > 0.0:
                mubar[i].Start = absdist[i,0] - newGamma
                mu[i].Start = 0.0
            else:
                mu[i].Start = newGamma - absdist[i,0]
                mubar[i].Start = 0.0
            z[i].Start = 0.0
        for i in range(q):
            row_idx = int(sortedabsdist[i,0])
            z[row_idx].Start = 1.0
        for j in range(n):
            beta[j].Start = beta_star[j,0]
    elif
    # line 376






return beta_star, f_beta_star


