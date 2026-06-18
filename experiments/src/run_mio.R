# The main function is run_mio() that calls one of two methods in
# MATLAB for minimizing the q^th absolute residual.  The methods are:
# MIO-BM from Bertsimas and Mazumder (2014), MIO1, an improved
# formulation developed by JWC.  Supporting functions are 
# hyper_dist_sq() for calculating the distance to a hyperplane and
# fit_lqs() which fits a hyperplane using a specified variable as 
# the dependent variable.
# MIO-BM and MIO1 are implemented in mio.m.  
#
# Main version and options:
# dep_var:
#   - TRUE: a dependent variable is specified, as in ordinary 
#           regression.  A residual is measured as the absolute 
#           difference between the response value and the 
#           vertical projection of the point on the fitted hyperplane
#   - FALSE: no dependent is specified. The intercept term is 
#            fixed to n, the original number of variables in the
#            dataset.  In the end, a residual is measured as the  
#            distance of a point to its orthogonal projection.
# q: the percentile used to evaluate the fit of a hyperplane in
#     in the MIO1 and MIO-BM formulations
# formulation:
#   - mio-bm: use the model proposed by Bertsimas and Mazumder (2014)
#   - mio1: use JWC's compact MIO model.
#   - alg3-mio-bm: mio-bm but with Algorithm 3 of Bertsimas and 
#                  Mazumder (2014) to generate an initial solution
#   - alg3-mio1: mio1 but with Algorithm 3 of Bertsimas and 
#                Mazumder (2014) to generate an initial solution
#   - lqs-mio-bm: mio-bm but with lqs in R to generate an initial 
#                 solution
#   - lqs-mio1: mio1 but with lqs in R to generate an initial 
#               solution
#   - cbq-mio-bm: mio-bm but with CBq to generate an initial 
#                 solution
#   - cbq-mio1: mio1 but with CBq in R to generate an initial 
#               solution
#   - mio-bm-first: take the first feasible solution discovered using 
#                   mio-bm 
#   - mio1-first: take the first feasible solution discovered using 
#                   mio1 
 
#
# calc_lqs_beta: 
#   - TRUE: use R's implementation of LQS to generate a warmstart
#           solution for MIO.  If dep_var=TRUE, allow each variable
#           to serve as the response in turn.
#   - FALSE: do not generate a warmstart using LQS.
#
# NOTES:
# - when dep_var=TRUE, the response coefficient is -1.  When one is
#   not specified, the coefficients are normalized so that the 
#   intercept is n (beta_0 + beta^T x = 0).
# - mio.m create output files
#
# INPUTS:
# - options mentioned above
# - dataloc: folder containing data file
# - srcloc: folder containing mio.m
# - fname: data file name
# - timelimit: timelimit for the MIP solver
# - resloc: folder for the output 
# - mosekloc: location of MOSEK files ($HOME/src/mosek/9.3/toolbox/r2015a)
# - gurobiloc: location of Gurobi Matlab setup file ($HOME/opt/gurobi1100/linux64/matlab)


# function for calculating the squared distance of a point to a
# hyperplane
hyper_dist_sq <- function(w,b,x) { 
  (sum(w*x) + b)^2/sum(w*w)
}


# function for applying LQS to a dataset with the j^th column 
# serving as the response.
fit_lqs <- function(j, X, m, n, q) {
  cat(j, " ")
  my_formula <- as.formula(paste("V", j, " ~ .", sep="")) # Vj is the response; all other variables are predictors
  if (q==1.0) {
    q=(nrow(X)-1)/nrow(X) 
  }
  my_lqs <- lqs(my_formula, data=X, method="lqs", quantile=floor(q*nrow(X)))  # Fit LQS model
  lqs_norm <- my_lqs$coefficients[2:length(my_lqs$coefficients)] # Get coefficients excluding the intercept
  lqs_norm[is.na(lqs_norm)] <- 0.0
  lqs_norm <- append(lqs_norm, -1.0, after=j-1) # The coefficient for the response is -1
  lqs_intercept <- my_lqs$coefficients[1] # get the intercept
  lqs_res <- apply(X,1,hyper_dist_sq,w=lqs_norm,b=lqs_intercept) # sum of squared distances for all points
  lqs_rq <- quantile(lqs_res, q) # get the q^th percentile residual
  lqs_dist <- lqs_rq
  #lqs_dist <- sum(apply(X[1:m,],
  #                     1,
  #                     hyper_dist_sq,
  #                     w=lqs_norm,
  #                     b=lqs_intercept)) # sum of squared distances to the hyperplane for non-outliers
  return(
         list(
              lqs_dist=lqs_dist, # non-outlier distances
              lqs_norm=lqs_norm, # coefficients
              lqs_intercept=lqs_intercept, # intercept
              lqs_rq=lqs_rq # q^th percentile residual
              )
  )
}

  
run_mio <- function(dataloc, srcloc, fname, q, dep_var, formulation, timelimit, resloc, calc_lqs_beta, mosekloc, gurobiloc) {
  my_regexec <- regexec("m([0-9]+)n([0-9]+).+i([0-9]+).+csv", fname) # get m, n, i from the dataset name
  my_regmatch <- regmatches(fname, my_regexec)
  m <- as.numeric(my_regmatch[[1]][2])
  n <- as.numeric(my_regmatch[[1]][3])
  i <- as.numeric(my_regmatch[[1]][4])
  cat(fname)
  cat("\n", i, "\n")
  X <- read.csv(paste(dataloc,"/",fname,sep=""), header=FALSE) # read data

  lqs_time <- c(0,0,0)
  lqs_beta <- -10
  if (calc_lqs_beta == TRUE) {
      if (dep_var == TRUE) { # response is first variable
          if (q==1.0) {
            q<-nrow(X)-1
          }
          lqs_time <- proc.time()
          my_lqs <- lqs(V1 ~ ., data=X, method="lqs", quantile=floor(q*nrow(X))) # fit LQS model for response; the first variable
          lqs_norm <- my_lqs$coefficients[2:length(my_lqs$coefficients)] # get coefficients but not intercept
          lqs_norm[is.na(lqs_norm)] <- 0.0
          lqs_intercept <- my_lqs$coefficients[1] # get intercept
          lqs_beta <- matrix(c(-1.0, lqs_intercept, lqs_norm),
                             nrow=length(lqs_norm)+2,
                             ncol=1) # beta_1 = -1 - the response
          lqs_time <- proc.time() - lqs_time
      } else {
          lqs_dist_min <- 10^8
          lqs_time <- proc.time()
          lqs_results <- lapply(1:n, fit_lqs, X, m, n, q)  # apply LQS with each variable 1...n as the response
          lqs_dists <- sapply(lqs_results, function(x) x$lqs_dist) # get all the distances
          lqs_dist <- min(lqs_dists) # get the best one

          lqs_norms <- sapply(lqs_results, function(x) x$lqs_norm) # get all the coefficients
          lqs_intercepts <- sapply(lqs_results, function(x) x$lqs_intercept) # get the intercepts
          lqs_norm <- lqs_norms[,which(lqs_dists == min(lqs_dists))] # get the coefficients of the best hyperplane
          lqs_intercept <- lqs_intercepts[which(lqs_dists == min(lqs_dists))] # get the intercept of the best hyperplane
          lqs_beta <- matrix(c(lqs_intercept, lqs_norm),
                             nrow=length(lqs_norm)+1,
                             ncol=1) # put intercept and coefficients together
          lqs_beta <- (lqs_beta/lqs_beta[1,1])*n  # normalize so that beta_0 = n
          lqs_time <- proc.time() - lqs_time
      }
  }

  if (class(q) == "numeric") {
    q <- floor(q*nrow(X)) # Convert q from percentile to order statistic for MATLAB implementations
  }
  # check reticulate for calling Python
  make_lqs_beta <- rmat_to_matlab_mat(lqs_beta, matname="lqs_beta") # create the warm start from LQS
  #add_path <- paste("addpath('", srcloc, "');", sep="") 
  add_path <- paste("addpath('", srcloc, "','", mosekloc, "','", gurobiloc,"');", sep="") # add the path to the MATLAB files and MOSEK
  if (formulation == "alg3-mio-bm" | formulation == "alg3-mio1" | formulation == "lqs-mio-bm" | formulation == "lqs-mio1" | formulation == "mio-bm" | formulation == "mio1" | formulation == "mio1-first" | formulation == "mio-bm-first") { # use the lqs solution from above or mio.m will call algorithm3.  The time used for lqs is subtracted from the time limit specified.
    if (dep_var == TRUE) { # first variable is response 
      print(paste(add_path, " ", make_lqs_beta, " ", "mio(", i, ",'",dataloc, "/", fname,"',",q,",lqs_beta,", m, ",true,'", formulation, "','", resloc, "',", timelimit-lqs_time[3], ")", sep="")) 
      run_matlab_code(paste(add_path, " ", make_lqs_beta, " ", "mio(", i, ",'",dataloc, "/", fname,"',",q,",lqs_beta,", m, ",true,'", formulation, "','", resloc, "',", timelimit-lqs_time[3], ")", sep="")) # dep_var = TRUE
    } else {
      run_matlab_code(paste(add_path, " ", make_lqs_beta, " ", "mio(", i, ",'",dataloc, "/", fname,"',",q,",lqs_beta,", m, ",false,'", formulation, "','", resloc, "',", timelimit-lqs_time[3], ")", sep="")) # dep_var = FALSE
    }
  } else if (formulation == "cbq-mio1" | formulation == "cbq-mio-bm") { # call run_cbqmio() in run_cbqmio.m
    if (dep_var == TRUE) {
      run_matlab_code(paste(add_path, " ", make_lqs_beta, " ", "run_cbqmio(", i, ",'",dataloc, "/", fname,"',",q,",", m, ",true,'", formulation, "','", resloc, "',", timelimit, ")", sep="")) # dep_var = TRUE
    } else{
      run_matlab_code(paste(add_path, " ", make_lqs_beta, " ", "run_cbqmio(", i, ",'",dataloc, "/", fname,"',",q,",", m, ",false,'", formulation, "','", resloc, "',", timelimit, ")", sep="")) # dep_var = FALSE
    }
  }
}


